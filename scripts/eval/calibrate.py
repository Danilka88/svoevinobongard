#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Калибровка порогов решения и веса геоверификации на proxy-наборе: SCORE_TAU, MARGIN_DELTA, CONFIDENCE_TEMPERATURE, GEO_WEIGHT
# SCOPE: parse_args, main
# DEPENDS: numpy, PIL, packages.common.catalog, packages.common.config, packages.common.images, packages.common.logging,
#          scripts.eval.augment, services.normalize.app.detect, services.normalize.app.tta,
#          services.retrieval.app.encoder, services.retrieval.app.geo, services.retrieval.app.head,
#          services.retrieval.app.index, services.retrieval.app.search
# LINKS: M-EVAL, M-RETRIEVAL
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# parse_args - разбор аргументов CLI
# main - сеточный поиск веса геоверификации и порогов, запись отчёта калибровки
# END_MODULE_MAP
"""Калибровка решения (C-003).

Что подбирается:
- **GEO_WEIGHT** — вклад геоверификации: максимум acc@1 без потери acc@5.
- **SCORE_TAU** — порог итогового скора, максимизирующий F1 разделения верных и неверных ответов.
- **MARGIN_DELTA** — порог отрыва top1-top2 по тому же критерию.
- **CONFIDENCE_TEMPERATURE** — температура softmax, дающая наиболее честную уверенность.

Все замеры идут на том же proxy-наборе (seed, samples), что и baseline_report,
поэтому значения переносимы в .env.example без пересчёта.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from packages.common.catalog import CatalogStore, load_catalog  # noqa: E402
from packages.common.images import flatten_alpha  # noqa: E402
from packages.common.config import settings
from packages.common.logging import log_marker  # noqa: E402
from scripts.eval.augment import field_augment  # noqa: E402
from services.normalize.app.detect import detect_label  # noqa: E402
from services.normalize.app.tta import build_variants  # noqa: E402
from services.retrieval.app.encoder import ImageEncoder  # noqa: E402
from services.retrieval.app.geo import GeoIndex, extract_features  # noqa: E402
from services.retrieval.app.head import load_head, read_head_meta  # noqa: E402
from services.retrieval.app.index import load_index  # noqa: E402
from services.retrieval.app.search import search_images  # noqa: E402

_DEFAULT_MATERIALS = (
    _ROOT.parent / "материалы для конкурса" / "Датасет"
    / "prod-svoe-vino-strapi" / "prod-svoe-vino" / "strapi" / "uploads"
)
_MAX_SIDE = 512
_WEIGHT_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
_TEMPERATURE_GRID = [0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3]

__all__ = ["parse_args", "main"]


def _resize_bgr(array: np.ndarray, max_side: int = _MAX_SIDE) -> np.ndarray:
    """Пропорционально уменьшает BGR-изображение."""
    import cv2

    height, width = array.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return array
    scale = max_side / float(longest)
    return cv2.resize(array, (max(1, int(width * scale)), max(1, int(height * scale))),
                      interpolation=cv2.INTER_AREA)


def _f1_at_threshold(scores: np.ndarray, correct: np.ndarray, threshold: float) -> float:
    """F1 разделения верных и неверных ответов порогом по скору."""
    predicted = scores >= threshold
    tp = int(np.sum(predicted & correct))
    fp = int(np.sum(predicted & ~correct))
    fn = int(np.sum(~predicted & correct))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def _best_threshold(scores: np.ndarray, correct: np.ndarray) -> tuple[float, float]:
    """Подбирает порог, максимизирующий F1 разделения."""
    if scores.size == 0:
        return 0.0, 0.0
    candidates = np.unique(np.round(scores, 4))
    best_threshold, best_f1 = float(candidates[0]), 0.0
    for threshold in candidates:
        score = _f1_at_threshold(scores, correct, float(threshold))
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), score
    return best_threshold, round(best_f1, 4)


def _softmax_top1(scores: list[float], temperature: float) -> float:
    """Softmax-вероятность top-1 по top-K."""
    if not scores:
        return 0.0
    scaled = np.asarray(scores, dtype=np.float64) / max(temperature, 1e-6)
    scaled -= scaled.max()
    weights = np.exp(scaled)
    return float(weights[0] / float(weights.sum()))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы CLI."""
    parser = argparse.ArgumentParser(description="Калибровка порогов и веса геоверификации")
    parser.add_argument("--index", default=str(_ROOT / "data" / "index"))
    parser.add_argument("--catalog", default=str(_ROOT / "data" / "catalog" / "catalog.jsonl"))
    parser.add_argument("--uploads", default=str(_DEFAULT_MATERIALS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--geo-topk", type=int, default=10)
    parser.add_argument("--geo-min-matches", type=int, default=8)
    parser.add_argument("--out", default=str(_ROOT / "reports" / "calibration.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Собирает данные прогона и подбирает вес геоверификации и пороги решения."""
    args = parse_args(argv)
    index_dir = Path(args.index)
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    index = load_index(index_dir, backend="numpy")
    encoder = ImageEncoder(meta["model"], device=args.device)
    head_payload = read_head_meta(meta)
    head = load_head(index_dir / head_payload["file"], device="cpu") if head_payload else None
    geo_index = GeoIndex.load(index_dir, meta)
    if geo_index is None:
        print("ERROR: features.npz not found; rebuild index with --features", file=sys.stderr)
        return 2

    catalog = load_catalog(args.catalog)
    store = CatalogStore(catalog)
    uploads = Path(args.uploads)
    available = [
        (slug, uploads / record["image"])
        for slug, record in sorted(catalog.items())
        if record.get("image") and (uploads / record["image"]).is_file()
    ]
    if args.samples and args.samples < len(available):
        picker = np.random.default_rng(args.seed)
        chosen = picker.choice(len(available), size=args.samples, replace=False)
        sample = [available[int(i)] for i in np.sort(chosen)]
    else:
        sample = available

    gold: list[str] = []
    retrieval_scores: list[list[float]] = []
    geo_scores: list[list[float]] = []
    slugs: list[list[str]] = []
    rng = np.random.default_rng(args.seed + 1)
    log_marker("M-EVAL", "calibrate", "START", samples=len(sample), geo_topk=args.geo_topk)

    for position, (slug, path) in enumerate(sample, 1):
        try:
            with Image.open(path) as raw:
                rgba = np.asarray(raw.convert("RGBA"))
        except Exception:  # noqa: BLE001
            continue
        augmented = field_augment(rgba, rng)
        query_image = augmented
        if args.normalize:
            detection = detect_label(augmented)
            query_image = build_variants(augmented, detection.quad)[0]

        hits = search_images(encoder, index, store, [query_image], k=args.geo_topk, head=head)
        if not hits:
            continue
        query_features = extract_features(query_image, method=geo_index.method)
        gold.append(slug)
        retrieval_scores.append([float(hit.score) for hit in hits])
        slugs.append([hit.slug for hit in hits])
        geo_scores.append([
            geo_index.verify_slug(hit.slug, query_features, min_matches=args.geo_min_matches).score
            for hit in hits
        ])
        if position % 50 == 0:
            print(f"  collected {position}/{len(sample)}", file=sys.stderr)

    total = len(gold)
    if total == 0:
        print("ERROR: no samples evaluated", file=sys.stderr)
        return 2

    weight_table: list[dict] = []
    best = None
    for weight in _WEIGHT_GRID:
        correct1 = 0
        correct5 = 0
        for row, slug in enumerate(gold):
            fused = [retrieval_scores[row][i] + weight * geo_scores[row][i]
                     for i in range(len(slugs[row]))]
            order = sorted(range(len(fused)), key=lambda i: fused[i], reverse=True)
            if slugs[row][order[0]] == slug:
                correct1 += 1
            if slug in [slugs[row][i] for i in order[:5]]:
                correct5 += 1
        entry = {"geo_weight": weight, "accuracy@1": round(correct1 / total, 4),
                 "accuracy@5": round(correct5 / total, 4)}
        weight_table.append(entry)
        if best is None or (entry["accuracy@1"], entry["accuracy@5"], -weight) > (
                best["accuracy@1"], best["accuracy@5"], -best["geo_weight"]):
            best = entry
    best_weight = float(best["geo_weight"])

    top1_scores = np.zeros(total, dtype=np.float64)
    margins = np.zeros(total, dtype=np.float64)
    correct = np.zeros(total, dtype=bool)
    top5_scores: list[list[float]] = []
    for row, slug in enumerate(gold):
        fused = [retrieval_scores[row][i] + best_weight * geo_scores[row][i]
                 for i in range(len(slugs[row]))]
        order = sorted(range(len(fused)), key=lambda i: fused[i], reverse=True)
        ranked_scores = [fused[i] for i in order]
        top1_scores[row] = ranked_scores[0]
        margins[row] = ranked_scores[0] - (ranked_scores[1] if len(ranked_scores) > 1 else 0.0)
        correct[row] = slugs[row][order[0]] == slug
        top5_scores.append(ranked_scores[:5])

    tau, tau_f1 = _best_threshold(top1_scores, correct)
    delta, delta_f1 = _best_threshold(margins, correct)

    temperature_table: list[dict] = []
    best_temperature = (_TEMPERATURE_GRID[0], 0.0, 0.0)
    for temperature in _TEMPERATURE_GRID:
        confidences = np.array([_softmax_top1(row, temperature) for row in top5_scores])
        threshold, f1 = _best_threshold(confidences, correct)
        temperature_table.append({"temperature": temperature, "confidence_threshold": threshold,
                                  "separation_f1": f1})
        if f1 > best_temperature[2]:
            best_temperature = (temperature, threshold, f1)

    # C-021: метрики действующих порогов — иначе нельзя доказать, что новые лучше
    current_tau_f1 = _f1_at_threshold(top1_scores, correct, float(settings.score_tau))
    current_delta_f1 = _f1_at_threshold(margins, correct, float(settings.margin_delta))
    current_confidences = np.array(
        [_softmax_top1(row, float(settings.confidence_temperature)) for row in top5_scores])
    current_temp_f1 = _best_threshold(current_confidences, correct)[1]
    current_weight_row = next(
        (row for row in weight_table if abs(float(row["geo_weight"]) - float(settings.geo_weight)) < 1e-9),
        None)

    report = {
        "encoder": meta.get("encoder"),
        "model": meta.get("model"),
        "index_dim": meta.get("dim"),
        "head": meta.get("head") or "none",
        "geo_method": (meta.get("features") or {}).get("method"),
        "samples": total,
        "seed": args.seed,
        "normalize_applied": bool(args.normalize),
        "weight_grid": weight_table,
        "chosen": {
            "geo_weight": best_weight,
            "score_tau": round(float(tau), 4),
            "score_tau_separation_f1": tau_f1,
            "margin_delta": round(float(delta), 4),
            "margin_delta_separation_f1": delta_f1,
            "confidence_temperature": best_temperature[0],
            "confidence_threshold": round(float(best_temperature[1]), 4),
            "confidence_separation_f1": best_temperature[2],
        },
        "temperature_grid": temperature_table,
        "current_settings": {
            "geo_weight": float(settings.geo_weight),
            "accuracy@1": (current_weight_row or {}).get("accuracy@1"),
            "accuracy@5": (current_weight_row or {}).get("accuracy@5"),
            "score_tau": float(settings.score_tau),
            "score_tau_separation_f1": current_tau_f1,
            "margin_delta": float(settings.margin_delta),
            "margin_delta_separation_f1": current_delta_f1,
            "confidence_temperature": float(settings.confidence_temperature),
            "confidence_separation_f1": current_temp_f1,
        },
        "baselines": {
            "accuracy@1_at_best_weight": best["accuracy@1"],
            "accuracy@5_at_best_weight": best["accuracy@5"],
            "accuracy@1_retrieval_only": weight_table[0]["accuracy@1"],
            "accuracy@5_retrieval_only": weight_table[0]["accuracy@5"],
            "mean_margin": round(float(np.mean(margins)), 4),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log_marker("M-EVAL", "calibrate", "DONE", geo_weight=best_weight,
               tau=report["chosen"]["score_tau"], delta=report["chosen"]["margin_delta"],
               temperature=best_temperature[0])

    print("=== CALIBRATION REPORT ===")
    print(f"samples:            {total} (seed {args.seed}, normalize={bool(args.normalize)})")
    print(f"geo method:         {report['geo_method']}")
    print("geo_weight  acc@1    acc@5")
    for entry in weight_table:
        print(f"  {entry['geo_weight']:<9} {entry['accuracy@1']:<8} {entry['accuracy@5']}")
    print(f"chosen geo_weight:  {best_weight}")
    print(f"SCORE_TAU:          {report['chosen']['score_tau']} (separation F1 {tau_f1})")
    print(f"MARGIN_DELTA:       {report['chosen']['margin_delta']} (separation F1 {delta_f1})")
    print(f"CONFIDENCE_TEMP:    {best_temperature[0]} (separation F1 {best_temperature[2]})")
    print(f"out:                {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
