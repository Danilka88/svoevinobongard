#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Baseline-отчёт Tier-1: self-retrieval санити-чек и proxy F1@1/@5 на синтетических полевых фото
# SCOPE: parse_args, main
# DEPENDS: numpy, cv2, PIL, packages.common.catalog, services.normalize.app.detect, services.normalize.app.tta, services.retrieval.app.encoder, services.retrieval.app.index, scripts.eval.augment
# LINKS: M-EVAL, M-RETRIEVAL
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# parse_args - разбор аргументов CLI
# main - прогон self-retrieval и synthetic proxy-метрик, запись отчёта
# END_MODULE_MAP
"""Baseline-отчёт качества Tier-1.

Что измеряется:
1. **self-retrieval** — контроль исправности индекса: эталон должен находить себя
   (ожидание accuracy@1 = 1.0, score ≈ 1.0).
2. **synthetic proxy** — эталон аугментируется в «полевое» фото и ищется в индексе:
   честная proxy F1@1/@5 для доменного разрыва «вырез → полевое фото».

Ground truth полевых фото у команды нет, поэтому реальные F1@1/@5 на
``data/samples/field`` не считаются — только кандидаты и latency.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from packages.common.catalog import load_catalog  # noqa: E402
from packages.common.images import flatten_alpha  # noqa: E402
from packages.common.logging import log_marker  # noqa: E402
from scripts.eval.augment import field_augment  # noqa: E402
from services.normalize.app.detect import detect_label  # noqa: E402
from services.normalize.app.tta import build_variants  # noqa: E402
from services.retrieval.app.encoder import ImageEncoder  # noqa: E402
from services.retrieval.app.index import load_index  # noqa: E402

__all__ = [
    "parse_args",
    "main",
]

_DEFAULT_MATERIALS = (
    _ROOT.parent / "материалы для конкурса" / "Датасет"
    / "prod-svoe-vino-strapi" / "prod-svoe-vino" / "strapi" / "uploads"
)
_MAX_SIDE = 512
_TOP_K = 5


def _resize_bgr(array: np.ndarray, max_side: int = _MAX_SIDE) -> np.ndarray:
    """Пропорционально уменьшает BGR-изображение по длинной стороне."""
    import cv2

    height, width = array.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return array
    scale = max_side / float(longest)
    return cv2.resize(array, (max(1, int(width * scale)), max(1, int(height * scale))),
                      interpolation=cv2.INTER_AREA)


def _load_reference_rgba(path: Path) -> np.ndarray | None:
    """Читает эталон как RGBA-массив."""
    try:
        with Image.open(path) as raw:
            return np.asarray(raw.convert("RGBA"))
    except Exception:  # noqa: BLE001 - битый файл пропускаем
        return None


def _macro_f1(pairs: list[tuple[str, str]]) -> float:
    """Считает macro-F1 по парам (gold, predicted)."""
    labels = {gold for gold, _ in pairs} | {pred for _, pred in pairs}
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        tp = sum(1 for gold, pred in pairs if gold == label and pred == label)
        fp = sum(1 for gold, pred in pairs if gold != label and pred == label)
        fn = sum(1 for gold, pred in pairs if gold == label and pred != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return sum(scores) / len(scores)


def _ratio(numerator: int, total: int) -> float:
    """Безопасная доля с округлением до 4 знаков."""
    return round(numerator / total, 4) if total else 0.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы CLI."""
    parser = argparse.ArgumentParser(description="Baseline-отчёт Tier-1 retrieval")
    parser.add_argument("--index", default=str(_ROOT / "data" / "index"))
    parser.add_argument("--catalog", default=str(_ROOT / "data" / "catalog" / "catalog.jsonl"))
    parser.add_argument("--uploads", default=str(_DEFAULT_MATERIALS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize", action="store_true",
                        help="прогнать синтетический запрос через normalize (детекция+кроп) до retrieval")
    parser.add_argument("--out", default=str(_ROOT / "reports" / "baseline.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Считает self-retrieval и synthetic proxy-метрики, пишет JSON-отчёт."""
    args = parse_args(argv)
    index_dir = Path(args.index)
    meta_path = index_dir / "meta.json"
    if not meta_path.is_file():
        print(f"ERROR: index meta not found: {meta_path}", file=sys.stderr)
        return 2
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    index = load_index(index_dir, backend="numpy")
    encoder = ImageEncoder(meta["model"], device=args.device)
    log_marker("M-EVAL", "baseline", "START", index=len(index.slugs), model=meta["model"],
               samples=args.samples)

    catalog = load_catalog(args.catalog)
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

    self_hits: list[str] = []
    self_scores: list[float] = []
    synthetic_gold: list[str] = []
    synthetic_top1: list[str] = []
    synthetic_top5: list[list[str]] = []
    latencies: list[float] = []
    misses: list[dict] = []
    rng = np.random.default_rng(args.seed + 1)

    for position, (slug, path) in enumerate(sample, 1):
        rgba = _load_reference_rgba(path)
        if rgba is None:
            continue

        plain = _resize_bgr(flatten_alpha(rgba[:, :, [2, 1, 0, 3]]))
        hits = index.search(encoder.encode_arrays([plain])[0], k=_TOP_K)
        self_hits.append(hits[0][0] if hits else "")
        self_scores.append(float(hits[0][1]) if hits else 0.0)

        augmented = field_augment(rgba, rng)
        query_image = augmented
        if args.normalize:
            detection = detect_label(augmented)
            query_image = build_variants(augmented, detection.quad)[0]
        started = time.perf_counter()
        vector = encoder.encode_arrays([query_image])[0]
        hits = index.search(vector, k=_TOP_K)
        latencies.append((time.perf_counter() - started) * 1000.0)

        ranked = [item[0] for item in hits]
        synthetic_gold.append(slug)
        synthetic_top1.append(ranked[0] if ranked else "")
        synthetic_top5.append(ranked)
        if not ranked or ranked[0] != slug:
            misses.append({"slug": slug, "predicted": ranked[0] if ranked else None, "top5": ranked})

        if position % 50 == 0:
            print(f"  evaluated {position}/{len(sample)}", file=sys.stderr)

    total = len(synthetic_gold)
    pairs = list(zip(synthetic_gold, synthetic_top1))
    self_gold = [slug for slug, _ in sample][: len(self_hits)]
    self_acc1 = _ratio(sum(1 for gold, top1 in zip(self_gold, self_hits) if gold == top1), len(self_hits))
    report = {
        "encoder": meta.get("encoder"),
        "model": meta.get("model"),
        "device": meta.get("device"),
        "index_size": len(index.slugs),
        "index_dim": meta.get("dim"),
        "vector_backend": meta.get("backend"),
        "samples": total,
        "seed": args.seed,
        "normalize_applied": bool(args.normalize),
        "self_retrieval": {
            "accuracy@1": self_acc1,
            "mean_top1_score": round(float(np.mean(self_scores)), 4) if self_scores else 0.0,
        },
        "synthetic_field": {
            "accuracy@1": _ratio(sum(1 for gold, pred in pairs if gold == pred), total),
            "accuracy@5": _ratio(sum(1 for gold, ranked in zip(synthetic_gold, synthetic_top5) if gold in ranked), total),
            "f1_macro@1": round(_macro_f1(pairs), 4),
            "latency_ms": {
                "p50": round(float(np.percentile(latencies, 50)), 1) if latencies else 0.0,
                "p95": round(float(np.percentile(latencies, 95)), 1) if latencies else 0.0,
            },
            "misses": len(misses),
            "miss_examples": misses[:10],
        },
        "note": ("Ground truth для data/samples/field отсутствует, поэтому реальные F1@1/@5 "
                 "не считаются; synthetic_field — proxy-оценка доменного разрыва."),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log_marker("M-EVAL", "baseline", "DONE", self_acc1=self_acc1,
               syn_acc1=report["synthetic_field"]["accuracy@1"],
               syn_acc5=report["synthetic_field"]["accuracy@5"])

    print("=== BASELINE REPORT ===")
    print(f"encoder:            {report['encoder']} ({report['model']}) on {report['device']}")
    print(f"index:              {report['index_size']} vectors, dim {report['index_dim']}, {report['vector_backend']}")
    print(f"samples:            {report['samples']} (seed {args.seed}, normalize={report['normalize_applied']})")
    print(f"self acc@1:         {report['self_retrieval']['accuracy@1']}  (mean score {report['self_retrieval']['mean_top1_score']})")
    print(f"synthetic acc@1:    {report['synthetic_field']['accuracy@1']}")
    print(f"synthetic acc@5:    {report['synthetic_field']['accuracy@5']}")
    print(f"synthetic f1_macro: {report['synthetic_field']['f1_macro@1']}")
    print(f"synthetic latency:  p50 {report['synthetic_field']['latency_ms']['p50']}ms  p95 {report['synthetic_field']['latency_ms']['p95']}ms")
    print(f"misses:             {report['synthetic_field']['misses']}")
    print(f"out:                {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
