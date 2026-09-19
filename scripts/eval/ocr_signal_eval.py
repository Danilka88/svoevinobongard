#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Замер OCR-сигнала: Tier-1+geo против +OCR на proxy и near-duplicate подвыборке, бюджет времени и чтение года
# SCOPE: parse_args, main
# DEPENDS: numpy, PIL, packages.common.catalog, packages.common.config, packages.common.images, packages.common.logging,
#          packages.common.ocr, scripts.eval.augment, services.matcher.app.merge, services.normalize.app.detect,
#          services.normalize.app.tta, services.retrieval.app.encoder, services.retrieval.app.geo,
#          services.retrieval.app.head, services.retrieval.app.index, services.retrieval.app.search
# LINKS: M-OCR, M-MATCHER, M-EVAL
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# parse_args - разбор аргументов CLI
# main - прогон Tier-1 и OCR-сигнала на proxy и near-duplicate подвыборке, запись отчёта
# END_MODULE_MAP
"""Замер OCR-сигнала (C-010).

Сравнивает Tier-1 (retrieval + geo) с Tier-1 + OCR → matcher. OCR вызывается
только для неуверенных ответов — как в проде, поэтому видна доля вызовов и
вклад сигнала. Отдельно считается, как часто OCR прочитал год.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from packages.common.catalog import CatalogStore, load_catalog  # noqa: E402
from packages.common.config import settings  # noqa: E402
from packages.common.images import flatten_alpha  # noqa: E402
from packages.common.logging import log_marker  # noqa: E402
from packages.common.ocr import resolve_engine, run_ocr  # noqa: E402
from scripts.eval.augment import field_augment  # noqa: E402
from services.matcher.app.merge import rerank_candidates  # noqa: E402
from services.normalize.app.detect import detect_label  # noqa: E402
from services.normalize.app.tta import build_variants  # noqa: E402
from services.retrieval.app.encoder import ImageEncoder  # noqa: E402
from services.retrieval.app.geo import GeoIndex  # noqa: E402
from services.retrieval.app.head import load_head, read_head_meta  # noqa: E402
from services.retrieval.app.index import load_index  # noqa: E402
from services.retrieval.app.search import search_images  # noqa: E402

_DEFAULT_MATERIALS = (
    _ROOT.parent / "материалы для конкурса" / "Датасет"
    / "prod-svoe-vino-strapi" / "prod-svoe-vino" / "strapi" / "uploads"
)
_MAX_SIDE = 512

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


def _group_key(record: dict) -> str:
    """Ключ группы near-duplicates."""
    winery = (record.get("winery") or "").strip().lower()
    name = " ".join(re.sub(r"\d+", " ", (record.get("name") or "").lower()).split())[:48]
    return f"{winery}|{name}"


def _truth_year(record: dict) -> str:
    """Год из метаданных каталога (поле year, название или слаг)."""
    if record.get("year"):
        return str(record["year"])
    match = re.search(r"\b((?:19|20)\d{2})\b", f"{record.get('name', '')} {record.get('slug', '')}")
    return match.group(1) if match else ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы CLI."""
    parser = argparse.ArgumentParser(description="Замер OCR-сигнала")
    parser.add_argument("--index", default=str(_ROOT / "data" / "index"))
    parser.add_argument("--catalog", default=str(_ROOT / "data" / "catalog" / "catalog.jsonl"))
    parser.add_argument("--uploads", default=str(_DEFAULT_MATERIALS))
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=str(_ROOT / "reports" / "ocr_signal.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Прогоняет выборку и пишет отчёт сравнения Tier-1 и +OCR."""
    args = parse_args(argv)
    engine = resolve_engine()
    if engine == "none":
        print("ERROR: no OCR engine available", file=sys.stderr)
        return 2

    index_dir = Path(args.index)
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    index = load_index(index_dir, backend="numpy")
    encoder = ImageEncoder(meta["model"], device=args.device)
    head_payload = read_head_meta(meta)
    head = load_head(index_dir / head_payload["file"], device="cpu") if head_payload else None
    geo_index = GeoIndex.load(index_dir, meta)

    catalog = load_catalog(args.catalog)
    store = CatalogStore(catalog)
    uploads = Path(args.uploads)
    group_sizes: dict[str, int] = {}
    for record in catalog.values():
        key = _group_key(record)
        group_sizes[key] = group_sizes.get(key, 0) + 1

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
    log_marker("M-EVAL", "ocr_eval", "START", engine=engine, samples=len(sample))

    rng = np.random.default_rng(args.seed + 1)
    rows: list[dict] = []
    ocr_latencies: list[float] = []
    ocr_calls = 0

    for position, (slug, path) in enumerate(sample, 1):
        try:
            with Image.open(path) as raw:
                rgba = np.asarray(raw.convert("RGBA"))
        except Exception:  # noqa: BLE001
            continue
        augmented = field_augment(rgba, rng)
        detection = detect_label(augmented)
        query_image = build_variants(augmented, detection.quad)[0]

        hits = search_images(encoder, index, store, [query_image], k=settings.top_k, head=head,
                             geo_index=geo_index, geo_weight=settings.geo_weight,
                             geo_topk=settings.geo_topk, geo_min_matches=settings.geo_min_matches)
        if not hits:
            continue
        tier_top1 = hits[0].slug
        tier_top5 = [hit.slug for hit in hits[:5]]
        margin = hits[0].score - (hits[1].score if len(hits) > 1 else 0.0)
        confident = hits[0].score >= settings.score_tau and margin >= settings.margin_delta

        final_top1, final_top5 = tier_top1, tier_top5
        vintage_read = ""
        if not confident:
            ocr_calls += 1
            ocr = run_ocr(query_image, max_side=settings.ocr_max_side)
            ocr_latencies.append(ocr.latency_ms)
            vintage_read = ocr.vintage
            if ocr.ok and (ocr.vintage or ocr.sweetness):
                reranked = rerank_candidates(
                    [{"slug": hit.slug, "score": hit.score} for hit in hits],
                    ocr.as_fields(), store,
                    text_weight=settings.matcher_text_weight,
                    vintage_bonus=settings.matcher_vintage_bonus,
                    sweetness_bonus=settings.matcher_sweetness_bonus,
                )
                if reranked:
                    final_top1 = reranked[0].slug
                    final_top5 = [item.slug for item in reranked[:5]]

        record = catalog.get(slug) or {}
        truth_year = _truth_year(record)
        rows.append({
            "slug": slug,
            "group": _group_key(record),
            "group_size": group_sizes.get(_group_key(record), 1),
            "tier_top1": tier_top1,
            "tier_correct": tier_top1 == slug,
            "tier_top5_correct": slug in tier_top5,
            "confident": confident,
            "ocr_called": not confident,
            "vintage_read": vintage_read,
            "truth_year": truth_year,
            "vintage_ok": bool(vintage_read) and vintage_read == truth_year,
            "final_top1": final_top1,
            "final_correct": final_top1 == slug,
            "final_top5_correct": slug in final_top5,
        })
        if position % 50 == 0:
            print(f"  evaluated {position}/{len(sample)}", file=sys.stderr)

    total = len(rows)
    hard = [row for row in rows if row["group_size"] > 1]
    with_year = [row for row in rows if row["truth_year"]]
    read_ok = [row for row in with_year if row["vintage_ok"]]

    def _acc(items: list[dict], key: str) -> float:
        return round(sum(1 for row in items if row[key]) / len(items), 4) if items else 0.0

    report = {
        "engine": engine,
        "samples": total,
        "seed": args.seed,
        "ocr_calls": ocr_calls,
        "ocr_call_share": round(ocr_calls / total, 4) if total else 0.0,
        "tier1": {"accuracy@1": _acc(rows, "tier_correct"),
                  "accuracy@5": _acc(rows, "tier_top5_correct")},
        "with_ocr": {"accuracy@1": _acc(rows, "final_correct"),
                     "accuracy@5": _acc(rows, "final_top5_correct")},
        "near_duplicate": {
            "samples": len(hard),
            "tier1_accuracy@1": _acc(hard, "tier_correct"),
            "with_ocr_accuracy@1": _acc(hard, "final_correct"),
            "tier1_accuracy@5": _acc(hard, "tier_top5_correct"),
            "with_ocr_accuracy@5": _acc(hard, "final_top5_correct"),
        },
        "vintage_reading": {
            "samples_with_truth_year": len(with_year),
            "read_correctly": len(read_ok),
            "read_rate": round(len(read_ok) / len(with_year), 4) if with_year else 0.0,
        },
        "ocr_latency_ms": {
            "p50": round(float(np.percentile(ocr_latencies, 50)), 1) if ocr_latencies else 0.0,
            "p95": round(float(np.percentile(ocr_latencies, 95)), 1) if ocr_latencies else 0.0,
            "max": round(max(ocr_latencies), 1) if ocr_latencies else 0.0,
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({**report, "rows": rows}, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    log_marker("M-EVAL", "ocr_eval", "DONE", engine=engine, n=total,
               tier_acc1=report["tier1"]["accuracy@1"], ocr_acc1=report["with_ocr"]["accuracy@1"],
               read_rate=report["vintage_reading"]["read_rate"])

    print("=== OCR SIGNAL REPORT ===")
    print(f"engine:                 {engine}")
    print(f"samples:                {total} (seed {args.seed})")
    print(f"Tier-1+geo   acc@1 {report['tier1']['accuracy@1']}  acc@5 {report['tier1']['accuracy@5']}")
    print(f"+OCR         acc@1 {report['with_ocr']['accuracy@1']}  acc@5 {report['with_ocr']['accuracy@5']}")
    print(f"near-dup n={report['near_duplicate']['samples']}: "
          f"Tier-1 acc@1 {report['near_duplicate']['tier1_accuracy@1']} → "
          f"+OCR {report['near_duplicate']['with_ocr_accuracy@1']} (acc@5 "
          f"{report['near_duplicate']['tier1_accuracy@5']} → "
          f"{report['near_duplicate']['with_ocr_accuracy@5']})")
    print(f"OCR calls:              {ocr_calls} ({report['ocr_call_share'] * 100:.0f}%)")
    print(f"vintage read rate:      {report['vintage_reading']['read_rate']} "
          f"({report['vintage_reading']['read_correctly']}/{report['vintage_reading']['samples_with_truth_year']})")
    print(f"OCR latency:            p50 {report['ocr_latency_ms']['p50']}ms  "
          f"p95 {report['ocr_latency_ms']['p95']}ms  max {report['ocr_latency_ms']['max']}ms")
    print(f"out: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
