#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Замер Tier-1 против VLM-fallback на подвыборке near-duplicate сэмплов: точность, латентность VLM, доля деградаций
# SCOPE: parse_args, main
# DEPENDS: numpy, PIL, packages.common.catalog, packages.common.config, packages.common.logging,
#          packages.common.llm.router, packages.common.near_dup, scripts.eval.augment, services.matcher.app.merge, services.normalize.app.detect,
#          services.normalize.app.tta, services.retrieval.app.encoder, services.retrieval.app.geo,
#          services.retrieval.app.head, services.retrieval.app.index, services.retrieval.app.search, services.vlm.app.extract
# LINKS: M-VLM, M-MATCHER, M-EVAL
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# parse_args - разбор аргументов CLI
# main - прогон Tier-1 и VLM-fallback на подвыборке, запись отчёта
# END_MODULE_MAP
"""Замер VLM-fallback.

Выборка намеренно состоит из **near-duplicate** групп (у вина есть «сестра» в
каталоге): именно там Tier-1 ошибается, различая год и сладость, и именно там
поля этикетки от VLM должны помочь.

VLM вызывается только для неуверенных Tier-1 ответов — как в проде, поэтому
доля вызовов и доля деградаций видны напрямую.
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

from packages.common.catalog import CatalogStore, load_catalog  # noqa: E402
from packages.common.config import settings  # noqa: E402
from packages.common.llm.router import LLMRouter  # noqa: E402
from packages.common.logging import log_marker  # noqa: E402
from packages.common.near_dup import group_key  # noqa: E402
from scripts.eval.augment import field_augment  # noqa: E402
from services.matcher.app.merge import rerank_candidates  # noqa: E402
from services.normalize.app.detect import detect_label  # noqa: E402
from services.normalize.app.tta import build_variants  # noqa: E402
from services.retrieval.app.encoder import ImageEncoder  # noqa: E402
from services.retrieval.app.geo import GeoIndex  # noqa: E402
from services.retrieval.app.head import load_head, read_head_meta  # noqa: E402
from services.retrieval.app.index import load_index  # noqa: E402
from services.retrieval.app.search import search_images  # noqa: E402
from services.vlm.app.extract import extract_fields  # noqa: E402

_DEFAULT_MATERIALS = (
    _ROOT.parent / "материалы для конкурса" / "Датасет"
    / "prod-svoe-vino-strapi" / "prod-svoe-vino" / "strapi" / "uploads"
)

__all__ = ["parse_args", "main"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы CLI."""
    parser = argparse.ArgumentParser(description="Замер Tier-1 против VLM-fallback")
    parser.add_argument("--index", default=str(_ROOT / "data" / "index"))
    parser.add_argument("--catalog", default=str(_ROOT / "data" / "catalog" / "catalog.jsonl"))
    parser.add_argument("--uploads", default=str(_DEFAULT_MATERIALS))
    parser.add_argument("--samples", type=int, default=20, help="сколько near-duplicate кейсов взять")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=str(_ROOT / "reports" / "vlm_fallback.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Прогоняет подвыборку и пишет отчёт сравнения."""
    args = parse_args(argv)
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
        key = group_key(record)
        group_sizes[key] = group_sizes.get(key, 0) + 1

    hard = [
        (slug, uploads / record["image"])
        for slug, record in sorted(catalog.items())
        if record.get("image") and (uploads / record["image"]).is_file()
        and group_sizes.get(group_key(record), 0) > 1
    ]
    if args.samples and args.samples < len(hard):
        picker = np.random.default_rng(args.seed)
        chosen = picker.choice(len(hard), size=args.samples, replace=False)
        sample = [hard[int(i)] for i in np.sort(chosen)]
    else:
        sample = hard
    log_marker("M-EVAL", "vlm_eval", "START", near_duplicate=len(hard), sample=len(sample))

    router = LLMRouter()
    rng = np.random.default_rng(args.seed + 1)
    rows: list[dict] = []
    vlm_latencies: list[float] = []
    vlm_calls = 0
    degradations = 0

    for position, (slug, path) in enumerate(sample, 1):
        try:
            with Image.open(path) as raw:
                rgba = np.asarray(raw.convert("RGBA"))
        except Exception:  # noqa: BLE001
            continue
        augmented = field_augment(rgba, rng)
        detection = detect_label(augmented)
        query_image = build_variants(augmented, detection.quad)[0]

        tier_started = time.perf_counter()
        hits = search_images(encoder, index, store, [query_image], k=settings.top_k, head=head,
                             geo_index=geo_index, geo_weight=settings.geo_weight,
                             geo_topk=settings.geo_topk, geo_min_matches=settings.geo_min_matches)
        tier_ms = (time.perf_counter() - tier_started) * 1000.0
        if not hits:
            continue
        tier_top1 = hits[0].slug
        tier_top5 = [hit.slug for hit in hits[:5]]
        margin = hits[0].score - (hits[1].score if len(hits) > 1 else 0.0)
        confident = hits[0].score >= settings.score_tau and margin >= settings.margin_delta

        final_top1, final_top5 = tier_top1, tier_top5
        vlm_ms = 0.0
        vlm_ok = False
        if not confident and settings.vlm_fallback_enabled:
            vlm_calls += 1
            started = time.perf_counter()
            fields = extract_fields(
                query_image, router,
                model=settings.vlm_online_model or None,
                max_tokens=settings.vlm_max_tokens,
                timeout_s=settings.vlm_timeout_s,
                extra={"reasoning_effort": settings.vlm_reasoning_effort},
            )
            vlm_ms = (time.perf_counter() - started) * 1000.0
            vlm_latencies.append(vlm_ms)
            if fields.ok:
                vlm_ok = True
                reranked = rerank_candidates(
                    [{"slug": hit.slug, "score": hit.score} for hit in hits],
                    fields.as_dict(), store,
                    text_weight=settings.matcher_text_weight,
                    vintage_bonus=settings.matcher_vintage_bonus,
                    sweetness_bonus=settings.matcher_sweetness_bonus,
                )
                if reranked:
                    final_top1 = reranked[0].slug
                    final_top5 = [item.slug for item in reranked[:5]]
            else:
                degradations += 1

        rows.append({
            "slug": slug,
            "tier_top1": tier_top1,
            "tier_correct": tier_top1 == slug,
            "tier_top5_correct": slug in tier_top5,
            "confident": confident,
            "vlm_called": not confident and settings.vlm_fallback_enabled,
            "vlm_ok": vlm_ok,
            "final_top1": final_top1,
            "final_correct": final_top1 == slug,
            "final_top5_correct": slug in final_top5,
            "tier_ms": round(tier_ms, 1),
            "vlm_ms": round(vlm_ms, 1),
        })
        if position % 10 == 0:
            print(f"  evaluated {position}/{len(sample)}", file=sys.stderr)

    total = len(rows)
    tier_acc1 = sum(1 for row in rows if row["tier_correct"])
    final_acc1 = sum(1 for row in rows if row["final_correct"])
    tier_acc5 = sum(1 for row in rows if row["tier_top5_correct"])
    final_acc5 = sum(1 for row in rows if row["final_top5_correct"])
    report = {
        "samples": total,
        "seed": args.seed,
        "selection": "near-duplicate groups (same winery + name without digits)",
        "samples_total_hard": len(hard),
        "vlm_model": settings.vlm_online_model,
        "vlm_max_tokens": settings.vlm_max_tokens,
        "vlm_timeout_s": settings.vlm_timeout_s,
        "vlm_reasoning_effort": settings.vlm_reasoning_effort,
        "tier1": {"accuracy@1": round(tier_acc1 / total, 4) if total else 0.0,
                  "accuracy@5": round(tier_acc5 / total, 4) if total else 0.0},
        "with_vlm": {"accuracy@1": round(final_acc1 / total, 4) if total else 0.0,
                     "accuracy@5": round(final_acc5 / total, 4) if total else 0.0},
        "vlm_calls": vlm_calls,
        "vlm_call_share": round(vlm_calls / total, 4) if total else 0.0,
        "degradations": degradations,
        "degradation_share": round(degradations / vlm_calls, 4) if vlm_calls else 0.0,
        "vlm_latency_ms": {
            "p50": round(float(np.percentile(vlm_latencies, 50)), 1) if vlm_latencies else 0.0,
            "p95": round(float(np.percentile(vlm_latencies, 95)), 1) if vlm_latencies else 0.0,
            "max": round(max(vlm_latencies), 1) if vlm_latencies else 0.0,
        },
        "rows": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log_marker("M-EVAL", "vlm_eval", "DONE", n=total, tier_acc1=report["tier1"]["accuracy@1"],
               vlm_acc1=report["with_vlm"]["accuracy@1"], vlm_calls=vlm_calls,
               degradations=degradations)

    print("=== VLM FALLBACK REPORT ===")
    print(f"samples (near-duplicate): {total} из {len(hard)} (seed {args.seed})")
    print(f"vlm model:                {settings.vlm_online_model} "
          f"(max_tokens {settings.vlm_max_tokens}, timeout {settings.vlm_timeout_s}s, "
          f"reasoning_effort={settings.vlm_reasoning_effort})")
    print(f"Tier-1  acc@1 {report['tier1']['accuracy@1']}  acc@5 {report['tier1']['accuracy@5']}")
    print(f"+VLM    acc@1 {report['with_vlm']['accuracy@1']}  acc@5 {report['with_vlm']['accuracy@5']}")
    print(f"vlm calls: {vlm_calls} ({report['vlm_call_share'] * 100:.0f}%), "
          f"degradations: {degradations} ({report['degradation_share'] * 100:.0f}% of calls)")
    print(f"vlm latency: p50 {report['vlm_latency_ms']['p50']}ms  "
          f"p95 {report['vlm_latency_ms']['p95']}ms  max {report['vlm_latency_ms']['max']}ms")
    print(f"out: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
