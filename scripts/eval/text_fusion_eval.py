#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Валидация текстового фьюжна C-013/C-014 поверх реального визуального top-K
# SCOPE: parse_args, main
# DEPENDS: numpy, PIL, rapidfuzz, packages.common.catalog, packages.common.config, packages.common.images,
#          packages.common.logging, packages.common.ocr, scripts.eval.augment, services.retrieval.app.encoder,
#          services.retrieval.app.index, services.retrieval.app.search, services.matcher.app.merge,
#          services.matcher.app.textfusion
# LINKS: M-EVAL, M-MATCHER, M-TEXT, M-RETRIEVAL, M-OCR
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# DEFAULT_MODES - режимы фьюжна, которые сравниваются на одном прогоне
# DEFAULT_SAMPLES - размер выборки валидации по умолчанию
# DEFAULT_TOPK - сколько кандидатов отдаёт визуальный поиск
# parse_args - аргументы CLI валидации
# main - измеряет acc@1/@5 и near-dup дизамбигуацию для off/always/conditional поверх визуальной выдачи
# END_MODULE_MAP
"""Валидация текстового фьюжна поверх реального визуального top-K.

Запрос — синтетическое «полевое» фото (`field_augment`) на фоне полки/витрины:
эталон, лежащий в индексе, дал бы тривиальный acc@1 и ничего не измерил.

Три режима на одном прогоне:
- `off` — только визуальный путь (как в проде, флаг выключен);
- `always` — безусловный текстовый фьюжн (C-013);
- `conditional` — триггер только если top-1 и top-2 визуальной выдачи принадлежат
  одной near-dup группе и их скоры близки.

Оговорка: домен — эталоны каталога + синтетическая аугментация, не полевые фото (C-011).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
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
from packages.common.ocr import run_ocr  # noqa: E402
from scripts.eval.augment import field_augment  # noqa: E402
from services.matcher.app.merge import rerank_candidates  # noqa: E402
from services.matcher.app.textfusion import load_enriched_catalog, near_dup_key, should_fuse  # noqa: E402
from services.retrieval.app.encoder import load_encoder  # noqa: E402
from services.retrieval.app.head import load_head  # noqa: E402
from services.retrieval.app.index import load_index  # noqa: E402
from services.retrieval.app.search import search_images  # noqa: E402

_DEFAULT_UPLOADS = (
    _ROOT.parent / "материалы для конкурса" / "Датасет"
    / "prod-svoe-vino-strapi" / "prod-svoe-vino" / "strapi" / "uploads"
)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
DEFAULT_SAMPLES = 150
DEFAULT_MODES = "off,always,conditional"
DEFAULT_TOPK = 30

__all__ = ["DEFAULT_MODES", "DEFAULT_SAMPLES", "DEFAULT_TOPK", "parse_args", "main"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы CLI."""
    parser = argparse.ArgumentParser(description="Валидация текстового фьюжна поверх визуального top-K")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--modes", default=DEFAULT_MODES)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(_ROOT / "reports" / "text_fusion.json"))
    parser.add_argument("--uploads", default=str(_DEFAULT_UPLOADS))
    parser.add_argument("--enriched", default="",
                        help="overlay обогащения; по умолчанию из настроек")
    return parser.parse_args(argv)


def _query_fields(image: np.ndarray) -> tuple[dict, float, str]:
    """Строит текстовый запрос из OCR-текста синтетического полевого фото."""
    start = time.perf_counter()
    ocr = run_ocr(image, max_side=settings.ocr_max_side)
    latency = (time.perf_counter() - start) * 1000.0
    lines = [line.strip() for line in ocr.text.splitlines() if line.strip()]
    candidates = [line for line in lines if not _YEAR_RE.search(line) and len(line) > 3]
    name_guess = max(candidates, key=len) if candidates else ""
    return {
        "winery": ocr.winery_hint or "",
        "wine_name": name_guess,
        "vintage": ocr.vintage or "",
        "wine_type": ocr.sweetness or "",
    }, latency, ocr.text


def _percentile(values: list[float], p: float) -> float:
    """p50/p95 без внешних зависимостей."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1))))
    return round(ordered[index], 1)


def main(argv: list[str] | None = None) -> int:
    """Прогоняет валидацию по всем режимам и пишет отчёт."""
    args = parse_args(argv)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    catalog = load_catalog(_ROOT / settings.catalog_jsonl)
    store = CatalogStore(catalog)
    enriched_path = Path(args.enriched) if args.enriched else (_ROOT / settings.catalog_enriched_jsonl)
    enriched = load_enriched_catalog(enriched_path)
    print(f"обогащение: {enriched_path.name} ({len(enriched)} slug)")
    uploads = Path(args.uploads)
    meta = json.loads((_ROOT / settings.index_dir / "meta.json").read_text(encoding="utf-8"))
    # энкодер обязан совпадать с тем, которым построен индекс (иначе 768 против 1152)
    encoder = load_encoder(encoder=meta.get("encoder"), model_id=meta.get("model"),
                           device=meta.get("device"))
    index = load_index(_ROOT / settings.index_dir, backend=settings.vector_backend)
    head_path = _ROOT / settings.index_dir / "head.pt"
    head = load_head(head_path) if head_path.is_file() else None

    groups: dict[str, list[str]] = defaultdict(list)
    for slug, record in catalog.items():
        groups[near_dup_key(record)].append(slug)
    near_members = {slug for slugs in groups.values() if len(slugs) > 1 for slug in slugs}

    targets = sorted(slug for slug in near_members if (uploads / (catalog[slug].get("image") or "")).is_file())
    rng = np.random.default_rng(args.seed)
    if len(targets) > args.samples:
        targets = sorted(rng.choice(targets, size=args.samples, replace=False).tolist())

    rows: list[dict] = []
    latencies: list[float] = []
    for slug in targets:
        path = uploads / (catalog[slug].get("image") or "")
        try:
            with Image.open(path) as raw:
                rgba = np.asarray(raw.convert("RGBA"))
        except Exception:  # noqa: BLE001
            continue
        query = field_augment(rgba, rng)
        fields, ocr_ms, text = _query_fields(query)
        latencies.append(ocr_ms)

        hits = search_images(encoder, index, store, [query], k=args.topk, head=head)
        candidates = [{"slug": hit.slug, "score": float(hit.score)} for hit in hits]
        triggered = should_fuse(candidates, store)
        group = set(groups[near_dup_key(catalog[slug])])

        ranks: dict[str, int] = {}
        timings: dict[str, float] = {}
        for mode in modes:
            start = time.perf_counter()
            ranked = rerank_candidates(candidates, fields, store, enriched=enriched, fusion=mode)
            timings[mode] = round((time.perf_counter() - start) * 1000.0, 1)
            order = [candidate.slug for candidate in ranked]
            ranks[mode] = order.index(slug) + 1 if slug in order else -1

        rows.append({
            "slug": slug,
            "in_near_dup_group": len(group) > 1,
            "triggered": triggered,
            "visual_top1": candidates[0]["slug"] if candidates else "",
            "visual_top1_correct": bool(candidates) and candidates[0]["slug"] == slug,
            "ranks": ranks,
            "ms": timings,
            "ocr_text": " | ".join(text.splitlines()[:3]),
        })

    def share(predicate) -> float:
        return round(sum(1 for row in rows if predicate(row)) / len(rows), 4) if rows else 0.0

    summary = {
        mode: {
            "acc_at_1": share(lambda r, m=mode: r["ranks"][m] == 1),
            "acc_at_5": share(lambda r, m=mode: 0 < r["ranks"][m] <= 5),
            "rerank_ms_p50": _percentile([r["ms"][mode] for r in rows], 50),
        }
        for mode in modes
    }
    fired = [row for row in rows if row["triggered"]]
    improved = [row for row in fired if row["ranks"].get("always", -1) != -1
                and row["ranks"].get("off", -1) != -1 and row["ranks"]["always"] < row["ranks"]["off"]]
    hurt = [row for row in fired if row["ranks"].get("always", -1) != -1
            and row["ranks"].get("off", -1) != -1 and row["ranks"]["always"] > row["ranks"]["off"]]

    report = {
        "phase": "validation",
        "domain_warning": "домен — эталоны каталога + синтетическая аугментация, не полевые фото (C-011)",
        "samples": len(rows),
        "catalog_size": len(catalog),
        "seed": args.seed,
        "topk": args.topk,
        "modes": modes,
        "mode_summary": summary,
        "baseline": summary.get("off", {}),
        "text_fusion": summary.get("always", {}),
        "conditional": summary.get("conditional", {}),
        "visual_baseline": {
            "acc_at_1": share(lambda r: r["visual_top1_correct"]),
            "near_dup_acc_at_1": share(lambda r: r["visual_top1_correct"] and r["in_near_dup_group"]),
        },
        "trigger": {
            "fired": len(fired),
            "fire_rate": round(len(fired) / len(rows), 4) if rows else 0.0,
            "improved_rank": len(improved),
            "hurt_rank": len(hurt),
        },
        "latency_ms": {"ocr_p50": _percentile(latencies, 50), "ocr_p95": _percentile(latencies, 95)},
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log_marker("M-EVAL", "text_fusion", "DONE", samples=len(rows),
               off_acc1=summary.get("off", {}).get("acc_at_1"),
               always_acc1=summary.get("always", {}).get("acc_at_1"),
               conditional_acc1=summary.get("conditional", {}).get("acc_at_1"),
               trigger_fired=len(fired))

    print("=== ТЕКСТОВЫЙ ФЬЮЖН ПОВЕРХ ВИЗУАЛЬНОГО TOP-K (Ф3) ===")
    print(f"выборка: {len(rows)} near-dup эталонов · каталог {len(catalog)} · top-K {args.topk} · seed {args.seed}")
    print(f"визуальный baseline: acc@1 {report['visual_baseline']['acc_at_1']}")
    print(f"{'метрика':<20} " + "  ".join(f"{mode:>12}" for mode in modes))
    for label, key in (("acc@1", "acc_at_1"), ("acc@5", "acc_at_5"), ("rerank p50, мс", "rerank_ms_p50")):
        print(f"{label:<20} " + "  ".join(f"{summary[mode][key]:>12}" for mode in modes))
    print(f"триггер сработал: {len(fired)}/{len(rows)} ({report['trigger']['fire_rate']}); "
          f"из них улучшил ранг {len(improved)}, ухудшил {len(hurt)}")
    print(f"OCR p50 {report['latency_ms']['ocr_p50']} мс")
    print(f"домен: {report['domain_warning']}")
    print(f"enriched: {enriched_path.name}")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
