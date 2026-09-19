#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Валидация текстового фьюжна C-013: текст эталона как pseudo-запрос против всего каталога
# SCOPE: parse_args, main
# DEPENDS: numpy, PIL, rapidfuzz, packages.common.catalog, packages.common.config, packages.common.images,
#          packages.common.logging, packages.common.ocr, services.matcher.app.merge, services.matcher.app.textfusion
# LINKS: M-EVAL, M-MATCHER, M-TEXT, M-OCR
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# DEFAULT_SAMPLES - размер выборки валидации по умолчанию
# parse_args - аргументы CLI валидации
# main - измеряет acc@1/@5 текстового retrieval и дизамбигуацию near-dup до/после обогащения
# END_MODULE_MAP
"""Валидация текстового фьюжна.

Дизайн (honest, без новых фото):
- **Тест A** — самопроверка: эталон против собственного индекса (домен тот же, метрика тривиальна,
  приводится только как sanity).
- **Тест B** — текст эталона как pseudo-запрос против ВСЕХ 2037 slug: acc@1/@5 базовым текстовым
  матчем (winery+name) и с текстовым фьюжном на обогащённом каталоге (grape/year/sweetness).
- **Тест C** — дизамбигуация внутри near-dup групп: попадёт ли запрос в свой член группы.

Оговорка: домен — эталоны каталога, а не полевые фото организатора (их у нас нет, C-011).
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

from packages.common.catalog import load_catalog  # noqa: E402
from packages.common.config import settings  # noqa: E402
from packages.common.images import flatten_alpha  # noqa: E402
from packages.common.logging import log_marker  # noqa: E402
from packages.common.ocr import run_ocr  # noqa: E402
from services.matcher.app.merge import rerank_candidates  # noqa: E402
from services.matcher.app.textfusion import load_enriched_catalog  # noqa: E402

_DEFAULT_UPLOADS = (
    _ROOT.parent / "материалы для конкурса" / "Датасет"
    / "prod-svoe-vino-strapi" / "prod-svoe-vino" / "strapi" / "uploads"
)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
DEFAULT_SAMPLES = 150

__all__ = ["DEFAULT_SAMPLES", "parse_args", "main"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы CLI."""
    parser = argparse.ArgumentParser(description="Валидация текстового фьюжна (C-013)")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(_ROOT / "reports" / "text_fusion.json"))
    parser.add_argument("--uploads", default=str(_DEFAULT_UPLOADS))
    return parser.parse_args(argv)


def _group_key(record: dict) -> str:
    """Ключ near-dup группы (как в KB-билдере)."""
    winery = (record.get("winery") or "").strip().lower()
    name = " ".join(re.sub(r"\d+", " ", (record.get("name") or "").lower()).split())[:48]
    return f"{winery}|{name}"


def _query_fields(image: np.ndarray) -> tuple[dict, float, str]:
    """Строит pseudo-запрос из OCR-текста эталона."""
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


def _rank(catalog: dict, fields: dict, enriched: dict | None, use_fusion: bool) -> list[str]:
    """Ранжирует весь каталог текстовым сигналом (флаг передаётся явно, см. rerank_candidates)."""
    ranked = rerank_candidates(
        [{"slug": slug, "score": 0.0} for slug in catalog], fields, catalog,
        enriched=enriched, fusion=use_fusion,
    )
    return [candidate.slug for candidate in ranked]


def _percentile(values: list[float], p: float) -> float:
    """p95/p50 без внешних зависимостей."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1))))
    return round(ordered[index], 1)


def main(argv: list[str] | None = None) -> int:
    """Прогоняет валидацию и пишет отчёт."""
    args = parse_args(argv)
    catalog = load_catalog(_ROOT / settings.catalog_jsonl)
    enriched = load_enriched_catalog(_ROOT / settings.catalog_enriched_jsonl)
    uploads = Path(args.uploads)

    groups: dict[str, list[str]] = defaultdict(list)
    for slug, record in catalog.items():
        groups[_group_key(record)].append(slug)
    near_members = {slug for slugs in groups.values() if len(slugs) > 1 for slug in slugs}
    ambiguous = {
        slug for slugs in groups.values() if len(slugs) > 1
        and not any(len({str(catalog[s].get(f) or "") for s in slugs}) > 1
                    for f in ("name", "grape", "category", "sweetness", "year", "color"))
        for slug in slugs
    }

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
        image = flatten_alpha(rgba[:, :, [2, 1, 0, 3]])
        fields, ocr_ms, text = _query_fields(image)
        latencies.append(ocr_ms)

        start = time.perf_counter()
        base_rank = _rank(catalog, fields, enriched, use_fusion=False)
        base_ms = (time.perf_counter() - start) * 1000.0
        start = time.perf_counter()
        fused_rank = _rank(catalog, fields, enriched, use_fusion=True)
        fused_ms = (time.perf_counter() - start) * 1000.0

        group = [s for s in groups[_group_key(catalog[slug])]]
        base_group = [s for s in base_rank if s in set(group)][:1]
        fused_group = [s for s in fused_rank if s in set(group)][:1]
        rows.append({
            "slug": slug,
            "query_vintage": fields["vintage"],
            "truth_year": str(catalog[slug].get("year") or ""),
            "truth_sweetness": catalog[slug].get("sweetness") or "",
            "enriched_year": (enriched.get(slug, {}).get("enriched") or {}).get("year", ""),
            "enriched_sweetness": (enriched.get(slug, {}).get("enriched") or {}).get("sweetness", ""),
            "base_top1": base_rank[0], "base_rank": base_rank.index(slug) + 1 if slug in base_rank[:200] else -1,
            "fused_top1": fused_rank[0], "fused_rank": fused_rank.index(slug) + 1 if slug in fused_rank[:200] else -1,
            "base_group_hit": bool(base_group and base_group[0] == slug),
            "fused_group_hit": bool(fused_group and fused_group[0] == slug),
            "base_ms": round(base_ms, 1), "fused_ms": round(fused_ms, 1),
            "ocr_text": " | ".join(text.splitlines()[:4]),
        })

    def share(predicate) -> float:
        return round(sum(1 for row in rows if predicate(row)) / len(rows), 4) if rows else 0.0

    report = {
        "phase": "validation",
        "domain_warning": "домен тот же (эталоны каталога), не полевые фото организатора — C-011 удалил полевой набор",
        "samples": len(rows),
        "catalog_size": len(catalog),
        "seed": args.seed,
        "text_fusion_enabled_default": False,
        "ambiguous_slugs": len(ambiguous),
        "baseline": {
            "acc_at_1": share(lambda r: r["base_rank"] == 1),
            "acc_at_5": share(lambda r: 0 < r["base_rank"] <= 5),
            "group_acc_at_1": share(lambda r: r["base_group_hit"]),
        },
        "text_fusion": {
            "acc_at_1": share(lambda r: r["fused_rank"] == 1),
            "acc_at_5": share(lambda r: 0 < r["fused_rank"] <= 5),
            "group_acc_at_1": share(lambda r: r["fused_group_hit"]),
        },
        "latency_ms": {
            "ocr_p50": _percentile(latencies, 50),
            "ocr_p95": _percentile(latencies, 95),
            "rerank_baseline_p50": _percentile([r["base_ms"] for r in rows], 50),
            "rerank_fused_p50": _percentile([r["fused_ms"] for r in rows], 50),
        },
        "enriched_year_hits": sum(1 for r in rows if r["enriched_year"]),
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log_marker("M-EVAL", "text_fusion", "DONE", samples=len(rows),
               base_acc1=report["baseline"]["acc_at_1"], fused_acc1=report["text_fusion"]["acc_at_1"])

    print("=== ТЕКСТОВЫЙ ФЬЮЖН: ВАЛИДАЦИЯ (Ф3) ===")
    print(f"выборка: {len(rows)} эталонов из {len(catalog)} slug · near-dup членов: {len(near_members)} · ambiguous: {len(ambiguous)}")
    print(f"{'метрика':<22} {'baseline':<12} {'+text fusion':<12} {'Δ'}")
    for label, key in (("acc@1", "acc_at_1"), ("acc@5", "acc_at_5"), ("group acc@1", "group_acc_at_1")):
        left = report["baseline"][key]
        right = report["text_fusion"][key]
        print(f"{label:<22} {left:<12} {right:<12} {round(right - left, 4):+}")
    print(f"latency: OCR p50 {report['latency_ms']['ocr_p50']} мс · rerank p50 "
          f"{report['latency_ms']['rerank_baseline_p50']} → {report['latency_ms']['rerank_fused_p50']} мс")
    print(f"домен: {report['domain_warning']}")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
