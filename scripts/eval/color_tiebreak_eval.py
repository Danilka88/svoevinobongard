#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Сборка палитр эталонов и замер цветового tie-breaker на реальных фото
# SCOPE: load_cases, visual_candidates, evaluate, main
# DEPENDS: json, numpy, packages.common.catalog, packages.common.config, packages.common.images,
#          packages.common.logging, packages.common.palette, scripts.eval.real_field_eval,
#          services.retrieval.app.encoder, services.retrieval.app.geo, services.retrieval.app.head,
#          services.retrieval.app.index, services.retrieval.app.search
# LINKS: M-EVAL, M-RETRIEVAL
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# DEFAULT_PALETTES - путь артефакта палитр
# DEFAULT_DEPTH - сколько кандидатов берётся для tie-зоны
# load_cases - фото со слабой GT
# visual_candidates - визуальные кандидаты со скорами (как в проде)
# evaluate - сравнение baseline и tie-breaker: identity@1/@5, ±ранги, latency
# main - сборка палитр или замер
# END_MODULE_MAP
"""Замер цветового tie-breaker.

Сравниваются два режима на одних и тех же фото: baseline (порядок визуального поиска)
и tie-breaker (перестановка в зоне близких скоров по палитре). Печатается, сколько раз
зона вообще срабатывала и какие ранги это изменило — без этого вывод был бы недоказуем.
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
from packages.common.images import flatten_alpha  # noqa: E402
from packages.common.logging import log_marker  # noqa: E402
from packages.common.palette import (DEFAULT_K, build_palettes, dominant_palette,  # noqa: E402
                                     load_palettes, tiebreak_order)
from scripts.eval.real_field_eval import identity_candidates, parse_label  # noqa: E402
from services.retrieval.app.encoder import load_encoder  # noqa: E402
from services.retrieval.app.geo import GeoIndex  # noqa: E402
from services.retrieval.app.head import load_head  # noqa: E402
from services.retrieval.app.index import load_index  # noqa: E402
from services.retrieval.app.search import dedupe_by_slug, search_images  # noqa: E402

DEFAULT_PALETTES = "data/index/palettes.json"
DEFAULT_DEPTH = 10

__all__ = ["DEFAULT_DEPTH", "DEFAULT_PALETTES", "evaluate", "load_cases", "main", "visual_candidates"]


def load_cases(directory: Path, catalog: dict) -> list[dict]:
    """Загружает реальные фото со слабой GT по идентичности."""
    manifest = directory / "manifest.tsv"
    labels: dict[str, str] = {}
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                name, _, text = line.partition("\t")
                labels[name] = text
    cases: list[dict] = []
    for name, text in sorted(labels.items()):
        path = directory / name
        if not path.is_file():
            continue
        try:
            with Image.open(path) as raw:
                image = flatten_alpha(np.asarray(raw.convert("RGBA"))[:, :, [2, 1, 0, 3]])
        except Exception:  # noqa: BLE001
            continue
        label = parse_label(text)
        label["candidates"] = identity_candidates(label, catalog, 72.0)
        cases.append({"file": name, "image": image, "label": label})
    return cases


def visual_candidates(case: dict, encoder, head, index, store: CatalogStore, geo_index,
                      depth: int) -> list[tuple[str, float]]:
    """Визуальные кандидаты со скорами тем же путём, что в проде (гео + дедуп)."""
    hits = search_images(encoder, index, store, [case["image"]], k=max(depth * 10, 200), head=head,
                         geo_index=geo_index, geo_weight=settings.geo_weight,
                         geo_topk=settings.geo_topk, geo_min_matches=settings.geo_min_matches)
    return dedupe_by_slug([(hit.slug, hit.score) for hit in hits], depth)


def _rank(ranked: list[str], candidates: list[str]) -> int:
    """Ранг первой правильной идентичности; 0, если её нет."""
    pool = set(candidates)
    for position, slug in enumerate(ranked, 1):
        if slug in pool:
            return position
    return 0


def evaluate(cases: list[dict], encoder, head, index, store: CatalogStore, geo_index,
             palettes: dict, depth: int) -> dict:
    """Сравнивает baseline и tie-breaker по identity@1/@5, изменённым рангам и latency."""
    rows: list[dict] = []
    for case in cases:
        candidates_gt = case["label"]["candidates"]
        if not candidates_gt:
            continue
        started = time.perf_counter()
        visual = visual_candidates(case, encoder, head, index, store, geo_index, depth)
        base_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        query_palette = dominant_palette(case["image"])
        palette_ms = (time.perf_counter() - started) * 1000.0
        baseline = [slug for slug, _ in visual]
        fused = tiebreak_order(visual, query_palette, palettes,
                               margin=settings.color_tiebreak_margin,
                               weight=settings.color_tiebreak_weight)
        in_zone = bool(len(visual) > 1
                       and sorted(visual, key=lambda item: item[1], reverse=True)[0][1]
                       - sorted(visual, key=lambda item: item[1], reverse=True)[1][1]
                       <= settings.color_tiebreak_margin)
        base_rank, fused_rank = _rank(baseline, candidates_gt), _rank(fused, candidates_gt)
        rows.append({
            "file": case["file"],
            "baseline_rank": base_rank,
            "fused_rank": fused_rank,
            "in_zone": in_zone,
            "changed": baseline[:5] != fused[:5],
            "baseline_ms": round(base_ms, 1),
            "palette_ms": round(palette_ms, 1),
        })

    def share(key: str, predicate) -> float:
        return round(sum(1 for row in rows if predicate(row[key])) / len(rows), 4) if rows else 0.0

    changed = [row for row in rows if row["baseline_rank"] != row["fused_rank"]]
    return {
        "cases": len(rows),
        "zone_hits": sum(1 for row in rows if row["in_zone"]),
        "baseline": {
            "identity_at_1": share("baseline_rank", lambda value: value == 1),
            "identity_at_5": share("baseline_rank", lambda value: 0 < value <= 5),
        },
        "tiebreak": {
            "identity_at_1": share("fused_rank", lambda value: value == 1),
            "identity_at_5": share("fused_rank", lambda value: 0 < value <= 5),
        },
        "improved": sum(1 for row in changed if row["fused_rank"] < row["baseline_rank"]
                        or (row["fused_rank"] > 0 and row["baseline_rank"] == 0)),
        "worsened": sum(1 for row in changed if row["fused_rank"] > row["baseline_rank"]
                        or (row["fused_rank"] == 0 and row["baseline_rank"] > 0)),
        "latency_ms": {
            "baseline_p50": round(sorted(row["baseline_ms"] for row in rows)[len(rows) // 2], 1) if rows else 0.0,
            "palette_p50": round(sorted(row["palette_ms"] for row in rows)[len(rows) // 2], 1) if rows else 0.0,
        },
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    """Собирает палитры или замеряет tie-breaker."""
    parser = argparse.ArgumentParser(description="Сборка палитр и замер цветового tie-breaker")
    parser.add_argument("--build", action="store_true", help="собрать палитры эталонов")
    parser.add_argument("--dir", default=str(_ROOT / "data/samples/real_field"))
    parser.add_argument("--palettes", default=str(_ROOT / DEFAULT_PALETTES))
    parser.add_argument("--uploads", default=str(
        _ROOT.parent / "материалы для конкурса" / "Датасет" / "prod-svoe-vino-strapi"
        / "prod-svoe-vino" / "strapi" / "uploads"))
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default=str(_ROOT / "reports/color_tiebreak.json"))
    args = parser.parse_args(argv)

    if args.build:
        slug_to_image = json.loads((_ROOT / "data/catalog/slug_to_image.json").read_text(encoding="utf-8"))
        build_palettes(slug_to_image, args.uploads, args.palettes, k=DEFAULT_K,
                       resume=True, limit=args.limit)
        return 0

    catalog = load_catalog(_ROOT / settings.catalog_jsonl)
    store = CatalogStore(catalog)
    cases = load_cases(Path(args.dir), catalog)
    palettes = load_palettes(args.palettes)
    print(f"фото: {len(cases)} · палитр эталонов: {len(palettes)} · "
          f"флагов tie-break: enabled={settings.color_tiebreak_enabled}")
    meta = json.loads((_ROOT / settings.index_dir / "meta.json").read_text(encoding="utf-8"))
    encoder = load_encoder(encoder=meta.get("encoder"), model_id=meta.get("model"), device=meta.get("device"))
    index = load_index(_ROOT / settings.index_dir, backend=settings.vector_backend)
    head_path = _ROOT / settings.index_dir / "head.pt"
    head = load_head(head_path) if head_path.is_file() else None
    geo_index = GeoIndex.load(_ROOT / settings.index_dir, meta)

    result = evaluate(cases, encoder, head, index, store, geo_index, palettes, args.depth)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== ЦВЕТОВОЙ TIE-BREAKER (реальные фото) ===")
    print(f"  случаев в tie-зоне: {result['zone_hits']} из {result['cases']}")
    print(f"{'режим':<12} {'identity@1':>11} {'identity@5':>11}")
    for name in ("baseline", "tiebreak"):
        print(f"{name:<12} {result[name]['identity_at_1']:>11} {result[name]['identity_at_5']:>11}")
    print(f"  изменено рангов: улучшено {result['improved']}, ухудшено {result['worsened']}")
    print(f"  latency: поиск p50 {result['latency_ms']['baseline_p50']} мс · "
          f"палитра запроса p50 {result['latency_ms']['palette_p50']} мс")
    print(f"отчёт: {target}")
    log_marker("M-EVAL", "color_tiebreak", "DONE", zone=result["zone_hits"],
               improved=result["improved"], worsened=result["worsened"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
