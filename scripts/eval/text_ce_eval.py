#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Замер cross-encoder реранка в режимах off/text/visual/both на реальных фото
# SCOPE: load_cases, visual_ranking, text_ranking, apply_ce, evaluate, main
# DEPENDS: json, numpy, packages.common.catalog, packages.common.config, packages.common.images,
#          packages.common.logging, packages.common.text_ce, packages.common.text_rank,
#          scripts.eval.real_field_eval, services.retrieval.app.encoder, services.retrieval.app.geo,
#          services.retrieval.app.head, services.retrieval.app.index, services.retrieval.app.search
# LINKS: M-EVAL, M-TEXT
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# DEFAULT_DEPTH - глубина визуальной выдачи
# DEFAULT_TEXT_DEPTH - сколько текстовых кандидатов подаётся в реранк
# load_cases - фото с текстовыми полями и слабой GT
# visual_ranking - визуальное ранжирование как в проде
# text_ranking - текстовое ранжирование по всему каталогу
# apply_ce - реранк списка cross-encoder-ом с сохранением хвоста
# evaluate - метрики по режимам off/text/visual/both
# main - прогон и отчёт
# END_MODULE_MAP
"""Замер cross-encoder реранка.

Сравниваются четыре режима на одних фото: off (текущий путь), реранк текстового списка,
реранк визуального top-K и реранк обоих с RRF. Печатаются identity@1/@5, изменённые ранги,
достижимость top-100 и latency — по ним и принимается решение о дефолте.
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
from packages.common.text_ce import rerank as ce_rerank  # noqa: E402
from packages.common.text_rank import fields_from_ocr, rrf, text_index, text_search  # noqa: E402
from scripts.eval.real_field_eval import identity_candidates, parse_label  # noqa: E402
from services.retrieval.app.encoder import load_encoder  # noqa: E402
from services.retrieval.app.geo import GeoIndex  # noqa: E402
from services.retrieval.app.head import load_head  # noqa: E402
from services.retrieval.app.index import load_index  # noqa: E402
from services.retrieval.app.search import dedupe_by_slug, search_images  # noqa: E402

DEFAULT_DEPTH = 10
DEFAULT_TEXT_DEPTH = 20

__all__ = ["DEFAULT_DEPTH", "DEFAULT_TEXT_DEPTH", "apply_ce", "evaluate", "load_cases",
           "main", "text_ranking", "visual_ranking"]


def load_cases(directory: Path, catalog: dict, labels_path: Path) -> list[dict]:
    """Фото с текстовыми полями из разметки и слабой GT по идентичности."""
    manifest = directory / "manifest.tsv"
    labels: dict[str, str] = {}
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                name, _, text = line.partition("\t")
                labels[name] = text
    fields_by_file: dict[str, dict] = {}
    if labels_path.is_file():
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            vlm = record.get("vlm") or {}
            fields = {"winery": vlm.get("winery") or "", "wine_name": vlm.get("name") or "",
                      "grape": vlm.get("grape") or "", "region": vlm.get("region") or "",
                      "vintage": str(vlm.get("year") or ""), "sweetness": vlm.get("sweetness") or ""}
            if not any(fields.values()):
                fields = fields_from_ocr(record.get("ocr_text") or "",
                                         vintage=record.get("ocr_vintage") or "",
                                         sweetness=record.get("ocr_sweetness") or "")
            fields_by_file[record.get("file", "")] = fields
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
        cases.append({"file": name, "image": image, "label": label,
                      "fields": fields_by_file.get(name) or {}})
    return cases


def visual_ranking(case: dict, encoder, head, index, store, geo_index, depth: int) -> list[tuple[str, float]]:
    """Визуальные кандидаты со скорами (гео + дедуп, как в проде)."""
    hits = search_images(encoder, index, store, [case["image"]], k=max(depth * 10, 200), head=head,
                         geo_index=geo_index, geo_weight=settings.geo_weight,
                         geo_topk=settings.geo_topk, geo_min_matches=settings.geo_min_matches)
    return dedupe_by_slug([(hit.slug, hit.score) for hit in hits], depth)


def text_ranking(case: dict, catalog: dict, cache: dict, depth: int) -> list[tuple[str, float]]:
    """Текстовые кандидаты по всему каталогу."""
    return text_search(case.get("fields") or {}, catalog, limit=depth, index=cache)


def apply_ce(case: dict, candidates: list[tuple[str, float]], store, budget_ms: float) -> tuple[list[tuple[str, float]], float]:
    """Реранк списка cross-encoder-ом; при недоступности возвращает исходный список."""
    started = time.perf_counter()
    reranked = ce_rerank(case.get("fields") or {}, candidates, store,
                         top_n=min(len(candidates), 20), budget_ms=budget_ms)
    latency = (time.perf_counter() - started) * 1000.0
    return (reranked if reranked is not None else candidates), latency


def _rank(ranked: list[str], candidates: list[str]) -> int:
    """Ранг первой правильной идентичности; 0, если её нет."""
    pool = set(candidates)
    for position, slug in enumerate(ranked, 1):
        if slug in pool:
            return position
    return 0


def evaluate(cases: list[dict], encoder, head, index, store, geo_index, catalog: dict,
             depth: int, text_depth: int) -> dict:
    """Считает метрики для четырёх режимов."""
    cache = text_index(catalog)
    rows: list[dict] = []
    for case in cases:
        candidates_gt = case["label"]["candidates"]
        if not candidates_gt:
            continue
        visual = visual_ranking(case, encoder, head, index, store, geo_index, depth)
        text = text_ranking(case, catalog, cache, text_depth)
        ce_text, ce_text_ms = apply_ce(case, text, store, settings.text_first_max_latency_ms)
        ce_visual, ce_visual_ms = apply_ce(case, visual, store, settings.text_first_max_latency_ms)
        orders = {
            "off": [slug for slug, _ in visual],
            "ce_text": rrf([[slug for slug, _ in visual], [slug for slug, _ in ce_text]], weights=[1.0, 0.5]),
            "ce_visual": [slug for slug, _ in ce_visual],
            "ce_both": rrf([[slug for slug, _ in ce_visual], [slug for slug, _ in ce_text]], weights=[1.0, 0.5]),
        }
        row = {"file": case["file"], "ce_text_ms": round(ce_text_ms, 1),
               "ce_visual_ms": round(ce_visual_ms, 1)}
        for mode, order in orders.items():
            row[f"{mode}_rank"] = _rank(order, candidates_gt)
        rows.append(row)

    summary: dict[str, dict] = {}
    for mode in ("off", "ce_text", "ce_visual", "ce_both"):
        key = f"{mode}_rank"
        latencies = [row["ce_text_ms"] + row["ce_visual_ms"] for row in rows]
        summary[mode] = {
            "identity_at_1": round(sum(1 for row in rows if row[key] == 1) / len(rows), 4) if rows else 0.0,
            "identity_at_5": round(sum(1 for row in rows if 0 < row[key] <= 5) / len(rows), 4) if rows else 0.0,
            "in_top_100": round(sum(1 for row in rows if 0 < row[key] <= 100) / len(rows), 4) if rows else 0.0,
            "latency_p50_ms": round(sorted(latencies)[len(latencies) // 2], 1) if latencies else 0.0,
        }
    changed = [row for row in rows if row["off_rank"] != row["ce_both_rank"]]
    summary["ranks"] = {
        "improved": sum(1 for row in changed if row["ce_both_rank"] < row["off_rank"]
                        or (row["ce_both_rank"] > 0 and row["off_rank"] == 0)),
        "worsened": sum(1 for row in changed if row["ce_both_rank"] > row["off_rank"]
                        or (row["ce_both_rank"] == 0 and row["off_rank"] > 0)),
    }
    summary["rows"] = rows
    return summary


def main(argv: list[str] | None = None) -> int:
    """Прогоняет замер по режимам и пишет отчёт."""
    parser = argparse.ArgumentParser(description="Замер cross-encoder реранка")
    parser.add_argument("--dir", default=str(_ROOT / "data/samples/real_field"))
    parser.add_argument("--labels", default=str(_ROOT / "reports/real_field_labels.jsonl"))
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--text-depth", type=int, default=DEFAULT_TEXT_DEPTH)
    parser.add_argument("--out", default=str(_ROOT / "reports/text_ce.json"))
    args = parser.parse_args(argv)

    catalog = load_catalog(_ROOT / settings.catalog_jsonl)
    store = CatalogStore(catalog)
    cases = load_cases(Path(args.dir), catalog, Path(args.labels))
    print(f"фото: {len(cases)} · с GT-идентичностью: {sum(1 for c in cases if c['label']['candidates'])}")
    from packages.common.text_ce import load_ce  # noqa: PLC0415

    loaded = load_ce()
    print(f"cross-encoder: {'загружен' if loaded else 'НЕДОСТУПЕН (веса не скачаны или сбой)'}")
    meta = json.loads((_ROOT / settings.index_dir / "meta.json").read_text(encoding="utf-8"))
    encoder = load_encoder(encoder=meta.get("encoder"), model_id=meta.get("model"), device=meta.get("device"))
    index = load_index(_ROOT / settings.index_dir, backend=settings.vector_backend)
    head_path = _ROOT / settings.index_dir / "head.pt"
    head = load_head(head_path) if head_path.is_file() else None
    geo_index = GeoIndex.load(_ROOT / settings.index_dir, meta)

    summary = evaluate(cases, encoder, head, index, store, geo_index, catalog, args.depth, args.text_depth)
    report = {"cases": len(cases), "ce_available": bool(loaded), "results": summary}
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== CROSS-ENCODER РЕРАНК (реальные фото) ===")
    print(f"{'режим':<12} {'identity@1':>11} {'identity@5':>11} {'в top-100':>10} {'p50, мс':>9}")
    for mode in ("off", "ce_text", "ce_visual", "ce_both"):
        block = summary[mode]
        print(f"{mode:<12} {block['identity_at_1']:>11} {block['identity_at_5']:>11} "
              f"{block['in_top_100']:>10} {block['latency_p50_ms']:>9}")
    print(f"  ранги (off -> ce_both): улучшено {summary['ranks']['improved']}, "
          f"ухудшено {summary['ranks']['worsened']}")
    print(f"отчёт: {target}")
    log_marker("M-EVAL", "text_ce", "DONE", cases=len(cases), available=bool(loaded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
