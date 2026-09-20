#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Замер text-first поиска и RRF на реальных фото: baseline, text-only, visual-only, RRF
# SCOPE: load_cases, visual_ranking, text_ranking, rank_of_identity, evaluate_configs, main
# DEPENDS: json, numpy, packages.common.catalog, packages.common.config, packages.common.images,
#          packages.common.logging, packages.common.text_rank, scripts.eval.real_field_eval,
#          services.retrieval.app.encoder, services.retrieval.app.geo, services.retrieval.app.head,
#          services.retrieval.app.index, services.retrieval.app.search
# LINKS: M-EVAL, M-TEXT, M-RETRIEVAL
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# DEFAULT_DEPTH - глубина выдачи для оценки достижимости
# LABELS_PATH - файл псевдо-разметки реальных фото
# load_cases - фото с полями запроса и слабой GT
# visual_ranking - визуальное ранжирование тем же путём, что в проде (гео + дедуп)
# text_ranking - текстовое ранжирование по всему каталогу
# catalog_of - записи каталога из read-only хранилища
# rank_of_identity - ранг первой правильной идентичности
# evaluate_configs - метрики для набора конфигураций слияния
# main - прогон и отчёт
# END_MODULE_MAP
"""Замер text-first и RRF на реальных фото.

Сравниваются четыре конфигурации на одних и тех же 21 фото: визуальный baseline
(как в проде), только текст по всему каталогу, и их слияние через RRF. Так видно,
даёт ли текст новых кандидатов, которых визуальный поиск не находит вообще.
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
from packages.common.text_rank import fields_from_ocr, rrf, text_index, text_search  # noqa: E402
from scripts.eval.real_field_eval import identity_candidates, parse_label  # noqa: E402
from services.retrieval.app.encoder import load_encoder  # noqa: E402
from services.retrieval.app.geo import GeoIndex  # noqa: E402
from services.retrieval.app.head import load_head  # noqa: E402
from services.retrieval.app.index import load_index  # noqa: E402
from services.retrieval.app.search import dedupe_by_slug, search_images  # noqa: E402

DEFAULT_DEPTH = 100
LABELS_PATH = "reports/real_field_labels.jsonl"

__all__ = ["DEFAULT_DEPTH", "evaluate_configs", "load_cases", "main", "rank_of_identity",
           "text_ranking", "visual_ranking"]


def load_cases(directory: Path, catalog: dict) -> list[dict]:
    """Собирает фото, слабую GT и текстовые поля запроса (из разметки 26b/OCR)."""
    manifest = directory / "manifest.tsv"
    labels: dict[str, str] = {}
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                name, _, text = line.partition("\t")
                labels[name] = text
    fields_by_file: dict[str, dict] = {}
    labels_file = _ROOT / LABELS_PATH
    if labels_file.is_file():
        for line in labels_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            vlm = record.get("vlm") or {}
            fields = {
                "winery": vlm.get("winery") or "",
                "wine_name": vlm.get("name") or "",
                "grape": vlm.get("grape") or "",
                "region": vlm.get("region") or "",
                "vintage": str(vlm.get("year") or ""),
                "sweetness": vlm.get("sweetness") or "",
            }
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


def visual_ranking(case: dict, encoder, head, index, store: CatalogStore, geo_index,
                   geo_weight: float, depth: int) -> tuple[list[str], float]:
    """Визуальное ранжирование тем же путём, что в проде: search_images + дедуп по slug."""
    started = time.perf_counter()
    hits = search_images(encoder, index, store, [case["image"]], k=depth, head=head,
                         geo_index=geo_index, geo_weight=geo_weight,
                         geo_topk=settings.geo_topk, geo_min_matches=settings.geo_min_matches)
    latency = (time.perf_counter() - started) * 1000.0
    ranked = [slug for slug, _ in dedupe_by_slug([(hit.slug, hit.score) for hit in hits], depth)]
    return ranked, latency


def text_ranking(case: dict, catalog: dict, depth: int, index: dict | None = None) -> tuple[list[str], float]:
    """Текстовое ранжирование по всему каталогу."""
    fields = case.get("fields") or {}
    started = time.perf_counter()
    ranked = [slug for slug, _ in text_search(fields, catalog, limit=depth, index=index)]
    latency = (time.perf_counter() - started) * 1000.0
    return ranked, latency


def rank_of_identity(ranked: list[str], candidates: list[str]) -> int:
    """Ранг первой правильной идентичности; 0, если её нет в выдаче."""
    pool = set(candidates)
    for position, slug in enumerate(ranked, 1):
        if slug in pool:
            return position
    return 0


def evaluate_configs(cases: list[dict], encoder, head, index, store: CatalogStore,
                     geo_index, geo_weight: float, depth: int) -> dict:
    """Считает метрики для визуального, текстового и RRF-ранжирований."""
    results: dict[str, dict] = {}
    rows: list[dict] = []
    text_cache = text_index(catalog_of(store))  # формы каталога считаются один раз на прогон
    for case in cases:
        candidates = case["label"]["candidates"]
        if not candidates:
            continue
        visual, visual_ms = visual_ranking(case, encoder, head, index, store, geo_index, geo_weight, depth)
        text, text_ms = text_ranking(case, catalog_of(store), depth, text_cache)
        # Веса RRF: визуал весит больше, иначе более слабый текстовый top-1 вытесняет визуальный
        fused = rrf([visual, text], weights=[1.0, 0.5])
        rows.append({
            "file": case["file"],
            "visual_rank": rank_of_identity(visual, candidates),
            "text_rank": rank_of_identity(text, candidates),
            "rrf_rank": rank_of_identity(fused, candidates),
            "visual_ms": round(visual_ms, 1),
            "text_ms": round(text_ms, 2),
            "text_candidates": len(text),
            "fields": case.get("fields") or {},
        })
    for name, key in (("visual_only", "visual_rank"), ("text_only", "text_rank"), ("rrf", "rrf_rank")):
        latencies = [row["visual_ms"] if key == "visual_rank" else (row["text_ms"] if key == "text_rank"
                     else row["visual_ms"] + row["text_ms"]) for row in rows]
        results[name] = {
            "cases": len(rows),
            "identity_at_1": round(sum(1 for row in rows if row[key] == 1) / len(rows), 4) if rows else 0.0,
            "identity_at_5": round(sum(1 for row in rows if 0 < row[key] <= 5) / len(rows), 4) if rows else 0.0,
            "in_top_100": round(sum(1 for row in rows if 0 < row[key] <= 100) / len(rows), 4) if rows else 0.0,
            "latency_p50": round(sorted(latencies)[len(latencies) // 2], 1) if latencies else 0.0,
        }
    results["rows"] = rows
    return results


def catalog_of(store: CatalogStore) -> dict:
    """Достаёт записи каталога из read-only хранилища."""
    return store.slug_to_record() if hasattr(store, "slug_to_record") else {}


def main(argv: list[str] | None = None) -> int:
    """Прогоняет сравнение конфигураций и пишет отчёт."""
    parser = argparse.ArgumentParser(description="Замер text-first и RRF")
    parser.add_argument("--dir", default=str(_ROOT / "data/samples/real_field"))
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--out", default=str(_ROOT / "reports/text_first.json"))
    args = parser.parse_args(argv)

    catalog = load_catalog(_ROOT / settings.catalog_jsonl)
    store = CatalogStore(catalog)
    cases = load_cases(Path(args.dir), catalog)
    with_fields = sum(1 for case in cases if any((case.get("fields") or {}).values()))
    print(f"фото: {len(cases)} · с текстовыми полями: {with_fields} · с GT-идентичностью: "
          f"{sum(1 for case in cases if case['label']['candidates'])}")

    meta = json.loads((_ROOT / settings.index_dir / "meta.json").read_text(encoding="utf-8"))
    encoder = load_encoder(encoder=meta.get("encoder"), model_id=meta.get("model"), device=meta.get("device"))
    index = load_index(_ROOT / settings.index_dir, backend=settings.vector_backend)
    head_path = _ROOT / settings.index_dir / "head.pt"
    head = load_head(head_path) if head_path.is_file() else None
    geo_index = GeoIndex.load(_ROOT / settings.index_dir, meta)

    results = evaluate_configs(cases, encoder, head, index, store, geo_index, settings.geo_weight, args.depth)
    report = {"depth": args.depth, "geo_weight": settings.geo_weight,
              "cases": len(cases), "results": results}
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== TEXT-FIRST И RRF (реальные фото) ===")
    print(f"{'конфигурация':<14} {'identity@1':>11} {'identity@5':>11} {'в top-100':>10} {'p50, мс':>9}")
    for name in ("visual_only", "text_only", "rrf"):
        block = results[name]
        print(f"{name:<14} {block['identity_at_1']:>11} {block['identity_at_5']:>11} "
              f"{block['in_top_100']:>10} {block['latency_p50']:>9}")
    print(f"отчёт: {target}")
    log_marker("M-EVAL", "text_first", "DONE", cases=len(cases),
               visual=results["visual_only"]["identity_at_1"],
               text=results["text_only"]["identity_at_1"], rrf=results["rrf"]["identity_at_1"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
