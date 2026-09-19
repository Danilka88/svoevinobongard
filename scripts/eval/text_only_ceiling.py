#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Потолок текстового сигнала на реальных фото: мапится ли текст этикетки в правильную идентичность
# SCOPE: load_labels, match_strict, main
# DEPENDS: json, pathlib, packages.common.catalog, packages.common.logging, packages.common.text_norm
# LINKS: M-EVAL, M-TEXT
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# DEFAULT_LABELS - файл псевдо-разметки реальных фото
# DEFAULT_THRESHOLD - порог совпадения для строгого матчинга
# load_labels - чтение псевдо-разметки (поля 26b и текст OCR) по фото
# match_strict - строгая проверка: совпали И винодельня, И название или сорт
# main - доля фото, где текст этикетки достаточен для правильной идентичности
# END_MODULE_MAP
"""Потолок текстового сигнала (подготовка к C-028).

Проверяется не поиск, а сам текст: если взять прочитанные поля (26b) и текст OCR,
нормализовать их транслитом и сравнить с каталогом, находится ли **правильная**
идентичность. Строгое правило: требуется совпадение и винодельни, и названия или сорта —
совпадения только по бренду недостаточно, иначе метрика завышается на крупных брендах.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from packages.common.catalog import load_catalog  # noqa: E402
from packages.common.logging import log_marker  # noqa: E402
from packages.common.text_norm import similarity

DEFAULT_LABELS = "reports/real_field_labels.jsonl"
DEFAULT_THRESHOLD = 0.82

__all__ = ["DEFAULT_LABELS", "DEFAULT_THRESHOLD", "load_labels", "main", "match_strict"]


def load_labels(path: Path) -> list[dict]:
    """Читает псевдо-разметку: файл, поля 26b и текст OCR."""
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def match_strict(entry: dict, catalog: dict, threshold: float) -> dict:
    """Строгий матч: винодельня И (название ИЛИ сорт) совпали с записью каталога."""
    vlm = entry.get("vlm") or {}
    query_winery = str(vlm.get("winery") or "")
    query_name = str(vlm.get("name") or "")
    query_grape = str(vlm.get("grape") or "")
    query_text = str(entry.get("ocr_text") or "")
    best: dict = {"matched": False, "slug": "", "winery_score": 0.0, "detail_score": 0.0,
                  "query_winery": query_winery, "query_name": query_name}
    for slug, record in catalog.items():
        catalog_winery = str(record.get("winery") or "")
        catalog_detail = f"{record.get('name', '')} {record.get('grape', '')}"
        for winery_source in (query_winery, query_text):
            winery_score = similarity(winery_source, catalog_winery) if winery_source else 0.0
            if winery_score < threshold:
                continue
            detail = max(
                similarity(query_name, catalog_detail) if query_name else 0.0,
                similarity(query_grape, catalog_detail) if query_grape else 0.0,
                similarity(query_text, catalog_detail),
            )
            if detail < threshold:
                continue
            if winery_score + detail > best["winery_score"] + best["detail_score"]:
                best = {"matched": True, "slug": slug, "winery_score": round(winery_score, 3),
                        "detail_score": round(detail, 3), "query_winery": query_winery,
                        "query_name": query_name}
    return best


def main(argv: list[str] | None = None) -> int:
    """Считает text-only потолок по реальным фото."""
    parser = argparse.ArgumentParser(description="Потолок текстового сигнала на реальных фото")
    parser.add_argument("--labels", default=str(_ROOT / DEFAULT_LABELS))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--out", default=str(_ROOT / "reports" / "text_only_ceiling.json"))
    args = parser.parse_args(argv)

    catalog = load_catalog(_ROOT / "data/catalog/catalog.jsonl")
    entries = load_labels(Path(args.labels))
    results = [match_strict(entry, catalog, args.threshold) for entry in entries]
    matched = [result for result in results if result["matched"]]
    report = {
        "cases": len(results),
        "matched": len(matched),
        "share": round(len(matched) / len(results), 4) if results else 0.0,
        "threshold": args.threshold,
        "rule": "строго: совпали винодельня И (название ИЛИ сорт); совпадения только по бренду недостаточно",
        "examples": matched[:8],
        "unmatched": [
            {"file": entry.get("file"), "winery": (entry.get("vlm") or {}).get("winery", ""),
             "name": (entry.get("vlm") or {}).get("name", ""),
             "ocr": (entry.get("ocr_text") or "")[:70]}
            for entry, result in zip(entries, results) if not result["matched"]
        ],
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== TEXT-ONLY ПОТОЛОК (реальные фото) ===")
    print(f"  фото: {report['cases']} · текст достаточен для идентичности: {report['matched']} "
          f"({report['share']}) при пороге {args.threshold}")
    print("  примеры совпадений:")
    for example in matched[:5]:
        print(f"    {example['query_winery'][:20]:<22} {example['query_name'][:18]:<20} -> {example['slug'][:44]}")
    log_marker("M-EVAL", "text_only_ceiling", "DONE", matched=len(matched), cases=len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
