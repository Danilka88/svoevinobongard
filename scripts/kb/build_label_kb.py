#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Label-text KB: точечное чтение текста эталонов (OCR везде, 12b для пробелов) и обогащение каталога
# SCOPE: parse_args, main
# DEPENDS: numpy, PIL, rapidfuzz, packages.common.catalog, packages.common.config, packages.common.images,
#          packages.common.logging, packages.common.ocr, packages.common.preprocess, packages.normalize биндинг через tta,
#          services.normalize.app.tta, services.vlm.app.extract
# LINKS: M-TEXT, M-OCR, M-CATALOG, M-VLM
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# ENRICH_FIELDS - поля, дозаполняемые в обогащённом каталоге
# parse_args - разбор аргументов CLI (режим pilot или enrich)
# main - пилот читаемости полей или точечное обогащение каталога в отдельный файл
# END_MODULE_MAP
"""KB текста этикеток.

Работаем точечно: только 228 slug из near-dup групп (10.8% каталога). OCR идёт по
всем целевым эталонам (≈0.3 с на кадр), gemma4:12b подключается лишь там, где OCR
не дал поле — его p50 ~34 с делает полный прогон офлайн-задачей на часы.

Результаты:
- ``data/kb/label_text.jsonl`` — что прочитано с каждого эталона и кем;
- ``data/catalog/catalog_enriched.jsonl`` — копия каталога с дозаполненными полями
  (исходный catalog.jsonl НЕ перезаписывается).

Свободный текст ``color`` не используется как признак: он нормализуется в ``category``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from rapidfuzz import fuzz

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from packages.common.catalog import load_catalog  # noqa: E402
from packages.common.config import settings  # noqa: E402
from packages.common.images import flatten_alpha  # noqa: E402
from packages.common.logging import log_marker  # noqa: E402
from packages.common.ocr import run_ocr  # noqa: E402
from packages.common.preprocess import PreprocessConfig  # noqa: E402
from services.normalize.app.tta import build_vlm_inputs  # noqa: E402
from services.vlm.app.extract import extract_fields  # noqa: E402

_DEFAULT_MATERIALS = (
    _ROOT.parent / "материалы для конкурса" / "Датасет"
    / "prod-svoe-vino-strapi" / "prod-svoe-vino" / "strapi" / "uploads"
)
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_SWEETNESS = ("экстра брют", "брют натюр", "брют", "полусладкое", "полусухое", "сладкое", "сухое")
_COLOR_MAP = {"белое": "Белое", "красное": "Красное", "розовое": "Розовое", "оранжевое": "Оранжевое"}
ENRICH_FIELDS = ("sweetness", "year", "grape")

__all__ = ["parse_args", "main"]


def _group_key(record: dict) -> str:
    """Ключ near-dup группы: винодельня + название без цифр."""
    winery = (record.get("winery") or "").strip().lower()
    name = " ".join(re.sub(r"\d+", " ", (record.get("name") or "").lower()).split())[:48]
    return f"{winery}|{name}"


def _near_dup_slugs(catalog: dict) -> tuple[list[str], dict[str, list[str]], list[str]]:
    """Возвращает целевые slug, карту групп и список полностью неразличимых групп."""
    groups: dict[str, list[str]] = defaultdict(list)
    for slug, record in catalog.items():
        groups[_group_key(record)].append(slug)
    near = {key: slugs for key, slugs in groups.items() if len(slugs) > 1}
    targets = sorted({slug for slugs in near.values() for slug in slugs})
    # ambiguous = все поля совпадают полностью (включая свободный color) — такие группы
    # неразличимы никаким сигналом, кроме изображения.
    all_fields = ("name", "grape", "category", "sweetness", "year", "color")
    core_fields = ("name", "grape", "category", "sweetness", "year")
    ambiguous = [
        key for key, slugs in near.items()
        if not any(len({str(catalog[s].get(f) or "") for s in slugs}) > 1 for f in all_fields)
    ]
    color_only = [
        key for key, slugs in near.items()
        if any(len({str(catalog[s].get(f) or "") for s in slugs}) > 1 for f in core_fields) is False
        and key not in set(ambiguous)
    ]
    return targets, near, ambiguous, color_only


def _read_reference(path: Path) -> np.ndarray | None:
    """Читает эталон и схлопывает прозрачность на белый фон."""
    try:
        with Image.open(path) as raw:
            rgba = np.asarray(raw.convert("RGBA"))
    except Exception:  # noqa: BLE001
        return None
    return flatten_alpha(rgba[:, :, [2, 1, 0, 3]])


def _match_score(ocr_value: str, catalog_value: str) -> float:
    """Похожесть прочитанного значения на каталожное (rapidfuzz, 0..1)."""
    left = " ".join((ocr_value or "").lower().split())
    right = " ".join((catalog_value or "").lower().split())
    if not left or not right:
        return 0.0
    return round(fuzz.token_set_ratio(left, right) / 100.0, 3)


def _vlm_fields(image: np.ndarray, model: str = "", timeout_s: float = 180.0) -> dict:
    """Пробует дочитать поля моделью (12b) поверх OCR."""
    images, _ = build_vlm_inputs(image, PreprocessConfig(pas_tiles=settings.pas_tiles))
    result = extract_fields(
        images, None,
        model=model or settings.vlm_online_model or None,
        max_tokens=512,
        timeout_s=timeout_s,
        num_ctx=settings.vlm_num_ctx,
        reasoning_effort=settings.vlm_reasoning_effort,
    )
    if not result.ok:
        return {}
    return {"winery": result.winery, "wine_name": result.wine_name, "vintage": result.vintage,
            "grape": result.grape, "sweetness": result.sweetness, "region": result.region}


def _normalize_color(record: dict) -> str:
    """Нормализует свободный color/category к четырём каноническим цветам."""
    haystack = f"{record.get('color','')} {record.get('category','')}".lower()
    for key, canonical in _COLOR_MAP.items():
        if key in haystack:
            return canonical
    return record.get("category", "")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы CLI."""
    parser = argparse.ArgumentParser(description="KB текста этикеток и обогащение каталога")
    parser.add_argument("command", choices=["pilot", "enrich"])
    parser.add_argument("--out", default=str(_ROOT / "data" / "kb"))
    parser.add_argument("--report", default="")
    parser.add_argument("--uploads", default=str(_DEFAULT_MATERIALS))
    parser.add_argument("--with-year", type=int, default=20)
    parser.add_argument("--without-year", type=int, default=20)
    parser.add_argument("--vlm-subset", type=int, default=6)
    parser.add_argument("--vlm-gap-limit", type=int, default=0)
    parser.add_argument("--vlm-timeout", type=float, default=180.0,
                        help="таймаут одной VLM-генерации (p95 12b на 5 изображениях ~50 с)")
    parser.add_argument("--vlm-model", default="gemma4:12b",
                        help="модель для добора полей (p50 ~34 с на кадр)")
    return parser.parse_args(argv)


def _pilot(args: argparse.Namespace, catalog: dict, uploads: Path) -> int:
    """Фаза 0: читаемость полей OCR против 12b."""
    targets, _, _, _ = _near_dup_slugs(catalog)
    with_year = [s for s in targets if _YEAR_RE.search(f"{catalog[s].get('name','')} {s}")]
    without_year = [s for s in targets if s not in set(with_year)]
    sample = with_year[: args.with_year] + without_year[: args.without_year]

    rows: list[dict] = []
    for slug in sample:
        path = uploads / (catalog[slug].get("image") or "")
        if not path.is_file():
            continue
        image = _read_reference(path)
        if image is None:
            continue
        ocr = run_ocr(image, max_side=settings.ocr_max_side)
        lines = ocr.text.splitlines()
        record = catalog[slug]
        row = {
            "slug": slug,
            "ocr_ok": ocr.ok,
            "ocr_lines": len(lines),
            "ocr_vintage": ocr.vintage,
            "ocr_sweetness": ocr.sweetness,
            "ocr_winery_match": _match_score(ocr.winery_hint, record.get("winery")),
            "ocr_name_match": max((_match_score(line, record.get("name")) for line in ocr.lines), default=0.0),
            "ocr_grape_match": max((_match_score(line, record.get("grape")) for line in ocr.lines), default=0.0),
            "truth_year": str(record.get("year") or ""),
            "truth_sweetness": record.get("sweetness") or "",
        }
        rows.append(row)

    official = _ROOT / "data" / "samples" / "eval_queries"
    for photo in sorted(official.glob("*.jpg")) + sorted(official.glob("*.png")):
        image = _read_reference(photo)
        if image is None:
            continue
        ocr = run_ocr(image, max_side=settings.ocr_max_side)
        lines = ocr.text.splitlines()
        rows.append({"slug": f"eval_query:{photo.stem}", "ocr_ok": ocr.ok, "ocr_lines": len(lines),
                     "ocr_vintage": ocr.vintage, "ocr_sweetness": ocr.sweetness,
                     "ocr_winery_match": 0.0,
                     "ocr_name_match": max((_match_score(l, "") for l in lines), default=0.0),
                     "ocr_grape_match": 0.0, "truth_year": "", "truth_sweetness": "",
                     "ocr_text": " | ".join(lines[:6])})

    vlm_rows: list[dict] = []
    for row in rows[: args.vlm_subset]:
        path = uploads / (catalog[row["slug"]].get("image") or "")
        image = _read_reference(path) if path.is_file() else None
        if image is None:
            continue
        fields = _vlm_fields(image, args.vlm_model, args.vlm_timeout)
        record = catalog[row["slug"]]
        vlm_rows.append({
            "slug": row["slug"],
            "filled": [key for key, value in fields.items() if value],
            "vlm_vintage": fields.get("vintage", ""),
            "vlm_winery": fields.get("winery", ""),
            "vlm_sweetness": fields.get("sweetness", ""),
            "truth_year": row["truth_year"],
            "truth_winery": record.get("winery", ""),
            "truth_sweetness": row["truth_sweetness"],
            "winery_match": _match_score(fields.get("winery", ""), record.get("winery")),
        })

    def share(items: list[dict], predicate) -> float:
        return round(sum(1 for item in items if predicate(item)) / len(items), 3) if items else 0.0

    report = {
        "phase": "pilot",
        "targets_total": len(targets),
        "sample": len(rows),
        "vlm_subset": len(vlm_rows),
        "vlm_note": f"12b прогонялся только на {len(vlm_rows)} эталонах: p50 ~34 с, полный прогон не влезает по времени",
        "ocr": {
            "any_line_share": share(rows, lambda r: r["ocr_lines"] > 0),
            "vintage_read_share": share(rows, lambda r: bool(r["ocr_vintage"])),
            "vintage_correct_share": share(rows, lambda r: bool(r["ocr_vintage"]) and r["ocr_vintage"] == r["truth_year"]),
            "sweetness_read_share": share(rows, lambda r: bool(r["ocr_sweetness"])),
            "winery_match_ge_60": share(rows, lambda r: r["ocr_winery_match"] >= 0.6),
            "name_match_ge_60": share(rows, lambda r: r["ocr_name_match"] >= 0.6),
            "grape_match_ge_60": share(rows, lambda r: r["ocr_grape_match"] >= 0.6),
        },
        "vlm": {
            "any_field_share": share(vlm_rows, lambda r: bool(r["filled"])),
            "vintage_read_share": share(vlm_rows, lambda r: bool(r["vlm_vintage"])),
            "vintage_correct_share": share(vlm_rows, lambda r: bool(r["vlm_vintage"]) and r["vlm_vintage"] == r["truth_year"]),
            "winery_match_ge_60": share(vlm_rows, lambda r: r["winery_match"] >= 0.6),
            "sweetness_read_share": share(vlm_rows, lambda r: bool(r["vlm_sweetness"])),
        },
        "rows": rows,
        "vlm_rows": vlm_rows,
    }
    _write_report(args, report)
    log_marker("M-TEXT", "pilot", "DONE", sample=len(rows), vlm=len(vlm_rows),
               ocr_vintage=report["ocr"]["vintage_read_share"],
               vlm_vintage=report["vlm"]["vintage_read_share"])
    print("=== KB PILOT (Ф0) ===")
    print(f"целевых slug: {len(targets)} · выборка OCR: {len(rows)} · подвыборка 12b: {len(vlm_rows)}")
    print(f"{'поле':<26} {'OCR':<10} {'12b':<10}")
    print(f"{'есть строки/поля':<26} {report['ocr']['any_line_share']:<10} {report['vlm']['any_field_share']:<10}")
    print(f"{'year прочитан':<26} {report['ocr']['vintage_read_share']:<10} {report['vlm']['vintage_read_share']:<10}")
    print(f"{'year верен':<26} {report['ocr']['vintage_correct_share']:<10} {report['vlm']['vintage_correct_share']:<10}")
    print(f"{'winery ≥0.6':<26} {report['ocr']['winery_match_ge_60']:<10} {report['vlm']['winery_match_ge_60']:<10}")
    print(f"{'sweetness прочитан':<26} {report['ocr']['sweetness_read_share']:<10} {report['vlm']['sweetness_read_share']:<10}")
    print(f"{'name ≥0.6 / grape ≥0.6':<26} {report['ocr']['name_match_ge_60']:<10} {report['ocr']['grape_match_ge_60']:<10}")
    print(f"note: {report['vlm_note']}")
    return 0


def _enrich(args: argparse.Namespace, catalog: dict, uploads: Path) -> int:
    """Фаза 1: точечное обогащение целевых slug в отдельный файл."""
    targets, near, ambiguous, color_only = _near_dup_slugs(catalog)
    ambiguous_slugs = {slug for key in ambiguous for slug in near[key]}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    enriched_count = 0
    vlm_used = 0
    for slug in targets:
        record = dict(catalog[slug])
        path = uploads / (record.get("image") or "")
        filled: dict = {}
        source = "none"
        if path.is_file():
            image = _read_reference(path)
            if image is not None:
                ocr = run_ocr(image, max_side=settings.ocr_max_side)
                source = "ocr"
                if not record.get("sweetness") and ocr.sweetness:
                    filled["sweetness"] = ocr.sweetness
                if not record.get("year") and ocr.vintage:
                    filled["year"] = int(ocr.vintage)
                if args.vlm_gap_limit and vlm_used < args.vlm_gap_limit and len(filled) < 2:
                    fields = _vlm_fields(image, args.vlm_model, args.vlm_timeout)
                    if fields:
                        vlm_used += 1
                        source = "ocr+vlm"
                        if not filled.get("sweetness") and fields.get("sweetness"):
                            filled["sweetness"] = fields["sweetness"]
                        if not record.get("year") and fields.get("vintage"):
                            year = _YEAR_RE.search(str(fields["vintage"]))
                            if year:
                                filled["year"] = int(year.group(1))
                        if not record.get("grape") and fields.get("grape"):
                            filled["grape"] = fields["grape"]
        record["category_normalized"] = _normalize_color(record)
        record["enriched"] = filled
        record["enrich_source"] = source if filled else "none"
        record["ambiguous"] = slug in ambiguous_slugs
        record["near_dup_group"] = _group_key(record)
        if filled:
            enriched_count += 1
        records.append(record)

    kb_path = out_dir / "label_text.jsonl"
    with kb_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    enriched_path = _ROOT / "data" / "catalog" / "catalog_enriched.jsonl"
    enriched_path.parent.mkdir(parents=True, exist_ok=True)
    with enriched_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = {
        "phase": "enrich",
        "targets": len(targets),
        "groups": len(near),
        "enriched_slugs": enriched_count,
        "vlm_used": vlm_used,
        "ambiguous_slugs": len(ambiguous_slugs),
        "ambiguous_groups": len(ambiguous),
        "color_only_groups": len(color_only),
        "ambiguous_note": "ambiguous = все поля совпали; color_only = различаются только свободным color, который не используется как признак",
        "catalog_jsonl_untouched": True,
        "kb_path": str(kb_path.relative_to(_ROOT)),
        "enriched_path": str(enriched_path.relative_to(_ROOT)),
        "vlm_budget_note": "12b вызывается только для пробелов; полное обогащение 228 slug офлайн ≈ 2.2 ч при p50 34 с",
        "enriched_by_field": {
            "sweetness": sum(1 for r in records if "sweetness" in r["enriched"]),
            "year": sum(1 for r in records if "year" in r["enriched"]),
            "grape": sum(1 for r in records if "grape" in r["enriched"]),
        },
    }
    _write_report(args, report)
    log_marker("M-TEXT", "enrich", "DONE", targets=len(targets), enriched=enriched_count,
               ambiguous=len(ambiguous_slugs), vlm=vlm_used)
    print("=== KB ENRICH (Ф1) ===")
    print(f"целевых slug: {len(targets)} из {len(near)} групп")
    print(f"обогащено:    {enriched_count} ({report['enriched_by_field']})")
    print(f"12b вызовов:  {vlm_used} ({report['vlm_budget_note']})")
    print(f"ambiguous:    {len(ambiguous_slugs)} slug в {len(ambiguous)} группах (все поля совпали)")
    print(f"color-only:   {len(color_only)} групп различаются только свободным color")
    print(f"catalog.jsonl не изменён: True")
    print(f"out: {enriched_path}")
    return 0


def _write_report(args: argparse.Namespace, report: dict) -> None:
    """Пишет JSON-отчёт, если путь задан."""
    if not args.report:
        return
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report: {target}")


def main(argv: list[str] | None = None) -> int:
    """Запускает пилот или обогащение."""
    args = parse_args(argv)
    catalog_path = _ROOT / settings.catalog_jsonl
    catalog = load_catalog(catalog_path)
    uploads = Path(args.uploads)
    before = catalog_path.read_bytes()
    code = _pilot(args, catalog, uploads) if args.command == "pilot" else _enrich(args, catalog, uploads)
    if catalog_path.read_bytes() != before:
        print("ERROR: исходный catalog.jsonl изменён — это запрещено", file=sys.stderr)
        return 3
    return code


if __name__ == "__main__":
    raise SystemExit(main())
