#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: ETL каталога «Своё Вино» — CSV и wines_sitemap XML в catalog.jsonl, slug_to_image.json и stats.json
# SCOPE: parse_sitemap, load_csv, merge_slug, detect_year, detect_sweetness, main; дедупликация 4147 строк CSV в 2103 slug
# DEPENDS: none (только stdlib)
# LINKS: M-CATALOG
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# YEAR_RE - регулярное выражение года в названии
# SWEETNESS - словарь нормализации сладости
# norm - нормализация пробелов и nbsp
# parse_sitemap - slug в filename по wines_sitemap XML
# detect_sweetness - определение сладости по name, slug, category, grape, description
# detect_year - извлечение года из name и slug
# load_csv - чтение CSV каталога с нормализацией ключей
# merge_slug - схлопывание дублирующихся строк одного slug
# main - CLI: ETL и отчёт
# END_MODULE_MAP
"""ETL каталога «Своё Вино».

Источники:
  - strapi_output*.csv                — метаданные каталога
  - wines_sitemap_*.xml               — авторитетный маппинг slug -> файл эталонного фото

Выход:
  - catalog.jsonl                     — по одному объекту на slug
  - slug_to_image.json                — {slug: image_filename|null}
  - stats.json                        — сводка

Зависимостей нет (только stdlib).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

__all__ = [
    "YEAR_RE",
    "SWEETNESS",
    "norm",
    "parse_sitemap",
    "detect_sweetness",
    "detect_year",
    "load_csv",
    "merge_slug",
    "main",
]

# START_BLOCK_ETL_CONSTANTS
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

SWEETNESS = [
    ("экстра брют", "экстра брют"),
    ("брют натюр", "брют натюр"),
    ("брют", "брют"),
    ("полусладкое", "полусладкое"),
    ("полусухое", "полусухое"),
    ("сладкое", "сладкое"),
    ("сухое", "сухое"),
]
# END_BLOCK_ETL_CONSTANTS


# START_BLOCK_ETL_NORM
def norm(value: str | None) -> str:
    return " ".join((value or "").replace("\xa0", " ").split())
# END_BLOCK_ETL_NORM


# START_BLOCK_ETL_PARSE_SITEMAP
def parse_sitemap(path: Path) -> dict[str, str]:
    """slug -> filename (последний сегмент image:loc)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    pairs = re.findall(
        r"<loc>(.*?)</loc>.*?<image:loc>(.*?)</image:loc>", text, re.S
    )
    mapping: dict[str, str] = {}
    for loc, img in pairs:
        slug = loc.rstrip("/").rsplit("/", 1)[-1]
        mapping[slug] = img.rsplit("/", 1)[-1]
    print(f"[M-CATALOG][PARSE_SITEMAP][DONE] entries={len(mapping)}", file=sys.stderr)
    return mapping
# END_BLOCK_ETL_PARSE_SITEMAP


# START_BLOCK_ETL_DETECT
def detect_sweetness(*parts: str) -> str | None:
    hay = " ".join(p.lower() for p in parts if p)
    for needle, label in SWEETNESS:
        if needle in hay:
            return label
    return None


def detect_year(*parts: str) -> int | None:
    for p in parts:
        if not p:
            continue
        m = YEAR_RE.search(p)
        if m:
            return int(m.group(0))
        m = re.search(r"[-\s](20\d{2}|19\d{2})(?:$|[-_\s])", p)
        if m:
            return int(m.group(1))
    return None
# END_BLOCK_ETL_DETECT


# START_BLOCK_ETL_LOAD_CSV
def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = []
        for raw in reader:
            row = {norm(k): norm(v) for k, v in raw.items() if k is not None}
            if row.get("Slug"):
                rows.append(row)
    return rows
# END_BLOCK_ETL_LOAD_CSV


# START_BLOCK_ETL_MERGE
def merge_slug(rows: list[dict]) -> dict:
    """Схлопнуть (возможно дублирующиеся) строки одного slug."""
    fields = ["Название вина", "Категория", "Цвет", "Регион",
              "Сорт винограда", "Описание", "Винодельня", "Название фото"]
    merged: dict[str, str] = {}
    for field in fields:
        for row in rows:
            val = norm(row.get(field))
            if val:
                merged[field] = val
                break
    return merged
# END_BLOCK_ETL_MERGE


# START_BLOCK_ETL_MAIN
def main() -> int:
    ap = argparse.ArgumentParser(description="ETL каталога «Своё Вино»")
    here = Path(__file__).resolve()
    default_materials = here.parents[3] / "материалы для конкурса" / "Датасет"
    ap.add_argument("--materials", default=str(default_materials),
                    help="папка с датасетом (CSV + strapi uploads)")
    ap.add_argument("--out", default=str(here.parents[2] / "data" / "catalog"),
                    help="папка для catalog.jsonl / slug_to_image.json / stats.json")
    args = ap.parse_args()

    materials = Path(args.materials)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[M-CATALOG][ETL][START] materials={materials} out={out}", file=sys.stderr)

    if not materials.is_dir():
        print(f"ERROR: materials dir not found: {materials}", file=sys.stderr)
        return 2

    csv_files = sorted(materials.glob("*.csv"))
    sitemaps = sorted(materials.glob("**/wines_sitemap_*.xml"))
    if not csv_files:
        print(f"ERROR: no CSV in {materials}", file=sys.stderr)
        return 2
    if not sitemaps:
        print(f"ERROR: no wines_sitemap_*.xml in {materials}", file=sys.stderr)
        return 2

    csv_path, sitemap_path = csv_files[0], sitemaps[0]
    print(f"[i] CSV:     {csv_path.name}")
    print(f"[i] sitemap: {sitemap_path.name}")

    slug2image = parse_sitemap(sitemap_path)
    rows = load_csv(csv_path)

    by_slug: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_slug[row["Slug"]].append(row)

    records: list[dict] = []
    for slug, slug_rows in sorted(by_slug.items()):
        m = merge_slug(slug_rows)
        name = m.get("Название вина", "")
        category = m.get("Категория", "")
        color = m.get("Цвет", "")
        grape = m.get("Сорт винограда", "")
        description = m.get("Описание", "")
        winery = m.get("Винодельня", "")
        image = slug2image.get(slug)

        records.append({
            "slug": slug,
            "name": name,
            "category": category,
            "color": color,
            "region": m.get("Регион", ""),
            "grape": grape,
            "description": description,
            "winery": winery,
            "year": detect_year(name, slug),
            "sweetness": detect_sweetness(name, slug, category, grape, description),
            "image": image,
            "has_image": image is not None,
        })

    catalog_path = out / "catalog.jsonl"
    with catalog_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    slug_to_image = {rec["slug"]: rec["image"] for rec in records}
    (out / "slug_to_image.json").write_text(
        json.dumps(slug_to_image, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    images = [r["image"] for r in records if r["image"]]
    img_counter = Counter(images)
    shared_images = {img: c for img, c in img_counter.items() if c > 1}

    stats = {
        "csv_rows": len(rows),
        "unique_slugs": len(records),
        "with_image": sum(1 for r in records if r["has_image"]),
        "without_image": sum(1 for r in records if not r["has_image"]),
        "sitemap_entries": len(slug2image),
        "shared_reference_images": len(shared_images),
        "wineries": len({r["winery"] for r in records if r["winery"]}),
        "regions": dict(Counter(r["region"] for r in records if r["region"]).most_common()),
        "categories": dict(Counter(r["category"] for r in records if r["category"]).most_common()),
        "years_found": sum(1 for r in records if r["year"]),
        "sweetness_found": sum(1 for r in records if r["sweetness"]),
        "examples_shared_images": {k: v for k, v in list(shared_images.items())[:10]},
    }
    (out / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== ETL REPORT ===")
    print(f"CSV строк:              {stats['csv_rows']}")
    print(f"Уникальных slug:        {stats['unique_slugs']}")
    print(f"  с эталонным фото:     {stats['with_image']}")
    print(f"  без фото:             {stats['without_image']}")
    print(f"Виноделен:              {stats['wineries']}")
    print(f"Год определён:          {stats['years_found']}")
    print(f"Сладость определена:    {stats['sweetness_found']}")
    print(f"Общих эталонных фото:   {stats['shared_reference_images']}")
    print(f"\nВыход:\n  {catalog_path}\n  {out/'slug_to_image.json'}\n  {out/'stats.json'}")
    print(f"[M-CATALOG][ETL][DONE] slugs={stats['unique_slugs']} "
          f"with_image={stats['with_image']} without_image={stats['without_image']}",
          file=sys.stderr)
    return 0
# END_BLOCK_ETL_MAIN


if __name__ == "__main__":
    raise SystemExit(main())
