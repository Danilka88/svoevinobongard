#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Медиа-обогащение каталога из уже скачанного кэша официального API
# SCOPE: parse_args, build_record, build_media, main
# DEPENDS: json, pathlib, packages.common.config, packages.common.logging, packages.common.official_api
# LINKS: M-CATALOG, M-GATEWAY
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# DEFAULT_CACHE - каталог кэша деталей API
# DEFAULT_REGIONS - файл кэша регионов (geojson и счётчики)
# DEFAULT_OUT - файл медиа-обогащения
# parse_args - аргументы CLI
# build_record - медиа-запись по одной детали вина
# build_media - сборка файла медиа с возобновлением и проверкой кэша
# main - точка входа
# END_MODULE_MAP
"""Медиа-обогащение каталога.

Источник — уже скачанное в C-017 (``data/api_cache/wines/*.json``), новых сетевых вызовов
нет. Файл пишется **отдельно** от каталога, поэтому ``catalog.jsonl`` и ``catalog_enriched_v3``
не меняются. Все URL приводятся к абсолютным через ``official_api.absolute_url``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from packages.common.logging import log_marker  # noqa: E402
from packages.common.official_api import absolute_url  # noqa: E402

DEFAULT_CACHE = "data/api_cache/wines"
DEFAULT_REGIONS = "data/api_cache/regions.json"
DEFAULT_OUT = "data/catalog/catalog_media.json"

__all__ = ["DEFAULT_CACHE", "DEFAULT_OUT", "DEFAULT_REGIONS", "build_media", "build_record", "main", "parse_args"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы CLI."""
    parser = argparse.ArgumentParser(description="Медиа-обогащение каталога из кэша API")
    parser.add_argument("--cache", default=str(_ROOT / DEFAULT_CACHE))
    parser.add_argument("--regions", default=str(_ROOT / DEFAULT_REGIONS))
    parser.add_argument("--out", default=str(_ROOT / DEFAULT_OUT))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true", default=True)
    return parser.parse_args(argv)


def _image_url(node: object) -> str:
    """Абсолютный URL изображения из узла API вида {"url": ...}."""
    if isinstance(node, dict):
        return absolute_url(str(node.get("url") or ""))
    return ""


def build_record(detail: dict, region_meta: dict) -> dict:
    """Собирает медиа-запись по одной детали вина.

    Args:
        detail: закэшированная деталь вина.
        region_meta: соответствие нормализованного имени региона -> запись регионов.

    Returns:
        Словарь медиа-полей; отсутствующие значения остаются пустыми.
    """
    region = detail.get("region") if isinstance(detail.get("region"), dict) else {}
    region_name = str(region.get("name") or "")
    meta = region_meta.get(region_name.lower()) or {}
    maker = detail.get("manufacturer") if isinstance(detail.get("manufacturer"), dict) else {}
    grapes = []
    for grape in detail.get("grapes") or []:
        if isinstance(grape, dict):
            grapes.append({"name": str(grape.get("name") or ""), "url": _image_url(grape.get("image"))})
    dishes = []
    for dish in detail.get("dishes") or []:
        if isinstance(dish, dict):
            dishes.append({"name": str(dish.get("name") or ""), "url": _image_url(dish.get("image"))})
    return {
        "image_url": _image_url(detail.get("image")),
        "category_gradient": str((detail.get("category") or {}).get("backgroundGradient") or ""),
        "category_official": str((detail.get("category") or {}).get("name") or ""),
        "region_name": region_name,
        "region_image_url": _image_url(region.get("image")),
        "region_geojson_url": absolute_url(str(meta.get("geojsonUrl") or "")),
        "manufacturer_name": str(maker.get("name") or ""),
        "manufacturer_slug": str(maker.get("slug") or ""),
        "grape_images": [item for item in grapes if item["url"]],
        "dish_images": [item for item in dishes if item["url"]],
    }


def build_media(args: argparse.Namespace) -> dict:
    """Собирает файл медиа с возобновлением и проверкой неизменности кэша."""
    cache = Path(args.cache)
    out_path = Path(args.out)
    files = sorted(cache.glob("*.json"))
    if args.limit:
        files = files[: args.limit]
    before = {path.name: (path.stat().st_size, path.stat().st_mtime) for path in files}

    regions_payload = {}
    regions_file = Path(args.regions)
    if regions_file.is_file():
        regions_payload = json.loads(regions_file.read_text(encoding="utf-8"))
    region_meta = {str(item.get("name") or "").lower(): item for item in (regions_payload.get("items") or [])}

    existing: dict = {}
    if args.resume and out_path.is_file():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    records = dict(existing)
    processed = 0
    for path in files:
        if path.name in records:
            continue
        try:
            detail = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        slug = str(detail.get("slug") or path.stem)
        records[path.name] = {"slug": slug, **build_record(detail, region_meta)}
        processed += 1
        if processed % 500 == 0:
            print(f"  медиа: {processed}/{len(files)}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    after = {path.name: (path.stat().st_size, path.stat().st_mtime) for path in files}
    if before != after:
        print("ERROR: исходный кэш API изменён — это запрещено", file=sys.stderr)
        return {}
    total = len(records)
    with_images = sum(1 for record in records.values() if record.get("image_url"))
    with_dishes = sum(1 for record in records.values() if record.get("dish_images"))
    with_geojson = sum(1 for record in records.values() if record.get("region_geojson_url"))
    log_marker("M-CATALOG", "media", "DONE", records=total, processed=processed,
               images=with_images, dishes=with_dishes, geojson=with_geojson)
    print(f"медиа-записей: {total} (посчитано сейчас {processed})")
    print(f"  с картинкой: {with_images} · с картинками блюд: {with_dishes} · с geojson региона: {with_geojson}")
    print("  исходный кэш не изменён: True")
    return records


def main(argv: list[str] | None = None) -> int:
    """Запускает сборку медиа."""
    args = parse_args(argv)
    build_media(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
