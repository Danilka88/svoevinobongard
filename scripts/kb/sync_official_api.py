#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Синхронизация официального API vino-svoe.ru в локальный кэш и сборка catalog_enriched_v3
# SCOPE: parse_args, fetch_json, sync_list, sync_details, sync_collections, build_v3, main
# DEPENDS: json, urllib, packages.common.catalog, packages.common.config, packages.common.logging,
#          services.matcher.app.textfusion
# LINKS: M-CATALOG, M-TEXT, M-DEPLOY
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# API_BASE - базовый URL публичного API
# PER_PAGE_MAX - максимальный размер страницы списка (проверено: 30)
# CACHE_DIR - каталог кэша ответов API по умолчанию
# parse_args - аргументы CLI (list, details, collections, build)
# fetch_json - запрос с ретраями, таймаутом и вежливой паузой
# sync_list - выгрузка списка вин по страницам в кэш
# sync_details - выгрузка деталей для slug, совпавших с нашим каталогом
# sync_collections - производители, регионы, туры, статьи
# build_v3 - сборка data/catalog/catalog_enriched_v3.jsonl
# main - точка входа
# END_MODULE_MAP
"""Синхронизация официального API.

Каталог-истина для оценки — CSV/sitemap, а API используется только как источник
обогащения по совпадающим slug (живой API не знает часть наших slug — например
``kokur-suhoe-2025`` отдаёт 404).

Все ответы кладутся в ``data/api_cache/`` (gitignored) по одному файлу на запрос,
поэтому любой шаг можно прервать и продолжить флагом ``--resume``. Базовые
``catalog.jsonl`` и ``catalog_enriched*.jsonl`` не перезаписываются: это
проверяется сравнением байт до и после.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from packages.common.catalog import load_catalog  # noqa: E402
from packages.common.config import settings  # noqa: E402
from packages.common.logging import log_marker  # noqa: E402
from services.matcher.app.textfusion import near_dup_key  # noqa: E402

API_BASE = "https://api.vino-svoe.ru/v1"
PER_PAGE_MAX = 30
CACHE_DIR = "data/api_cache"
_UA = "svoe-vino-scanner/1.0 (+offline enrichment; contact: project owner)"
_YEAR_RE = None  # заполняется лениво, чтобы не тянуть re на импорте

__all__ = ["API_BASE", "PER_PAGE_MAX", "parse_args", "fetch_json", "sync_list",
           "sync_details", "sync_collections", "build_v3", "main"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы CLI."""
    parser = argparse.ArgumentParser(description="Синхронизация официального API")
    parser.add_argument("command", choices=["list", "details", "collections", "build", "all"])
    parser.add_argument("--cache", default=CACHE_DIR)
    parser.add_argument("--out-v3", default="data/catalog/catalog_enriched_v3.jsonl")
    parser.add_argument("--report", default="")
    parser.add_argument("--pause", type=float, default=0.15, help="пауза между запросами, с")
    parser.add_argument("--timeout", type=float, default=25.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="ограничить число деталей (0 — без лимита)")
    parser.add_argument("--time-budget-min", type=float, default=20.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--articles", type=int, default=150, help="сколько статей синкать")
    return parser.parse_args(argv)


def fetch_json(url: str, timeout: float = 25.0, retries: int = 3, pause: float = 0.15) -> object:
    """GET с ретраями и вежливой паузой; возвращает распарсенный JSON."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            time.sleep(pause)
            return payload
        except Exception as error:  # noqa: BLE001 - сеть шумит, нужен повтор
            last = error
            time.sleep(pause + 0.4 * (attempt + 1))
    log_marker("M-CATALOG", "api", "FETCH_FAILED", url=url[:120], error=type(last).__name__)
    raise RuntimeError(f"fetch failed: {url}: {last}")


def _cached(path: Path, resume: bool) -> object | None:
    """Возвращает содержимое кэша, если файл есть и включён resume."""
    if resume and path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


def _store(path: Path, payload: object) -> None:
    """Пишет JSON в кэш."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def sync_list(args: argparse.Namespace) -> list[dict]:
    """Выгружает список вин по страницам; возвращает все items."""
    cache = _ROOT / args.cache / "wines_list"
    first = _cached(cache / "page_001.json", args.resume)
    if first is None:
        first = fetch_json(f"{API_BASE}/wines?page=1&perPage={PER_PAGE_MAX}", args.timeout, args.retries, args.pause)
        _store(cache / "page_001.json", first)
    total_pages = int(first.get("totalPages") or 1)
    items = list(first.get("items") or [])
    for page in range(2, total_pages + 1):
        target = cache / f"page_{page:03d}.json"
        payload = _cached(target, args.resume)
        if payload is None:
            payload = fetch_json(f"{API_BASE}/wines?page={page}&perPage={PER_PAGE_MAX}",
                                 args.timeout, args.retries, args.pause)
            _store(target, payload)
        items.extend(payload.get("items") or [])
    log_marker("M-CATALOG", "api_sync", "LIST_DONE", pages=total_pages, items=len(items))
    print(f"список: {len(items)} позиций из {total_pages} страниц (perPage={PER_PAGE_MAX})")
    return items


def _api_slugs(items: list[dict]) -> dict[str, dict]:
    """Индекс slug -> карточка списка."""
    return {str(item.get("slug")): item for item in items if item.get("slug")}


def sync_details(args: argparse.Namespace, items: list[dict]) -> dict[str, dict]:
    """Выгружает детали для slug, совпавших с нашим каталогом."""
    cache = _ROOT / args.cache / "wines"
    store = load_catalog(_ROOT / settings.catalog_jsonl)
    api = _api_slugs(items)
    matched = sorted(set(store) & set(api))
    missing = sorted(set(store) - set(api))
    print(f"совпало slug с API: {len(matched)} из {len(store)} (нет в API: {len(missing)})")
    pending = [slug for slug in matched if not (args.resume and (cache / f"{slug}.json").is_file())]
    if args.limit:
        pending = pending[: args.limit]
    started = time.perf_counter()
    fetched = 0
    for slug in pending:
        if (time.perf_counter() - started) / 60.0 > args.time_budget_min:
            print(f"TIME-BOX {args.time_budget_min} мин: скачано {fetched}, осталось {len(pending) - fetched}")
            break
        target = cache / f"{slug}.json"
        if target.is_file() and args.resume:
            continue
        try:
            _store(target, fetch_json(f"{API_BASE}/wines/{slug}", args.timeout, args.retries, args.pause))
            fetched += 1
        except RuntimeError as error:
            log_marker("M-CATALOG", "api_sync", "DETAIL_FAILED", slug=slug, error=str(error)[:120])
        if fetched and fetched % 100 == 0:
            rate = (time.perf_counter() - started) / fetched
            print(f"  детали: {fetched}/{len(pending)} · {rate*1000:.0f} мс/шт", flush=True)
    log_marker("M-CATALOG", "api_sync", "DETAILS_DONE", matched=len(matched), fetched=fetched)
    return {"matched": matched, "missing": missing}


def sync_collections(args: argparse.Namespace) -> dict:
    """Синкает производителей, регионы, туры и статьи."""
    cache = _ROOT / args.cache
    result = {}
    for name, url in (("manufacturers", f"{API_BASE}/manufacturers"),
                      ("regions", f"{API_BASE}/regions?page=1&perPage={PER_PAGE_MAX}"),
                      ("tours", f"{API_BASE}/tours?page=1&perPage={PER_PAGE_MAX}")):
        target = cache / f"{name}.json"
        payload = _cached(target, args.resume)
        if payload is None:
            payload = fetch_json(url, args.timeout, args.retries, args.pause)
            _store(target, payload)
        result[name] = payload
    pages = max(1, -(-args.articles // PER_PAGE_MAX))
    articles: list[dict] = []
    for page in range(1, pages + 1):
        target = cache / "articles" / f"page_{page:03d}.json"
        payload = _cached(target, args.resume)
        if payload is None:
            payload = fetch_json(f"{API_BASE}/articles?page={page}&perPage={PER_PAGE_MAX}",
                                 args.timeout, args.retries, args.pause)
            _store(target, payload)
        articles.extend(payload.get("items") or [])
    result["articles"] = articles
    log_marker("M-CATALOG", "api_sync", "COLLECTIONS_DONE", articles=len(articles))
    print(f"коллекции: производители={len(result['manufacturers'])}, регионы="
          f"{len((result['regions'] or {}).get('items', []))}, туры="
          f"{len((result['tours'] or {}).get('items', []))}, статьи={len(articles)}")
    return result


def _detail_for(slug: str, cache: Path) -> dict | None:
    """Читает закэшированные детали вина."""
    target = cache / f"{slug}.json"
    if not target.is_file():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def build_v3(args: argparse.Namespace, items: list[dict]) -> dict:
    """Собирает catalog_enriched_v3.jsonl из кэша деталей."""
    cache = _ROOT / args.cache / "wines"
    ocr_enriched = _ROOT / "data" / "catalog" / "catalog_enriched.jsonl"
    vlm_enriched = _ROOT / "data" / "catalog" / "catalog_vlm_enriched.jsonl"
    base = load_catalog(_ROOT / settings.catalog_jsonl)
    api = _api_slugs(items)
    extra: dict[str, dict] = {}
    for path in (ocr_enriched, vlm_enriched):
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("slug"):
                    extra.setdefault(str(record["slug"]), {}).update(
                        record.get("enriched") or record.get("fields") or {})
    out = _ROOT / args.out_v3
    stats = {"total": len(base), "api_match": 0, "details": 0, "alcohol": 0, "dishes": 0,
             "grapes": 0, "temperature": 0, "region": 0, "description": 0, "near_dup_api": 0}
    groups: dict[str, list[str]] = {}
    for slug, record in base.items():
        groups.setdefault(near_dup_key(record), []).append(slug)
    near = {slug for slugs in groups.values() if len(slugs) > 1 for slug in slugs}

    with out.open("w", encoding="utf-8") as handle:
        for slug, record in base.items():
            listed = api.get(slug)
            detail = _detail_for(slug, cache)
            row = dict(record)
            row["source"] = "official+csv" if detail else ("csv+official-list" if listed else "csv")
            row["api_match"] = bool(detail)
            if listed:
                stats["api_match"] += 1
                listed_image = (listed.get("image") or {}).get("url", "")
                row["image_path"] = listed_image
                row["image_url"] = (listed_image if listed_image.startswith("http")
                                    else f"{API_BASE.rsplit('/v1', 1)[0]}{listed_image}")
                row["api_category"] = listed.get("category", "")
                row["api_manufacturer"] = listed.get("manufacturer", "")
                row["api_region"] = listed.get("region", "")
                row["api_category"] = listed.get("category", "")
            if detail:
                stats["details"] += 1
                alcohol = detail.get("alcohol")
                # API хранит крепость целым процентом (12, 13, 11), и лишь единичные записи —
                # в десятых долях (135 = 13.5, таких 2 из 2035). Нормализуем по величине.
                if isinstance(alcohol, (int, float)):
                    row["alcohol_pct"] = round(float(alcohol) / 10.0, 1) if alcohol >= 40 else round(float(alcohol), 1)
                else:
                    row["alcohol_pct"] = None
                row["temperature"] = detail.get("temperature", "")
                row["description"] = detail.get("description", "")
                row["dishes"] = [dish.get("name", "") for dish in (detail.get("dishes") or [])]
                row["grapes"] = [g.get("name", "") if isinstance(g, dict) else str(g)
                                 for g in (detail.get("grapes") or []) if g]
                region = detail.get("region")
                row["region_name"] = region.get("name", "") if isinstance(region, dict) else str(region or "")
                maker = detail.get("manufacturer")
                if isinstance(maker, dict):
                    row["manufacturer_name"] = maker.get("name", "")
                    row["manufacturer_slug"] = maker.get("slug", "")
                else:
                    row["manufacturer_name"] = str(maker or "")
                    row["manufacturer_slug"] = ""
                # свободный цвет API не перетирает CSV-поле color (на него опирается нормализация категории)
                row["color_official"] = detail.get("color", "")
                row["category_official"] = (detail.get("category") or {}).get("name", "")
                path_url = (detail.get("image") or {}).get("url", "") or row.get("image_url", "")
                row["image_path"] = path_url
                row["image_url"] = path_url if path_url.startswith("http") else f"{API_BASE.rsplit('/v1', 1)[0]}{path_url}"
                stats["alcohol"] += bool(row["alcohol_pct"])
                stats["dishes"] += bool(row["dishes"])
                stats["grapes"] += bool(row["grapes"])
                stats["temperature"] += bool(row["temperature"])
                stats["region"] += bool(row["region_name"])
                stats["description"] += bool(row["description"])
                if slug in near:
                    stats["near_dup_api"] += 1
            if extra.get(slug):
                row["enriched"] = {**extra[slug], **(row.get("enriched") or {})}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    log_marker("M-CATALOG", "api_build", "DONE", **{k: v for k, v in stats.items()})
    return stats


def main(argv: list[str] | None = None) -> int:
    """Выполняет шаги синхронизации и/или сборку."""
    args = parse_args(argv)
    guarded = [_ROOT / settings.catalog_jsonl, _ROOT / "data/catalog/catalog_enriched.jsonl",
               _ROOT / "data/catalog/catalog_vlm_enriched.jsonl"]
    before = {path: path.read_bytes() for path in guarded if path.is_file()}
    try:
        items = sync_list(args) if args.command in {"list", "details", "all", "build"} else []
        details_info: dict = {}
        if args.command in {"details", "all"}:
            details_info = sync_details(args, items)
        if args.command in {"collections", "all"}:
            sync_collections(args)
        if args.command in {"build", "all"}:
            stats = build_v3(args, items)
            print("=== catalog_enriched_v3 ===")
            print(f"  всего slug: {stats['total']} · совпало с API: {stats['api_match']} · с деталями: {stats['details']}")
            print(f"  alcohol: {stats['alcohol']} · dishes: {stats['dishes']} · grapes: {stats['grapes']} · "
                  f"temperature: {stats['temperature']} · region: {stats['region']} · description: {stats['description']}")
            print(f"  из near-dup slug получили детали: {stats['near_dup_api']}")
            if args.report:
                report = {"stats": stats, "api_matched": len(details_info.get("matched") or []),
                          "api_missing": len(details_info.get("missing") or []),
                          "missing_examples": sorted(details_info.get("missing") or [])[:15],
                          "per_page_max": PER_PAGE_MAX}
                target = _ROOT / args.report
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"  report: {target.relative_to(_ROOT)}")
    finally:
        for path, payload in before.items():
            if path.read_bytes() != payload:
                print(f"ERROR: запрещённый файл изменён: {path}", file=sys.stderr)
                return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
