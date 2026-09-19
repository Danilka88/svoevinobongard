# START_MODULE_CONTRACT
# PURPOSE: Энотуризм и статьи для удержания: региональный факт, туры региона и релевантные материалы портала
# SCOPE: load_collections, tour_url, region_of, tours_for, articles_for, enotourism_for
# DEPENDS: json, pathlib, packages.common.config, packages.common.logging, packages.common.official_api
# LINKS: M-ENOTOURISM, M-GATEWAY, M-COMMON
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# COLLECTIONS - имена коллекций API, нужные фиче энотуризма
# load_collections - читает закэшированные туры, регионы и статьи
# tour_url - ссылка на страницу тура на портале
# region_of - название региона вина из официального обогащения
# tours_for - туры, относящиеся к региону или винодельне вина
# articles_for - релевантные статьи портала по региону, категории и сорту
# enotourism_for - собирает полный ответ энотуризма для slug
# END_MODULE_MAP
"""Энотуризм и «Истории в бокале».

Данные берутся только из официального API (туры, регионы, статьи) — ничего не
генерируется моделью. Тяжёлого поиска нет: релевантность статей — совпадение
названия региона, категории вина и сорта в тексте материала. Если совпадений
меньше трёх, подборка добирается свежими материалами портала, и каждый элемент
несёт поле relevance, чтобы источник решения был виден.

При отсутствии кэша функции возвращают пустые списки: фича удержания не должна
ломать основной сценарий сканирования.
"""
from __future__ import annotations

import json
from pathlib import Path

from packages.common.logging import log_marker
from packages.common.official_api import absolute_url

COLLECTIONS = ("tours", "regions", "articles")
DEFAULT_CACHE = "data/api_cache"
ARTICLES_MIN = 3
ARTICLES_MAX = 5
PORTAL = "https://vino-svoe.ru"

__all__ = ["COLLECTIONS", "load_collections", "tour_url", "region_of",
           "tours_for", "articles_for", "enotourism_for"]


def _read_json(path: Path) -> object | None:
    """Читает JSON, молча возвращая None при отсутствии или битом файле."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log_marker("M-ENOTOURISM", "load", "CORRUPT", path=str(path))
        return None


def load_collections(cache_dir: str | Path = DEFAULT_CACHE) -> dict:
    """Читает туры, регионы и статьи из кэша синхронизации.

    Returns:
        Словарь с ключами tours, regions, articles (списки; пустые, если кэша нет).
    """
    base = Path(cache_dir)
    tours_payload = _read_json(base / "tours.json") or {}
    regions_payload = _read_json(base / "regions.json") or {}
    articles: list[dict] = []
    articles_dir = base / "articles"
    if articles_dir.is_dir():
        for page in sorted(articles_dir.glob("page_*.json")):
            payload = _read_json(page) or {}
            articles.extend(payload.get("items") or [])
    result = {
        "tours": list(tours_payload.get("items") or []),
        "regions": list(regions_payload.get("items") or []),
        "articles": articles,
    }
    log_marker("M-ENOTOURISM", "load", "DONE", tours=len(result["tours"]),
               regions=len(result["regions"]), articles=len(articles))
    return result


def _text_of(item: dict) -> str:
    """Склеивает строковые значения элемента (включая вложенные картинки)."""
    parts: list[str] = []
    for value in (item or {}).values():
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, dict):
            parts.extend(str(inner) for inner in value.values() if isinstance(inner, str))
        elif isinstance(value, list):
            parts.extend(str(inner) for inner in value if isinstance(inner, str))
    return " ".join(parts).lower()


def tour_url(slug: str) -> str:
    """Ссылка на тур на портале."""
    return f"{PORTAL}/tours/{slug}" if slug else PORTAL


def region_of(official: dict | None, fallback: str = "") -> str:
    """Название региона вина: из официального обогащения, иначе из каталога."""
    record = official or {}
    return str(record.get("region_name") or record.get("region") or fallback or "")


def _region_fact(regions: list[dict], region: str) -> dict:
    """Факт и события региона по его названию."""
    needle = (region or "").strip().lower()
    for item in regions:
        if needle and needle in _text_of(item):
            return {
                "name": str(item.get("name") or region),
                "fact": str(item.get("fact") or ""),
                "events_count": int(item.get("eventsCount") or 0),
                "events_description": str(item.get("eventsCountDescription") or ""),
            }
    return {"name": region, "fact": "", "events_count": 0, "events_description": ""}


def tours_for(tours: list[dict], region: str, winery: str = "", limit: int = 3) -> list[dict]:
    """Туры, относящиеся к региону или винодельне вина.

    Сначала туры винодельни (совпадение в названии/адресе), затем туры региона;
    если ничего не совпало — пустой список (выдумывать «рядом» нельзя).
    """
    winery_needle = (winery or "").strip().lower()
    region_needle = (region or "").strip().lower()
    by_winery: list[dict] = []
    by_region: list[dict] = []
    for item in tours:
        haystack = _text_of(item)
        payload = {
            "slug": str(item.get("slug") or ""),
            "title": str(item.get("title") or item.get("name") or ""),
            "address": str(item.get("address") or ""),
            "price": item.get("price"),
            "image_url": absolute_url((item.get("image") or {}).get("url") if isinstance(item.get("image"), dict) else ""),
            "url": tour_url(str(item.get("slug") or "")),
            "match": "",
        }
        if winery_needle and winery_needle in haystack:
            payload["match"] = "winery"
            by_winery.append(payload)
        elif region_needle and region_needle in haystack:
            payload["match"] = "region"
            by_region.append(payload)
    return (by_winery + by_region)[:limit]


def articles_for(articles: list[dict], region: str, category: str = "", grape: str = "",
                 limit: int = ARTICLES_MAX) -> list[dict]:
    """Релевантные статьи портала: регион весит больше категории, категория — больше сорта.

    Если релевантных меньше трёх, подборка добирается свежими материалами с пометкой latest.
    """
    region_needle = (region or "").strip().lower()
    category_needle = (category or "").strip().lower()
    grape_needle = (grape or "").strip().lower().split(",")[0].strip()
    scored: list[tuple[int, dict]] = []
    for item in articles:
        haystack = _text_of(item)
        score = 0
        if region_needle and region_needle in haystack:
            score += 3
        if category_needle and category_needle in haystack:
            score += 2
        if grape_needle and grape_needle in haystack:
            score += 1
        scored.append((score, item))
    hay_signal = bool(region_needle or category_needle or grape_needle)
    if not hay_signal:
        # не о чем судить: у вина нет ни региона, ни категории — показывать «свежее» было бы нерелевантно
        return []
    scored.sort(key=lambda pair: (-pair[0], str(pair[1].get("firstPublishedAt") or "")), reverse=False)
    matched = [item for score, item in scored if score > 0]
    latest = [item for score, item in scored if score == 0]
    picked = matched[:limit]
    relevance = {str(item.get("slug")): "matched" for item in picked}
    if len(picked) < ARTICLES_MIN:
        for item in latest:
            if len(picked) >= ARTICLES_MIN:
                break
            picked.append(item)
            relevance[str(item.get("slug"))] = "latest"
    seen: set[str] = set()
    out: list[dict] = []
    for item in picked:
        slug = str(item.get("slug") or "")
        if not slug or slug in seen:
            continue
        seen.add(slug)
        image = item.get("image") if isinstance(item.get("image"), dict) else {}
        out.append({
            "slug": slug,
            "title": str(item.get("title") or ""),
            "category": str(item.get("category") or ""),
            "author": str(item.get("author") or ""),
            "published_at": str(item.get("firstPublishedAt") or ""),
            "image_url": absolute_url(image.get("url") or ""),
            "url": f"{PORTAL}/articles/{slug}",
            "relevance": relevance.get(slug, "latest"),
        })
    return out[:limit]


def enotourism_for(slug: str, official: dict | None = None, fallback_region: str = "",
                   fallback_winery: str = "", cache_dir: str | Path = DEFAULT_CACHE) -> dict:
    """Собирает ответ энотуризма для вина: регион с фактом, туры и подборка статей."""
    collections = load_collections(cache_dir)
    record = official or {}
    region = region_of(record, fallback_region)
    winery = str(record.get("manufacturer_name") or fallback_winery or "")
    grapes = record.get("grapes") or []
    grape = str(grapes[0]) if grapes else str(record.get("grape") or "")
    category = str(record.get("category_official") or record.get("category") or "")
    fact = _region_fact(collections["regions"], region)
    tours = tours_for(collections["tours"], region, winery)
    articles = articles_for(collections["articles"], region, category, grape)
    note = ""
    if not collections["tours"] and not collections["articles"]:
        note = "collections cache is empty: enotourism data unavailable"
    elif not region and not category:
        note = "wine metadata is unknown: nothing to match tours or articles against"
    payload = {
        "slug": slug,
        "region": fact,
        "tours": tours,
        "articles": articles,
        "source": "official-api",
        "note": note,
    }
    log_marker("M-ENOTOURISM", "build", "DONE", slug=slug, region=region,
               tours=len(tours), articles=len(articles))
    return payload
