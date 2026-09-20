# START_MODULE_CONTRACT
# PURPOSE: Медиа-обогащение каталога, метрики демо и браузер каталога для UI
# SCOPE: load_media, media_for, demo_metrics, catalog_search
# DEPENDS: json, pathlib, packages.common.catalog, packages.common.logging, packages.common.text_norm
# LINKS: M-GATEWAY, M-CATALOG, M-WEB
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# DEFAULT_MEDIA - файл медиа-обогащения (отдельно от каталога)
# REPORT_FILES - отчёты, из которых собираются метрики демо
# load_media - ленивая загрузка медиа по slug
# media_for - медиа-запись позиции или пустой словарь
# demo_metrics - метрики для дашборда из готовых отчётов
# catalog_search - поиск и фильтры по каталогу для браузера
# END_MODULE_MAP
"""Медиа, метрики демо и браузер каталога.

Всё читается из уже готовых артефактов: медиа — отдельный файл рядом с каталогом, метрики —
отчёты о качестве. Никаких моделей и сетевых вызовов, поэтому эндпоинты дешёвые и не влияют
на основной путь распознавания.
"""
from __future__ import annotations

import json
from pathlib import Path

from packages.common.logging import log_marker
from packages.common.text_norm import fold, similarity

DEFAULT_MEDIA = "data/catalog/catalog_media.json"
REPORT_FILES = {
    "final": "reports/final.json",
    "real_field": "reports/real_field.json",
    "readiness": "reports/submission_readiness.json",
}
_media_cache: dict[str, dict] = {}
_media_by_slug: dict[str, dict] = {}

__all__ = ["DEFAULT_MEDIA", "REPORT_FILES", "catalog_search", "demo_metrics", "load_media", "media_for"]


def load_media(path: str | None = None) -> dict[str, dict]:
    """Лениво загружает медиа-обогащение как индекс ``slug -> запись``."""
    if _media_by_slug:
        return _media_by_slug
    target = Path(path or DEFAULT_MEDIA)
    if not target.is_file():
        log_marker("M-GATEWAY", "media", "SKIP", reason="file-missing", path=str(target))
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    for record in payload.values():
        slug = str((record or {}).get("slug") or "")
        if slug:
            _media_by_slug[slug] = record
    log_marker("M-GATEWAY", "media", "READY", slugs=len(_media_by_slug))
    return _media_by_slug


def media_for(slug: str) -> dict:
    """Медиа-запись позиции; пустой словарь, если данных нет."""
    return load_media().get(slug) or {}


def demo_metrics(root: Path | None = None) -> dict:
    """Собирает метрики для дашборда из отчётов.

    Отсутствие файла не является ошибкой: поле просто не попадает в ответ.
    """
    base = root or Path.cwd()
    result: dict = {"sources": {}}
    for name, relative in REPORT_FILES.items():
        path = base / relative
        if not path.is_file():
            result["sources"][name] = "missing"
            continue
        try:
            result[name] = json.loads(path.read_text(encoding="utf-8"))
            result["sources"][name] = "ok"
        except json.JSONDecodeError:
            result["sources"][name] = "corrupt"
    final = result.get("final") or {}
    real = result.get("real_field") or {}
    result["dashboard"] = {
        "proxy_acc_at_1": (final.get("quality") or {}).get("acc_at_1"),
        "proxy_acc_at_5": (final.get("quality") or {}).get("acc_at_5"),
        "near_dup_acc_at_1": (final.get("quality") or {}).get("near_duplicate_acc_at_1"),
        "latency_p50": (final.get("latency_ms") or {}).get("p50"),
        "latency_p95": (final.get("latency_ms") or {}).get("p95"),
        "sla_ms": final.get("sla_ms"),
        "real_identity_at_1": real.get("identity_at_1_russian"),
        "real_identity_at_5": real.get("identity_at_5_russian"),
        "real_samples": real.get("samples") or len(real.get("detail") or []) or None,
        "real_latency_p95": (real.get("latency_ms") or {}).get("p95"),
        "gate_min_acc1": final.get("min_acc_at_1"),
    }
    return result


def _facets(items: list[dict], fields: tuple[str, ...] = ("region", "category", "grape", "winery"),
            limit: int = 40) -> dict:
    """Счётчики значений для фильтров UI: сколько записей на регион, категорию, сорт и винодельню."""
    facets: dict[str, list[dict]] = {}
    for field in fields:
        counts: dict[str, int] = {}
        for item in items:
            value = str(item.get(field) or "").strip()
            if value:
                counts[value] = counts.get(value, 0) + 1
        facets[field] = [{"value": value, "count": count}
                         for value, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]]
    return facets


def catalog_search(records: dict, media: dict, *, query: str = "", region: str = "",
                   category: str = "", grape: str = "", winery: str = "",
                   page: int = 1, per_page: int = 20, sort: str = "name") -> dict:
    """Ищет позиции каталога с фильтрами, пагинацией и сортировкой.

    Сравнение фильтров идёт через нормализацию текста, поэтому «Крым» находит записи
    с латинским написанием и наоборот.
    """
    items: list[dict] = []
    query_folded = fold(query)
    for slug, record in records.items():
        entry_media = media.get(slug) or {}
        haystack = fold(" ".join(str(record.get(field) or "") for field in
                                 ("name", "winery", "region", "category", "grape")))
        if query_folded and query_folded not in haystack and similarity(query, haystack) < 0.6:
            continue
        if region and fold(region) not in fold(str(record.get("region") or "")):
            continue
        if category and fold(category) not in fold(str(record.get("category") or "")):
            continue
        if grape and fold(grape) not in fold(str(record.get("grape") or "")):
            continue
        if winery and fold(winery) not in fold(str(record.get("winery") or "")):
            continue
        items.append({
            "slug": slug,
            "name": record.get("name", ""),
            "winery": record.get("winery", ""),
            "region": record.get("region", ""),
            "category": record.get("category", ""),
            "grape": record.get("grape", ""),
            "year": record.get("year"),
            "image_url": entry_media.get("image_url", ""),
            "category_gradient": entry_media.get("category_gradient", ""),
        })
    key = {"name": lambda item: fold(item["name"]),
           "winery": lambda item: fold(item["winery"]),
           "year": lambda item: (item["year"] or 0)}.get(sort, lambda item: fold(item["name"]))
    facets = _facets(items)
    items.sort(key=key)
    total = len(items)
    per_page = max(1, min(int(per_page), 100))
    page = max(1, int(page))
    start = (page - 1) * per_page
    return {"total": total, "page": page, "per_page": per_page,
            "pages": max(1, -(-total // per_page)), "items": items[start:start + per_page],
            "facets": facets}
