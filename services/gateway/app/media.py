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

import hashlib
import json
from pathlib import Path

from rapidfuzz import fuzz

from packages.common.logging import log_marker
from packages.common.official_api import image_proxy_url
from packages.common.text_norm import MAX_FORMS, fold, forms

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


_INDEX_CACHE: dict[int, tuple[int, dict[str, dict]]] = {}
# Формы каталога постоянны, а их расчёт стоит ~3.6 мс на запись. Дисковый кэш убирает эту работу
# из холодного запроса: файл пересобирается только при изменении каталога.
_FORMS_CACHE_PATH = Path("data/catalog/catalog_forms.json")
_FORMS_CACHE_VERSION = 2
# Мелкие наборы (например синтетические в тестах) не пишутся в кэш: иначе они затирали бы формы
# боевого каталога, и следующий старт пересчитывал бы всё заново.
_FORMS_CACHE_MIN_RECORDS = 500


def _prepared_similarity(query_forms: list[str], record_forms: list[str]) -> float:
    """Точная копия similarity, но формы уже предвычислены.

    BUG-001: дорогая часть similarity — построение вариантов написания (translit, алиасы) для
    обеих строк на каждой из ~2000 записей, отсюда 13-16 с на запрос. Здесь варианты каталога
    берутся готовыми, а сравнение (token_set_ratio) остаётся прежним, поэтому результат совпадает
    со старой логикой.
    """
    if not query_forms or not record_forms:
        return 0.0
    best = 0.0
    for left_value in query_forms[:MAX_FORMS]:
        for right_value in record_forms[:MAX_FORMS]:
            best = max(best, fuzz.token_set_ratio(left_value, right_value) / 100.0)
    return round(best, 4)


def _forms_signature(records: dict) -> str:
    """Подпись каталога: кэш форм действителен, пока состав и содержимое записей не изменились.

    C-047: подпись считается по содержимому (отсортированные slug и длины значимых полей), а не по
    порядку записей. Прежняя подпись опиралась на первый и последний slug и не переживала ни смену
    порядка, ни подмену каталога: кэш, записанный на синтетическом наборе, заставлял боевой каталог
    пересчитывать формы заново.
    """
    digest = hashlib.md5()  # noqa: S324 - не криптография, только отпечаток состава каталога
    for slug in sorted(records):
        record = records[slug]
        lengths = "|".join(str(len(str(record.get(field) or "")))
                           for field in ("name", "winery", "region", "category", "grape"))
        digest.update(f"{slug}:{lengths}\n".encode())
    return f"{_FORMS_CACHE_VERSION}:{len(records)}:{digest.hexdigest()[:16]}"


def warm_catalog_index(records: dict) -> int:
    """Готовит индекс форм каталога заранее (C-047).

    Вызывается на старте gateway: первый пользовательский запрос каталога приходит на тёплый индекс
    вместо ожидания построения форм. Возвращает число записей в индексе.
    """
    return len(_catalog_index(records))


def _load_forms_cache(signature: str) -> dict[str, list[str]] | None:
    """Читает кэш форм с диска, если он соответствует текущему каталогу."""
    try:
        if not _FORMS_CACHE_PATH.is_file():
            return None
        payload = json.loads(_FORMS_CACHE_PATH.read_text(encoding="utf-8"))
        if payload.get("signature") != signature:
            return None
        cached = payload.get("forms") or {}
        return cached if cached else None
    except Exception:  # noqa: BLE001 - повреждённый кэш не критичен, формы пересчитаются
        return None


def _save_forms_cache(signature: str, forms_by_slug: dict[str, list[str]]) -> None:
    """Сохраняет кэш форм, чтобы следующий запуск не платил за пересчёт."""
    try:
        _FORMS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _FORMS_CACHE_PATH.write_text(
            json.dumps({"signature": signature, "forms": forms_by_slug}, ensure_ascii=False),
            encoding="utf-8")
    except Exception:  # noqa: BLE001 - кэш не обязателен для работы
        pass


def _catalog_index(records: dict) -> dict[str, dict]:
    """Готовит индекс поиска: свёрнутые поля фильтров и предвычисленные формы для сравнения."""
    key = id(records)
    cached = _INDEX_CACHE.get(key)
    if cached is not None and cached[0] == len(records):
        return cached[1]
    signature = _forms_signature(records)
    forms_by_slug = _load_forms_cache(signature)
    if forms_by_slug is None:
        forms_by_slug = {slug: forms(fold(" ".join(str(record.get(field) or "") for field in
                                                   ("name", "winery", "region", "category", "grape"))))
                         for slug, record in records.items()}
        if len(forms_by_slug) >= _FORMS_CACHE_MIN_RECORDS:
            _save_forms_cache(signature, forms_by_slug)
    index: dict[str, dict] = {}
    for slug, record in records.items():
        index[slug] = {
            "haystack": fold(" ".join(str(record.get(field) or "") for field in
                                      ("name", "winery", "region", "category", "grape"))),
            "forms": forms_by_slug.get(slug) or [],
            "region": fold(str(record.get("region") or "")),
            "category": fold(str(record.get("category") or "")),
            "grape": fold(str(record.get("grape") or "")),
            "winery": fold(str(record.get("winery") or "")),
        }
    _INDEX_CACHE[key] = (len(records), index)
    return index


def catalog_search(records: dict, media: dict, *, query: str = "", region: str = "",
                   category: str = "", grape: str = "", winery: str = "",
                   page: int = 1, per_page: int = 20, sort: str = "name") -> dict:
    """Ищет позиции каталога с фильтрами, пагинацией и сортировкой.

    Сравнение фильтров идёт через нормализацию текста, поэтому «Крым» находит записи
    с латинским написанием и наоборот.
    """
    items: list[dict] = []
    query_folded = fold(query)
    index = _catalog_index(records)
    query_forms = forms(query)
    filters = {"region": fold(region), "category": fold(category),
               "grape": fold(grape), "winery": fold(winery)}
    for slug, record in records.items():
        entry_media = media.get(slug) or {}
        entry = index[slug]
        # C-046/BUG-001: сравнение по предвычисленным формам, без вызовов similarity в цикле.
        if query_folded and query_folded not in entry["haystack"] \
                and _prepared_similarity(query_forms, entry["forms"]) < 0.6:
            continue
        if filters["region"] and filters["region"] not in entry["region"]:
            continue
        if filters["category"] and filters["category"] not in entry["category"]:
            continue
        if filters["grape"] and filters["grape"] not in entry["grape"]:
            continue
        if filters["winery"] and filters["winery"] not in entry["winery"]:
            continue
        items.append({
            "slug": slug,
            "name": record.get("name", ""),
            "winery": record.get("winery", ""),
            "region": record.get("region", ""),
            "category": record.get("category", ""),
            "grape": record.get("grape", ""),
            "year": record.get("year"),
            # C-050: рабочий адрес портала, даже если в производном файле остался сырой путь.
            "image_url": image_proxy_url(entry_media.get("image_url", "")),
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
