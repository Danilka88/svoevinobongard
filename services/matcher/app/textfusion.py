# START_MODULE_CONTRACT
# PURPOSE: Обогащённые метаданные каталога и текстовый фьюжн для гибридного retrieval (C-013)
# SCOPE: text_fusion_enabled, fusion_weights, color_normalize, load_enriched_catalog, enriched_record
# DEPENDS: json, pathlib, packages.common.config, packages.common.logging
# LINKS: M-MATCHER, M-TEXT, M-CATALOG
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# ENRICH_KEYS - дозаполняемые поля обогащения
# text_fusion_enabled - включён ли текстовый фьюжн флагом окружения
# fusion_weights - веса текстового сигнала и бонусов за дозаполненные поля
# color_normalize - нормализация свободного color/category к каноническому цвету
# load_enriched_catalog - читает catalog_enriched.jsonl как overlay по slug
# enriched_record - накладывает overlay на запись каталога перед матчингом
# END_MODULE_MAP
"""Обогащённые метаданные и текстовый фьюжн.

Текстовый сигнал включается флагом ``TEXT_FUSION_ENABLED`` (по умолчанию выключен):
пока прирост не подтверждён на валидации, онлайн-путь обязан работать как раньше.

Overlay из ``catalog_enriched.jsonl`` даёт матчеру то, чего нет в исходном каталоге:
дозаполненные ``sweetness``/``year``/``grape`` и нормализованный цвет. Свободный
``color`` как признак не используется — только через нормализацию в ``category``.
"""
from __future__ import annotations

import json
from pathlib import Path

from packages.common.config import settings
from packages.common.logging import log_marker

ENRICH_KEYS = ("sweetness", "year", "grape")
_COLOR_MAP = {"белое": "Белое", "красное": "Красное", "розовое": "Розовое", "оранжевое": "Оранжевое"}

__all__ = [
    "ENRICH_KEYS",
    "text_fusion_enabled",
    "fusion_weights",
    "color_normalize",
    "load_enriched_catalog",
    "enriched_record",
]


def text_fusion_enabled() -> bool:
    """Включён ли текстовый фьюжн (флаг TEXT_FUSION_ENABLED)."""
    return bool(getattr(settings, "text_fusion_enabled", False))


def fusion_weights() -> dict[str, float]:
    """Веса текстового сигнала: матч, бонус за год, бонус за сладость."""
    return {
        "text_weight": float(getattr(settings, "text_fusion_weight", 0.6)),
        "year_bonus": float(getattr(settings, "text_fusion_year_bonus", 0.35)),
        "sweetness_bonus": float(getattr(settings, "text_fusion_sweetness_bonus", 0.25)),
    }


def color_normalize(record: dict) -> str:
    """Нормализует свободный color/category к каноническому цвету.

    Свободный ``color`` вроде «красное сухое вино» не используется как признак напрямую:
    из него извлекается только каноническая категория.
    """
    if record.get("category_normalized"):
        return str(record["category_normalized"])
    haystack = f"{record.get('color', '')} {record.get('category', '')}".lower()
    for key, canonical in _COLOR_MAP.items():
        if key in haystack:
            return canonical
    return str(record.get("category", "") or "")


def load_enriched_catalog(path: str | Path | None = None) -> dict[str, dict]:
    """Читает обогащённый каталог как overlay ``slug -> поля``.

    Отсутствие файла — не ошибка: обогащение опционально, матчер работает и без него.
    """
    target = Path(path or getattr(settings, "catalog_enriched_jsonl", "data/catalog/catalog_enriched.jsonl"))
    if not target.is_file():
        log_marker("M-TEXT", "load_enriched", "SKIP", reason="file-missing", path=str(target))
        return {}
    overlay: dict[str, dict] = {}
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            slug = str(record.get("slug") or "")
            if slug:
                overlay[slug] = record
    log_marker("M-TEXT", "load_enriched", "DONE", slugs=len(overlay))
    return overlay


def enriched_record(record: dict, overlay: dict[str, dict]) -> dict:
    """Накладывает overlay на запись каталога перед матчингом.

    Дозаполняются только отсутствующие поля: если каталог уже знает значение,
    обогащение его не перетирает.
    """
    datum = dict(record)
    entry = overlay.get(str(datum.get("slug") or datum.get("id") or ""))
    if entry:
        for key, value in (entry.get("enriched") or {}).items():
            if key in ENRICH_KEYS and value and not datum.get(key):
                datum[key] = value
        datum["category"] = color_normalize(entry) or datum.get("category", "")
    else:
        datum["category"] = color_normalize(datum)
    return datum
