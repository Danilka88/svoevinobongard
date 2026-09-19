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
# TEXT_KEYS - поля официального API, участвующие в текстовом сигнале
# text_signal - дополнительный текстовый сигнал (блюда, крепость, температура)
# fusion_mode - режим фьюжна: off, conditional или always
# trigger_max_gap - максимальный разрыв top-1/top-2 для условного триггера
# near_dup_key - ключ near-dup группы: винодельня + название без цифр
# should_fuse - решает, включать ли фьюжн по визуальной выдаче
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
import re
from pathlib import Path

from packages.common.config import settings
from packages.common.logging import log_marker
from packages.common.text_norm import cyr_to_lat, fold

ENRICH_KEYS = ("sweetness", "year", "grape")
# C-017: поля официального API, пригодные для текстового матча
TEXT_KEYS = ("dishes", "alcohol_pct", "temperature", "manufacturer", "region", "grapes")
_COLOR_MAP = {"белое": "Белое", "красное": "Красное", "розовое": "Розовое", "оранжевое": "Оранжевое"}

__all__ = [
    "ENRICH_KEYS",
    "TEXT_KEYS",
    "text_signal",
    "fusion_mode",
    "trigger_max_gap",
    "near_dup_key",
    "should_fuse",
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


def fusion_mode() -> str:
    """Режим фьюжна: off | conditional | always.

    Обратная совместимость: TEXT_FUSION_ENABLED=1 равносилен режиму always.
    """
    if getattr(settings, "text_fusion_enabled", False):
        return "always"
    mode = str(getattr(settings, "text_fusion_mode", "off") or "off").strip().lower()
    return mode if mode in {"off", "conditional", "always"} else "off"


def trigger_max_gap() -> float:
    """Максимальный разрыв скоров top-1 и top-2, при котором триггер ещё срабатывает."""
    return float(getattr(settings, "text_fusion_trigger_max_gap", 0.05))


def near_dup_key(record: dict) -> str:
    """Ключ near-dup группы: винодельня + название без цифр, приведённые к латинице.

    Транслит нужен потому, что каталог смешанный: «Пино Нуар» и `pino-nuar` — одна
    позиция, и без приведения к одному алфавиту они попадали в разные группы.
    """
    winery = cyr_to_lat(fold(str(record.get("winery") or "")))
    name = " ".join(re.sub(r"\d+", " ", cyr_to_lat(fold(str(record.get("name") or "")))).split())[:48]
    return f"{winery}|{name}"


def should_fuse(candidates: list[dict], store, max_gap: float | None = None) -> bool:
    """Срабатывает ли условный триггер: top-1 и top-2 одной near-dup группы и скоры близки.

    Смысл: если визуальная выдача уверенно развела два разных вина, текстовый сигнал
    не нужен и не должен менять порядок. Он нужен только когда впереди стоят
    near-duplicates, которые эмбеддинг не различает.
    """
    scored = [item for item in candidates if item.get("slug")]
    if len(scored) < 2:
        return False
    ordered = sorted(scored, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    first, second = ordered[0], ordered[1]
    gap = float(first.get("score", 0.0)) - float(second.get("score", 0.0))
    limit = trigger_max_gap() if max_gap is None else max_gap
    if gap > limit:
        return False
    left = near_dup_key(dict(store.get(str(first["slug"])) or {}))
    right = near_dup_key(dict(store.get(str(second["slug"])) or {}))
    return bool(left) and left == right


def text_signal(fields: dict, record: dict) -> tuple[float, list[str]]:
    """Дополнительный текстовый сигнал из официальных полей (C-017).

    Включается флагами TEXT_FUSION_USE_DISHES и TEXT_FUSION_USE_NUMBERS и по умолчанию
    выключен: сигнал требует данных со стороны запроса, которых в OCR полевого фото
    почти нет (см. C-016 — год в запросе встречается у 6 из 150 кадров).

    Returns:
        (скор 0..1, список совпавших компонентов)
    """
    matched: list[str] = []
    score = 0.0
    if getattr(settings, "text_fusion_use_dishes", False):
        query_dishes = " ".join(str(part) for part in (fields.get("dishes") or [])).lower()
        catalog_dishes = " ".join(str(part) for part in (record.get("dishes") or [])).lower()
        if query_dishes and catalog_dishes:
            overlap = set(query_dishes.split()) & set(catalog_dishes.split())
            if overlap:
                score += 0.5
                matched.append("dishes")
    if getattr(settings, "text_fusion_use_numbers", False):
        for query_key, record_key in (("alcohol", "alcohol_pct"), ("temperature", "temperature")):
            left, right = str(fields.get(query_key) or ""), str(record.get(record_key) or "")
            if left and right and left[:3] == right[:3]:
                score += 0.25
                matched.append(query_key)
    return min(score, 1.0), matched
