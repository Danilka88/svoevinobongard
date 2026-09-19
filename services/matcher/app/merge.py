# START_MODULE_CONTRACT
# PURPOSE: Фьюжн полей VLM с каталогом: текстовый матч и дизамбигуация near-duplicates по году и сладости
# SCOPE: Candidate, rerank_candidates, vintage_hit, sweetness_hit, color_hit
# DEPENDS: rapidfuzz, packages.common.catalog, packages.common.logging
# LINKS: M-MATCHER
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# Candidate - кандидат после фьюжна: slug, итоговый скор, вклад текста и совпадения полей
# rerank_candidates - переранжирование Top-K по полям VLM и метаданным каталога
# vintage_hit - совпадение года этикетки и года каталога
# sweetness_hit - совпадение сладости этикетки и каталога
# color_hit - совпадение цвета этикетки и категории каталога
# END_MODULE_MAP
"""Фьюжн VLM и каталога.

Главный рычаг — дизамбигуация near-duplicates: у одной серии этикетка почти
одинакова, но различаются год и сладость. Эмбеддинг их не различает
(acc@5 0.790 против acc@1 0.685), а поля, прочитанные VLM, различают.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from packages.common.catalog import CatalogStore
from packages.common.logging import log_marker

__all__ = ["Candidate", "rerank_candidates", "vintage_hit", "sweetness_hit", "color_hit"]

_SWEETNESS_TOKENS = (
    "экстра брют", "брют натюр", "брют", "полусладкое", "полусухое", "сладкое", "сухое",
)
_COLOR_TOKENS = {
    "белое": ("белое", "белый", "белая", "white"),
    "красное": ("красное", "красный", "красная", "red"),
    "розовое": ("розовое", "розовый", "розовая", "rose", "rosé"),
    "оранжевое": ("оранжевое", "оранжевый", "orange"),
}


@dataclass
class Candidate:
    """Кандидат после фьюжна."""

    slug: str
    score: float
    base_score: float = 0.0
    text_score: float = 0.0
    vintage_match: bool = False
    sweetness_match: bool = False
    color_match: bool = False
    name: str = ""
    winery: str = ""
    notes: list[str] = field(default_factory=list)


def _normalize(value: str | None) -> str:
    """Нижний регистр без пунктуации и лишних пробелов."""
    return " ".join(re.sub(r"[^0-9a-zа-яё\s]", " ", (value or "").lower()).split())


def _years(value: str | None) -> set[str]:
    """Все четырехзначные годы из строки."""
    return set(re.findall(r"\b(?:19|20)\d{2}\b", value or ""))


def vintage_hit(vlm_vintage: str | None, record: dict) -> bool:
    """Совпадает ли год с этикетки с годом каталога."""
    wanted = _years(str(vlm_vintage or ""))
    if not wanted:
        return False
    catalog_year = record.get("year")
    catalog_years = {str(catalog_year)} if catalog_year else set()
    catalog_years |= _years(str(record.get("name") or ""))
    catalog_years |= _years(str(record.get("slug") or ""))
    return bool(wanted & catalog_years)


def sweetness_hit(vlm_value: str | None, record: dict) -> bool:
    """Совпадает ли сладость с этикетки со сладостью каталога."""
    haystack = _normalize(vlm_value)
    if not haystack:
        return False
    wanted = next((token for token in _SWEETNESS_TOKENS if token in haystack), "")
    if not wanted:
        return False
    catalog = " ".join(_normalize(record.get(field)) for field in ("sweetness", "name", "category"))
    return wanted in catalog


def color_hit(vlm_value: str | None, record: dict) -> bool:
    """Совпадает ли цвет с этикетки с категорией каталога."""
    haystack = _normalize(vlm_value)
    if not haystack:
        return False
    for canonical, tokens in _COLOR_TOKENS.items():
        if not any(token in haystack for token in tokens):
            continue
        catalog = " ".join(_normalize(record.get(field)) for field in ("category", "color", "name"))
        if canonical in catalog:
            return True
    return False


def rerank_candidates(
    candidates: list[dict],
    fields: dict,
    store: CatalogStore,
    *,
    text_weight: float = 0.6,
    vintage_bonus: float = 0.3,
    sweetness_bonus: float = 0.25,
    enriched: dict | None = None,
    fusion: bool | None = None,
) -> list[Candidate]:
    """Переранжирует Top-K по полям VLM и метаданным каталога.

    Args:
        candidates: список словарей с ``slug`` и ``score`` (после retrieval и geo).
        fields: поля этикетки от VLM (winery, wine_name, vintage, wine_type, region, grape).
        store: read-only каталог.
        text_weight: вклад текстового матча.
        vintage_bonus: бонус за совпадение года (дизамбигуация серий).
        sweetness_bonus: бонус за совпадение сладости.
        enriched: overlay ``slug -> запись`` из catalog_enriched.jsonl (C-013). Применяется
            только когда включён TEXT_FUSION_ENABLED; иначе путь работает как раньше.
        fusion: явный override флага (для тестов и валидации); None — брать из окружения.

    Returns:
        Список Candidate, отсортированный по убыванию итогового скора.
    """
    from .textfusion import enriched_record, fusion_mode, fusion_weights, should_fuse, text_signal

    mode = fusion_mode() if fusion is None else fusion
    if mode is True or mode == "always":
        fusion_on = True
    elif mode == "conditional":
        fusion_on = should_fuse(candidates, store)
    else:
        fusion_on = False
    overlay = enriched if (fusion_on and enriched) else None
    weights = fusion_weights() if fusion_on else {
        "text_weight": text_weight, "year_bonus": vintage_bonus, "sweetness_bonus": sweetness_bonus}

    winery_q = _normalize(fields.get("winery"))
    name_q = _normalize(fields.get("wine_name"))
    grape_q = _normalize(fields.get("grape"))
    reranked: list[Candidate] = []

    for item in candidates:
        slug = str(item.get("slug", ""))
        if not slug:
            continue
        record = dict(store.get(slug) or {})
        if overlay is not None:
            record.setdefault("slug", slug)
            record = enriched_record(record, overlay)
        base = float(item.get("score", 0.0))

        winery_score = fuzz.token_set_ratio(winery_q, _normalize(record.get("winery"))) / 100.0 if winery_q else 0.0
        name_score = fuzz.token_set_ratio(name_q, _normalize(record.get("name"))) / 100.0 if name_q else 0.0
        extra_notes: list[str] = []
        if fusion_on:
            grape_score = (fuzz.token_set_ratio(grape_q, _normalize(record.get("grape"))) / 100.0
                           if grape_q else 0.0)
            text_score = round(0.55 * winery_score + 0.35 * name_score + 0.10 * grape_score, 4)
            # C-017: гастросочетания и цифровые поля официального API (флаги по умолчанию выключены)
            extra, extra_notes = text_signal(fields, record)
            if extra:
                text_score = round(min(1.0, text_score + 0.1 * extra), 4)
        else:
            text_score = round(0.6 * winery_score + 0.4 * name_score, 4)

        v_hit = vintage_hit(fields.get("vintage"), record)
        s_hit = sweetness_hit(fields.get("wine_type"), record)
        c_hit = color_hit(fields.get("wine_type"), record)

        final = base + weights["text_weight"] * text_score + (weights["year_bonus"] if v_hit else 0.0) \
            + (weights["sweetness_bonus"] if s_hit else 0.0)
        notes = list(extra_notes)
        if v_hit:
            notes.append("vintage")
        if s_hit:
            notes.append("sweetness")
        if c_hit:
            notes.append("color")
        reranked.append(Candidate(
            slug=slug, score=round(final, 6), base_score=round(base, 6),
            text_score=text_score, vintage_match=v_hit, sweetness_match=s_hit,
            color_match=c_hit, name=record.get("name", ""), winery=record.get("winery", ""),
            notes=notes,
        ))

    reranked.sort(key=lambda candidate: candidate.score, reverse=True)
    if reranked:
        log_marker("M-MATCHER", "rerank", "DONE", candidates=len(reranked),
                   top1=reranked[0].slug, vintage=str(reranked[0].vintage_match),
                   sweetness=str(reranked[0].sweetness_match))
    return reranked
