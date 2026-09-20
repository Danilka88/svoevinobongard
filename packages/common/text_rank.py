# START_MODULE_CONTRACT
# PURPOSE: Текстовый поиск по всему каталогу и слияние ранжирований RRF (текст как отдельный источник кандидатов)
# SCOPE: TEXT_WEIGHTS, RRF_K, text_search, rrf, fields_from_ocr
# DEPENDS: packages.common.catalog, packages.common.text_norm
# LINKS: M-TEXT, M-MATCHER, M-RETRIEVAL
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# TEXT_WEIGHTS - веса текстовых полей и атрибутивных бонусов
# RRF_K - константа сглаживания в Reciprocal Rank Fusion
# text_index - предвычисленные формы полей каталога (одноразовая подготовка)
# text_search - ранжирование всего каталога по текстовым полям запроса
# rrf - слияние нескольких ранжирований по reciprocal rank fusion
# fields_from_ocr - поля запроса из текста OCR (без VLM, дешёвый онлайн-путь)
# END_MODULE_MAP
"""Текстовый поиск по всему каталогу и RRF.

Замер C-026 показал, что у 32% реальных фото правильного вина нет даже в визуальном
top-100 из 2103 позиций. Значит реранк поверх визуального top-K принципиально не может
их спасти: текст должен быть **отдельным источником кандидатов по всему каталогу**.

Потолок текстового сигнала измерен отдельно (C-027): 28.6% строго, поэтому текст
подключается как второй источник в слиянии RRF, а не как замена визуального поиска.
Нормализация идёт через ``text_norm``: транслит лат↔кир, OCR-двойники и алиасы сортов,
типов и брендов.
"""
from __future__ import annotations

from packages.common.text_norm import forms, similarity

TEXT_WEIGHTS = {
    "winery": 0.34,
    "name": 0.40,
    "grape": 0.14,
    "region": 0.06,
    "vintage": 0.18,   # атрибутивный бонус за совпадение года
    "sweetness": 0.10,  # атрибутивный бонус за совпадение типа
}
RRF_K = 60

__all__ = ["RRF_K", "TEXT_WEIGHTS", "fields_from_ocr", "rrf", "text_index", "text_search"]


def text_index(catalog: dict, fields: tuple[str, ...] = ("winery", "name", "grape", "region")) -> dict:
    """Предвычисляет варианты написания полей каталога: без этого поиск стоил ~32 с на запрос.

    Возвращает ``slug -> {поле: список форм}``; формы считаются один раз на каталог
    и переиспользуются всеми запросами.
    """
    index: dict[str, dict[str, list[str]]] = {}
    for slug, record in catalog.items():
        index[slug] = {field: forms(record.get(field)) for field in fields}
    return index


def text_search(fields: dict, catalog: dict, *, weights: dict | None = None,
                limit: int | None = None, index: dict | None = None,
                max_forms: int = 3) -> list[tuple[str, float]]:
    """Ранжирует весь каталог по текстовым полям запроса.

    Args:
        fields: нормализуемые поля запроса (winery, wine_name, grape, vintage,
            sweetness/wine_type, region). Пустые поля не дают вклада.
        catalog: записи каталога ``slug -> запись``.
        weights: переопределение весов (по умолчанию ``TEXT_WEIGHTS``).
        limit: ограничить длину ответа.

    Returns:
        Список ``(slug, score)`` по убыванию скора; позиции с нулевым скором отброшены.
    """
    cfg = {**TEXT_WEIGHTS, **(weights or {})}
    winery_query = fields.get("winery") or ""
    name_query = fields.get("wine_name") or fields.get("name") or ""
    grape_query = fields.get("grape") or ""
    region_query = fields.get("region") or ""
    vintage_query = str(fields.get("vintage") or "")
    sweetness_query = fields.get("sweetness") or fields.get("wine_type") or ""

    from rapidfuzz import fuzz  # noqa: PLC0415

    query_forms = {"winery": forms(winery_query)[:max_forms], "name": forms(name_query)[:max_forms],
                   "grape": forms(grape_query)[:max_forms], "region": forms(region_query)[:max_forms]}
    precomputed = index if index is not None else text_index(catalog)

    def best(query_values: list[str], catalog_values: list[str]) -> float:
        """Максимум похожести по предвычисленным формам обеих сторон."""
        best_score = 0.0
        for left in query_values:
            for right in catalog_values[:max_forms]:
                best_score = max(best_score, fuzz.token_set_ratio(left, right) / 100.0)
        return best_score

    scored: list[tuple[str, float]] = []
    for slug, record in catalog.items():
        cached = precomputed.get(slug) or {}
        score = 0.0
        if winery_query:
            score += cfg["winery"] * best(query_forms["winery"], cached.get("winery") or [])
        if name_query:
            score += cfg["name"] * best(query_forms["name"], cached.get("name") or [])
        if grape_query:
            score += cfg["grape"] * best(query_forms["grape"], cached.get("grape") or [])
        if region_query:
            score += cfg["region"] * best(query_forms["region"], cached.get("region") or [])
        if vintage_query and vintage_query == str(record.get("year") or ""):
            score += cfg["vintage"]
        if sweetness_query and similarity(sweetness_query, record.get("sweetness")) >= 0.95:
            score += cfg["sweetness"]
        if score > 0.0:
            scored.append((slug, round(score, 6)))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:limit] if limit else scored


def rrf(rankings: list[list[str]], k: int = RRF_K, weights: list[float] | None = None) -> list[str]:
    """Сливает ранжирования по Reciprocal Rank Fusion.

    Каждое ранжирование даёт вклад ``w / (k + rank)``; позиции, которых нет в списке,
    вклада не дают. Ручное сложение скоров не используется намеренно: у визуального и
    текстового ранжирований разные шкалы, RRF работает только с порядком.
    """
    totals: dict[str, float] = {}
    factors = weights or [1.0] * len(rankings)
    for ranking, weight in zip(rankings, factors):
        for rank, slug in enumerate(ranking, 1):
            if not slug:
                continue
            totals[slug] = totals.get(slug, 0.0) + float(weight) / (k + rank)
    return [slug for slug, _ in sorted(totals.items(), key=lambda item: item[1], reverse=True)]


def fields_from_ocr(text: str, winery_hint: str = "", vintage: str = "",
                    sweetness: str = "") -> dict:
    """Собирает поля запроса из текста OCR, без обращения к VLM."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    name = ""
    if lines:
        # самая длинная буквенная строка обычно и есть название
        name = max(lines, key=lambda line: sum(1 for char in line if char.isalpha()))
    return {"winery": winery_hint or "", "wine_name": name, "vintage": vintage or "",
            "sweetness": sweetness or "", "grape": "", "region": ""}
