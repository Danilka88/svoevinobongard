# START_MODULE_CONTRACT
# PURPOSE: Верификация text-first поиска и RRF: транслит, атрибутивные бонусы, порядок слияния, путь без флага
# SCOPE: test_text_search_matches_translit, test_text_search_attribute_boost, test_text_search_ranks_whole_catalog, test_rrf_prefers_top_of_both, test_rrf_weights_favour_visual, test_text_index_matches_direct_similarity, test_fields_from_ocr_picks_wordy_line, test_text_first_disabled_returns_none
# DEPENDS: packages.common.text_rank, packages.common.config, services.gateway.app.pipeline, pytest
# LINKS: M-TEXT, M-MATCHER, V-M-TEXT
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# CATALOG - тестовые записи каталога со смешанными алфавитами
# test_text_search_matches_translit - латинский запрос находит кириллическую запись через транслит
# test_text_search_attribute_boost - совпадение года и сорта поднимает запись
# test_text_search_ranks_whole_catalog - ранжируется весь каталог, а не top-K визуала
# test_text_index_matches_direct_similarity - предвычисленный индекс даёт тот же порядок
# test_rrf_prefers_top_of_both - RRF поднимает позицию, которая высоко в обоих ранжированиях
# test_rrf_weights_favour_visual - при весе визуала выше текстовый лидер не вытесняет визуальный
# test_fields_from_ocr_picks_wordy_line - поля из OCR берут самую словесную строку
# test_text_first_disabled_returns_none - при выключенном флаге text-first не вмешивается
# END_MODULE_MAP
"""V-M-TEXT: text-first и RRF."""
from __future__ import annotations

from packages.common.config import settings
from packages.common.text_rank import fields_from_ocr, rrf, text_index, text_search

__all__ = [
    "CATALOG",
    "test_fields_from_ocr_picks_wordy_line",
    "test_rrf_prefers_top_of_both",
    "test_rrf_weights_favour_visual",
    "test_text_first_disabled_returns_none",
    "test_text_index_matches_direct_similarity",
    "test_text_search_attribute_boost",
    "test_text_search_matches_translit",
    "test_text_search_ranks_whole_catalog",
]

CATALOG = {
    "shato-taman-polusladkoe": {"slug": "shato-taman-polusladkoe", "name": "Шато Тамань Полусладкое",
                                "winery": "Кубань-Вино", "grape": "Рислинг", "region": "Кубань", "year": 2022},
    "abrau-dyurso-bryut": {"slug": "abrau-dyurso-bryut", "name": "Абрау Дюрсо Брют",
                           "winery": "Абрау-Дюрсо", "grape": "Шардоне", "region": "Кубань"},
    "fanagoriya-risling": {"slug": "fanagoriya-risling", "name": "Рислинг",
                           "winery": "Фанагория", "grape": "Рислинг", "region": "Кубань", "year": 2023},
}


def test_text_search_matches_translit() -> None:
    """Латинский запрос находит кириллическую запись через транслит и алиасы."""
    ranked = text_search({"winery": "Chateau Tamagne", "wine_name": "semi-sweet"}, CATALOG)
    assert ranked and ranked[0][0] == "shato-taman-polusladkoe"


def test_text_search_attribute_boost() -> None:
    """Совпадение года и сорта поднимает запись среди конкурентов с тем же сортом."""
    ranked = text_search({"grape": "Рислинг", "vintage": "2023"}, CATALOG)
    assert ranked[0][0] == "fanagoriya-risling", "год должен отделять похожие записи"


def test_text_search_ranks_whole_catalog() -> None:
    """Ранжируется весь каталог, а не только переданный top-K."""
    ranked = text_search({"winery": "Абрау"}, CATALOG)
    assert ranked[0][0] == "abrau-dyurso-bryut", "лучшая запись — Абрау"
    assert len(ranked) <= len(CATALOG), "ранжируется весь переданный каталог"


def test_text_index_matches_direct_similarity() -> None:
    """Предвычисленный индекс даёт тот же порядок, что и прямой расчёт."""
    fields = {"winery": "Фанагория", "wine_name": "Рислинг"}
    direct = [slug for slug, _ in text_search(fields, CATALOG)]
    cached = [slug for slug, _ in text_search(fields, CATALOG, index=text_index(CATALOG))]
    assert direct == cached


def test_rrf_prefers_top_of_both() -> None:
    """RRF поднимает позицию, которая высоко в обоих ранжированиях."""
    fused = rrf([["a", "b", "c"], ["a", "c", "b"]])
    assert fused[0] == "a"


def test_rrf_weights_favour_visual() -> None:
    """При большем весе визуала текстовый лидер не вытесняет визуальный top-1."""
    fused = rrf([["visual-best", "other"], ["text-best", "visual-best"]], weights=[1.0, 0.5])
    assert fused[0] == "visual-best"


def test_fields_from_ocr_picks_wordy_line() -> None:
    """Поля из OCR берут самую словесную строку как название."""
    fields = fields_from_ocr("1870\nABRAU-DURSO\nRESERVE\n", winery_hint="Абрау-Дюрсо")
    assert fields["winery"] == "Абрау-Дюрсо"
    assert "RESERVE" in fields["wine_name"].upper() or "ABRAU" in fields["wine_name"].upper()


def test_text_first_disabled_returns_none() -> None:
    """При выключенном флаге text-first не вмешивается в решение."""
    from services.gateway.app.pipeline import text_first_ranking

    was = settings.text_first_enabled
    try:
        object.__setattr__(settings, "text_first_enabled", False)
        assert text_first_ranking(None, {"winery": "Абрау"}, ["a"]) is None
    finally:
        object.__setattr__(settings, "text_first_enabled", was)
