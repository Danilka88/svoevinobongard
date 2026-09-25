# START_MODULE_CONTRACT
# PURPOSE: Тесты C-073 улучшений наблюдателя: омоглифы, инвариант пустого наблюдения, структурированный разбор
# SCOPE: CATALOG, test_homoglyphs_greek_to_cyrillic, test_mixed_latin_homoglyphs_normalised,
#        test_latin_brands_unchanged, test_placeholder_values_are_not_observation,
#        test_empty_observation_yields_no_candidates, test_vintage_alone_is_not_informative,
#        test_free_text_still_informative, test_structured_json_observation,
#        test_structured_dirty_json_repaired, test_structured_falls_back_to_key_value
# DEPENDS: packages.common.observer, packages.common.text_norm
# LINKS: M-COMMON, M-VLM
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# CATALOG - синтетический каталог для проверки инварианта пустого наблюдения
# test_homoglyphs_greek_to_cyrillic - греческие двойники переводятся в кириллицу
# test_mixed_latin_homoglyphs_normalised - смешанные латино-кириллические токены приводятся к кириллице
# test_latin_brands_unchanged - чистые латинские бренды не портятся
# test_placeholder_values_are_not_observation - заглушки не становятся полями наблюдения
# test_empty_observation_yields_no_candidates - пустое наблюдение не даёт случайного slug
# test_vintage_alone_is_not_informative - один год без названия не считается наблюдением
# test_free_text_still_informative - информативный свободный текст по-прежнему ищется
# test_structured_json_observation - JSON по схеме разбирается структурированным парсером
# test_structured_dirty_json_repaired - «грязный» JSON (фенсы, хвост) восстанавливается
# test_structured_falls_back_to_key_value - не-JSON откатывается на терпимый «ключ: значение»
# END_MODULE_MAP
"""Тесты улучшений наблюдателя C-073 (без сети и моделей)."""
from __future__ import annotations

from packages.common.observer import (is_informative, observation_fields, observer_candidates,
                                      parse_observation, parse_observation_structured)
from packages.common.text_norm import homoglyphs_to_cyrillic

CATALOG = {
    "skalistyy-bereg-shepot-czvetov": {"name": "Шёпот цветов", "winery": "Скалистый берег",
                                       "grape": "Рислинг", "year": 2024, "sweetness": "брют"},
    "massandra-muskatel": {"name": "Мускатель белый", "winery": "Массандра", "grape": "Мускатель",
                           "year": 2023, "sweetness": "сладкое"},
}

__all__ = ["CATALOG", "test_empty_observation_yields_no_candidates", "test_free_text_still_informative",
           "test_homoglyphs_greek_to_cyrillic", "test_latin_brands_unchanged",
           "test_mixed_latin_homoglyphs_normalised", "test_placeholder_values_are_not_observation",
           "test_structured_dirty_json_repaired", "test_structured_falls_back_to_key_value",
           "test_structured_json_observation", "test_vintage_alone_is_not_informative"]


def test_homoglyphs_greek_to_cyrillic() -> None:
    """Греческие двойники кириллицы переводятся, как и раньше (совместимость C-064)."""
    assert homoglyphs_to_cyrillic("ΛΣΚΙΦ") == "ЛСКИФ"
    assert homoglyphs_to_cyrillic("ΜΑΣΣΑΝΔΡΑ") == "МАССАНДРА"
    assert homoglyphs_to_cyrillic("ΚΡΑСНАЯ СТРЕЛА") == "КРАСНАЯ СТРЕЛА"


def test_mixed_latin_homoglyphs_normalised() -> None:
    """Смешанный токен с латинскими двойниками приводится к кириллице."""
    assert homoglyphs_to_cyrillic("СKАЛИСТЫЙ") == "СКАЛИСТЫЙ"
    assert homoglyphs_to_cyrillic("МACCAНДРА") == "МАССАНДРА"
    assert homoglyphs_to_cyrillic("ШАРДОНЕ") == "ШАРДОНЕ"


def test_latin_brands_unchanged() -> None:
    """Чистые латинские бренды не портятся: иначе мы сломали бы корректные наблюдения."""
    for case in ("FANAGORIA", "CHATEAU PINOT", "UVA VALLIS", "MASANDRA", "BOSCA"):
        assert homoglyphs_to_cyrillic(case) == case


def test_placeholder_values_are_not_observation() -> None:
    """Заглушки («неразборчиво», «(нет)») не становятся полями наблюдения."""
    observation = parse_observation("winery: неразборчиво\nwine_name: (нет)\nraw_text: неразборчиво")
    assert observation == {} or not observation_fields(observation)
    assert not is_informative(observation)


def test_empty_observation_yields_no_candidates() -> None:
    """Инвариант: пустое/неинформативное наблюдение не должно выдавать случайное вино."""
    for text in ("", None, "winery: неразборчиво", "raw_text: не указано", "winery: (нет)"):
        observation = parse_observation(text)
        assert observer_candidates(observation, CATALOG, limit=5) == []


def test_vintage_alone_is_not_informative() -> None:
    """Один год (и другие атрибуты) без названия вино не идентифицирует."""
    observation = parse_observation("vintage: 2024\nsweetness: брют")
    assert not is_informative(observation)


def test_free_text_still_informative() -> None:
    """Информативный свободный текст по-прежнему попадает в поиск."""
    observation = parse_observation("winery:\nwine_name:\nraw_text: СКАЛИСТЫЙ БЕРЕГ ШЁПОТ ЦВЕТОВ")
    assert is_informative(observation)
    hits = observer_candidates(observation, CATALOG, limit=1)
    assert hits and hits[0][0] == "skalistyy-bereg-shepot-czvetov"


def test_structured_json_observation() -> None:
    """JSON по схеме разбирается структурированным парсером без потери значений."""
    text = ('```json\n{"winery": "Массандра", "wine_name": "Мускатель белый", "sweetness": "dry", '
            '"vintage": 2023}\n```')
    observation = parse_observation_structured(text)
    assert observation["winery"] == "Массандра"
    assert observation["wine_name"] == "Мускатель белый"
    assert observation["vintage"] == "2023"


def test_structured_dirty_json_repaired() -> None:
    """Хвостовая запятая и текст вокруг JSON не мешают разбору."""
    text = 'Вот результат: {"winery": "Массандра", "wine_name": "Мускатель",}'
    observation = parse_observation_structured(text)
    assert observation["winery"] == "Массандра"


def test_structured_falls_back_to_key_value() -> None:
    """Не-JSON откатывается на терпимый «ключ: значение» без изменения поведения."""
    observation = parse_observation_structured("winery: Скалистый берег\nwine_name: Шёпот цветов")
    assert observation["winery"] == "Скалистый берег"
    assert observation["wine_name"] == "Шёпот цветов"
