# START_MODULE_CONTRACT
# PURPOSE: Верификация фичи энотуризма: подбор туров, релевантность статей и graceful деградация
# SCOPE: test_load_collections_missing_cache, test_region_of_prefers_official, test_tours_for_match_kinds, test_articles_for_matched_first, test_articles_for_tops_up_with_latest, test_articles_for_no_signal_returns_empty, test_enotourism_for_unknown_slug, test_tour_url_points_to_portal
# DEPENDS: services.gateway.app.enotourism, json, pytest
# LINKS: M-ENOTOURISM, V-M-ENOTOURISM
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# TOURS - тестовый набор туров портала
# ARTICLES - тестовый набор статей портала
# test_load_collections_missing_cache - при отсутствии кэша коллекции пустые, без исключения
# test_region_of_prefers_official - регион берётся из официального обогащения, иначе из каталога
# test_tours_for_match_kinds - туры винодельни идут раньше туров региона и помечаются match
# test_articles_for_matched_first - релевантная статья идёт первой, добор помечается latest
# test_articles_for_tops_up_with_latest - при известном регионе подборка добирается свежими материалами
# test_articles_for_no_signal_returns_empty - без региона, категории и сорта подборка пустая
# test_enotourism_for_unknown_slug - неизвестное вино даёт пустые списки и пояснение
# test_tour_url_points_to_portal - ссылка на тур ведёт на портал
# END_MODULE_MAP
"""V-M-ENOTOURISM: энотуризм и статьи на официальных данных."""
from __future__ import annotations

from services.gateway.app.enotourism import (
    articles_for,
    enotourism_for,
    load_collections,
    region_of,
    tour_url,
    tours_for,
)

__all__ = [
    "ARTICLES", "TOURS",
    "test_articles_for_matched_first", "test_articles_for_no_signal_returns_empty",
    "test_articles_for_tops_up_with_latest", "test_enotourism_for_unknown_slug",
    "test_load_collections_missing_cache", "test_region_of_prefers_official",
    "test_tour_url_points_to_portal", "test_tours_for_match_kinds",
]

TOURS = [
    {"slug": "t-kuban", "title": "Мастер-класс Кубань-Вино", "address": "Краснодарский край, ст. Тамань",
     "price": 2500, "image": {"url": "/uploads/tour.webp"}},
    {"slug": "t-other", "title": "Экскурсия по Крыму", "address": "Республика Крым, Ялта",
     "price": 1500, "image": {"url": "/uploads/crimea.webp"}},
]
ARTICLES = [
    {"slug": "a-kuban", "title": "Кубань: итоги аудита", "category": "Новости", "author": "А. Тест",
     "firstPublishedAt": "2026-08-12T21:00:00.000Z"},
    {"slug": "a-crimea", "title": "Крымские вина вышли на первое место", "category": "Вино", "author": "Б. Тест",
     "firstPublishedAt": "2026-08-20T21:00:00.000Z"},
    {"slug": "a-other", "title": "Как выбрать бокал", "category": "Истории", "author": "В. Тест",
     "firstPublishedAt": "2026-09-01T21:00:00.000Z"},
]


def test_load_collections_missing_cache(tmp_path) -> None:
    """При отсутствии кэша коллекции пустые, без исключения."""
    result = load_collections(tmp_path / "нет-такого")
    assert result == {"tours": [], "regions": [], "articles": []}


def test_region_of_prefers_official() -> None:
    """Регион берётся из официального обогащения, иначе из каталога."""
    assert region_of({"region_name": "Кубань", "region": "Крым"}, "Дон") == "Кубань"
    assert region_of({}, "Дон") == "Дон"
    assert region_of(None) == ""


def test_tours_for_match_kinds() -> None:
    """Туры винодельни идут раньше туров региона и помечаются полем match."""
    items = tours_for(TOURS, region="Кубань", winery="Кубань-Вино")
    assert items and items[0]["slug"] == "t-kuban"
    assert items[0]["match"] == "winery"
    assert items[0]["address"] != ""
    assert items[0]["image_url"].startswith("https://")
    by_region = tours_for(TOURS, region="Крым", winery="Нет такой")
    assert [item["slug"] for item in by_region] == ["t-other"]
    assert by_region[0]["match"] == "region"


def test_articles_for_matched_first() -> None:
    """Релевантная статья идёт первой и помечается matched; добор идёт с пометкой latest."""
    items = articles_for(ARTICLES, region="Крым", category="Вино", limit=3)
    assert items[0]["slug"] == "a-crimea"
    assert items[0]["relevance"] == "matched"
    assert all(item["relevance"] in {"matched", "latest"} for item in items)
    assert sum(1 for item in items if item["relevance"] == "latest") >= 1


def test_articles_for_tops_up_with_latest() -> None:
    """При известном регионе подборка добирается свежими материалами с пометкой latest."""
    items = articles_for(ARTICLES, region="Крым", limit=3)
    assert items[0]["slug"] == "a-crimea"
    assert len(items) == 3
    assert any(item["relevance"] == "latest" for item in items)


def test_articles_for_no_signal_returns_empty() -> None:
    """Без региона, категории и сорта подборка пустая: судить о релевантности нечем."""
    assert articles_for(ARTICLES, region="", category="", grape="") == []


def test_enotourism_for_unknown_slug() -> None:
    """Неизвестное вино даёт пустые списки и пояснение вместо ошибки."""
    payload = enotourism_for("нет-такого-slug")
    assert payload["tours"] == []
    assert payload["articles"] == []
    assert "unknown" in payload["note"]


def test_tour_url_points_to_portal() -> None:
    """Ссылка на тур ведёт на портал, пустой slug даёт корень."""
    assert tour_url("test") == "https://vino-svoe.ru/tours/test"
    assert tour_url("") == "https://vino-svoe.ru"
