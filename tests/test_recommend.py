# START_MODULE_CONTRACT
# PURPOSE: Верификация retention-логики: аналоги из других виноделен, гастрономические пары, разбор ответа сомелье
# SCOPE: test_find_similar_excludes_same_winery, test_find_similar_scores_grape_higher, test_find_similar_unknown_slug, test_pairing_for_colors, test_catalog_recommendations_filters, test_parse_sommelier_reply_filters_unknown_slugs, test_parse_sommelier_reply_rejects_garbage
# DEPENDS: packages.common.catalog, services.gateway.app.recommend
# LINKS: M-GATEWAY
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# test_find_similar_excludes_same_winery - аналоги не содержат вин той же винодельни
# test_find_similar_scores_grape_higher - совпадение сорта дает больший скор, чем только регион
# test_find_similar_unknown_slug - неизвестный slug дает пустой список
# test_pairing_for_colors - пары подбираются по цвету и сладости
# test_catalog_recommendations_filters - подбор без LLM фильтрует по стилю и блюду
# test_parse_sommelier_reply_filters_unknown_slugs - ответ модели фильтруется по существующим slug
# test_parse_sommelier_reply_rejects_garbage - мусор вместо JSON не дает рекомендаций
# END_MODULE_MAP
"""V-M-GATEWAY: аналоги и цифровой сомелье."""
from __future__ import annotations

from packages.common.catalog import CatalogStore
from services.gateway.app.recommend import (
    catalog_recommendations,
    find_similar,
    pairing_for,
    parse_sommelier_reply,
)

__all__ = [
    "test_find_similar_excludes_same_winery",
    "test_find_similar_scores_grape_higher",
    "test_find_similar_unknown_slug",
    "test_pairing_for_colors",
    "test_catalog_recommendations_filters",
    "test_parse_sommelier_reply_filters_unknown_slugs",
    "test_parse_sommelier_reply_rejects_garbage",
]

_RECORDS = {
    "base": {"slug": "base", "name": "Кокур Сухое", "winery": "Захарьин", "region": "Крым",
             "category": "Белое", "grape": "Кокур", "sweetness": "сухое", "description": ""},
    "same-grape": {"slug": "same-grape", "name": "Кокур Резерв", "winery": "Другая", "region": "Кубань",
                   "category": "Белое", "grape": "Кокур", "sweetness": "сухое", "description": ""},
    "same-region": {"slug": "same-region", "name": "Рислинг", "winery": "Третья", "region": "Крым",
                    "category": "Белое", "grape": "Рислинг", "sweetness": "сухое", "description": ""},
    "same-winery": {"slug": "same-winery", "name": "Кокур Полусладкое", "winery": "Захарьин",
                    "region": "Крым", "category": "Белое", "grape": "Кокур",
                    "sweetness": "полусладкое", "description": ""},
    "red": {"slug": "red", "name": "Каберне", "winery": "Четвёртая", "region": "Кубань",
            "category": "Красное", "grape": "Каберне", "sweetness": "сухое", "description": ""},
}


def _store() -> CatalogStore:
    return CatalogStore(_RECORDS)


def test_find_similar_excludes_same_winery() -> None:
    items = find_similar(_store(), "base")
    slugs = {item["slug"] for item in items}
    assert "same-winery" not in slugs
    assert "base" not in slugs


def test_find_similar_scores_grape_higher() -> None:
    items = find_similar(_store(), "base")
    assert items[0]["slug"] == "same-grape"
    assert "тот же сорт" in items[0]["reason"]
    by_slug = {item["slug"]: item for item in items}
    assert by_slug["same-grape"]["score"] > by_slug["same-region"]["score"]


def test_find_similar_unknown_slug() -> None:
    assert find_similar(_store(), "missing") == []


def test_pairing_for_colors() -> None:
    assert "Мясо на гриле" in pairing_for(_RECORDS["red"])
    assert any("Рыба" in item or "морепродукты" in item.lower() for item in pairing_for(_RECORDS["base"]))
    assert any("Икра" in item for item in pairing_for({"category": "Игристое", "name": "Брют"}))


def test_catalog_recommendations_filters() -> None:
    items = catalog_recommendations(_store(), "base", {"dish": "рыба", "style": "сухое"})
    slugs = {item["slug"] for item in items}
    assert "red" not in slugs
    assert items and all(item["why"] for item in items)


def test_parse_sommelier_reply_filters_unknown_slugs() -> None:
    reply = ('{"recommendations":[{"slug":"same-grape","why":"тот же сорт"},'
             '{"slug":"hallucinated","why":"нет в каталоге"}],"pairing":["Рыба","Сыры"]}')
    parsed = parse_sommelier_reply(reply, {"same-grape"})
    assert [item["slug"] for item in parsed["recommendations"]] == ["same-grape"]
    assert parsed["pairing"] == ["Рыба", "Сыры"]


def test_parse_sommelier_reply_rejects_garbage() -> None:
    assert parse_sommelier_reply("совсем не json", {"a"}) == {}
    assert parse_sommelier_reply('{"recommendations":[],"pairing":[]}', {"a"}) == {}
