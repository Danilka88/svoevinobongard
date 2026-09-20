# START_MODULE_CONTRACT
# PURPOSE: Верификация восьми фиксов второго аудита (C-030): сортировка, скор мультивью, уверенность,
#          эндпоинты, TTL кэша, лимит вариантов, сетки тайлов, ложные регионы
# SCOPE: test_articles_sorted_fresh_first, test_latest_fill_prefers_fresh, test_region_requires_word_boundary, test_region_exact_name_matches, test_index_score_takes_max_over_views, test_health_lists_all_endpoints, test_cached_json_max_age, test_forms_limit_is_configurable, test_tile_grid_arbitrary
# DEPENDS: services.gateway.app.enotourism, services.retrieval.app.index, packages.common.official_api, packages.common.text_norm, packages.common.preprocess, pytest
# LINKS: M-ENOTOURISM, M-RETRIEVAL, M-COMMON, M-GATEWAY, V-M-ENOTOURISM
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# ARTICLES - тестовый набор статей портала
# REGIONS - тестовый набор регионов
# test_articles_sorted_fresh_first - при равном score выше более свежая статья
# test_latest_fill_prefers_fresh - добор latest идёт от свежих материалов
# test_region_requires_word_boundary - регион не ловится подстрокой («Крым» в «Крымский район»)
# test_region_exact_name_matches - точное название региона находится
# test_index_score_takes_max_over_views - скор позиции это максимум по всем видам мультивью-индекса
# test_health_lists_all_endpoints - /v1/health перечисляет все публичные эндпоинты
# test_cached_json_max_age - устаревший кэш перекачивается, max_age_s=None сохраняет прежнее поведение
# test_forms_limit_is_configurable - лимит вариантов написания не режет список жёстко
# test_tile_grid_arbitrary - сетка тайлов поддерживает произвольное число тайлов
# END_MODULE_MAP
"""V-M-ENOTOURISM / V-M-RETRIEVAL / V-M-COMMON: фиксы C-030."""
from __future__ import annotations

import json

import numpy as np

from packages.common import official_api, preprocess, text_norm
from services.gateway.app.enotourism import _region_fact, articles_for
from services.retrieval.app.index import VectorIndex

__all__ = [
    "test_articles_sorted_fresh_first",
    "test_articles_sorted_fresh_first",
    "test_cached_json_max_age",
    "test_forms_limit_is_configurable",
    "test_health_lists_all_endpoints",
    "test_index_score_takes_max_over_views",
    "test_latest_fill_prefers_fresh",
    "test_region_exact_name_matches",
    "test_region_requires_word_boundary",
    "test_tile_grid_arbitrary",
]

ARTICLES = [
    {"slug": "old-same", "title": "Крым: тот же балл, старее", "firstPublishedAt": "2026-01-01T00:00:00.000Z"},
    {"slug": "new-same", "title": "Крым: тот же балл, свежее", "firstPublishedAt": "2026-09-01T00:00:00.000Z"},
    {"slug": "unrelated", "title": "Не про регион", "firstPublishedAt": "2026-12-01T00:00:00.000Z"},
]
REGIONS = [
    {"name": "Крым", "fact": "факт Крыма", "eventsCount": 2},
    {"name": "Крымский район", "fact": "факт района", "eventsCount": 1},
]


def test_articles_sorted_fresh_first() -> None:
    """При равном score выше более свежая статья."""
    picked = articles_for(ARTICLES, region="Крым", limit=1)
    assert [item["slug"] for item in picked] == ["new-same"]


def test_latest_fill_prefers_fresh() -> None:
    """Добор latest идёт от свежих материалов, а не от старых."""
    picked = articles_for(ARTICLES, region="Крым", limit=3)
    slugs = [item["slug"] for item in picked]
    assert slugs[0] == "new-same", "свежая статья с тем же score идёт первой"
    assert slugs.index("old-same") < slugs.index("unrelated"), "релевантные материалы идут выше добора"


def test_region_requires_word_boundary() -> None:
    """Подстрока не ловится: запрос «район» не должен подставлять факт региона «Крымский район» по «Крым»."""
    fact = _region_fact(REGIONS, "Крым")
    assert fact["fact"] == "факт Крыма", "«Крым» не должен матчиться на «Крымский район» подстрокой"


def test_region_exact_name_matches() -> None:
    """Точное название региона находится."""
    assert _region_fact(REGIONS, "Крымский район")["fact"] == "факт района"
    assert _region_fact(REGIONS, "НетТакогоРегиона")["fact"] == ""


def test_index_score_takes_max_over_views() -> None:
    """Скор позиции — максимум по всем видам, а не первый вид."""
    index = VectorIndex(backend="numpy")
    index.build(["a", "b", "a"], np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=np.float32))
    best = index.score("a", np.array([0.0, 1.0], dtype=np.float32))
    assert best == 1.0, "должен выбраться лучший вид позиции a"
    assert index.score("нет-такого", np.array([1.0, 0.0], dtype=np.float32)) == 0.0


def test_cached_json_max_age(tmp_path) -> None:
    """Устаревший кэш перекачивается; max_age_s=None оставляет прежнее поведение."""
    target = tmp_path / "cache.json"
    target.write_text(json.dumps({"value": "из-кэша"}), encoding="utf-8")
    fresh = official_api.cached_json("http://127.0.0.1:1/none", target, retries=1, pause=0.0)
    assert fresh == {"value": "из-кэша"}, "без TTL кэш читается как раньше"
    stale = official_api.cached_json("http://127.0.0.1:1/none", target, retries=1, pause=0.0, max_age_s=0.0)
    assert stale is None, "при недоступной сети и устаревшем кэше возвращается None"


def test_forms_limit_is_configurable() -> None:
    """Лимит вариантов написания не жёстко 5 и настраивается модулем."""
    assert text_norm.MAX_FORMS >= 8
    many = text_norm.forms("Шато Тамань Резерв Кюве Миллезимато Блан де Блан")
    assert len(many) >= 2 and len(many) <= text_norm.MAX_FORMS


def test_health_lists_all_endpoints() -> None:
    """Список эндпоинтов в /v1/health содержит все публичные маршруты."""
    from fastapi.testclient import TestClient

    from services.gateway.app.main import app

    with TestClient(app) as client:
        payload = client.get("/v1/health").json()
    endpoints = payload["endpoints"]
    for expected in ("/v1/health", "/v1/warmup", "/v1/eval/predict", "/v1/scan",
                     "/v1/similar/{slug}", "/v1/sommelier", "/v1/enotourism/{slug}"):
        assert expected in endpoints, f"в /v1/health нет {expected}"


def test_tile_grid_arbitrary() -> None:
    """Сетки тайлов покрывают произвольные значения и сохраняют прежние."""
    assert preprocess._tile_grid(1) == (1, 1)
    assert preprocess._tile_grid(2) == (1, 2)
    assert preprocess._tile_grid(3) == (1, 3)
    assert preprocess._tile_grid(4) == (2, 2)
    assert preprocess._tile_grid(6) == (2, 3)
    assert preprocess._tile_grid(5) == (1, 5)
