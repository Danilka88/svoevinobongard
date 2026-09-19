# START_MODULE_CONTRACT
# PURPOSE: Верификация прототипов виноделен: построение, отображение позиций и бонус к скорам
# SCOPE: SLUGS, MAKERS, test_build_prototypes_empty_without_manufacturers, test_prototype_is_normalized_mean, test_winery_of_maps_positions, test_boost_rewards_close_wineries, test_boost_skips_unknown_positions, test_load_manufacturers_reads_v3, test_boost_returns_input_when_empty
# DEPENDS: services.retrieval.app.winery, numpy, json, pytest
# LINKS: M-RETRIEVAL, V-M-RETRIEVAL
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# SLUGS - тестовые slug позиций индекса
# MAKERS - соответствие slug на производителя
# test_build_prototypes_empty_without_manufacturers - без данных о производителях набор прототипов пуст
# test_prototype_is_normalized_mean - прототип винодельни это нормированное среднее её позиций
# test_winery_of_maps_positions - позиция индекса отображается на своего производителя
# test_boost_rewards_close_wineries - кандидаты близкой винодельни получают бонус и поднимаются
# test_boost_skips_unknown_positions - позиции без производителя бонуса не получают
# test_load_manufacturers_reads_v3 - производители читаются из обогащённого каталога v3
# test_boost_returns_input_when_empty - пустые прототипы не меняют выдачу
# END_MODULE_MAP
"""V-M-RETRIEVAL: прототипы виноделен."""
from __future__ import annotations

import json

import numpy as np

from services.retrieval.app.winery import WineryPrototypes, boost, build_prototypes, load_manufacturers

__all__ = [
    "test_boost_returns_input_when_empty",
    "test_boost_rewards_close_wineries",
    "test_boost_skips_unknown_positions",
    "test_build_prototypes_empty_without_manufacturers",
    "test_load_manufacturers_reads_v3",
    "test_prototype_is_normalized_mean",
    "test_winery_of_maps_positions",
]

SLUGS = ["a-1", "a-2", "b-1", "b-2"]
MAKERS = {"a-1": "maker-a", "a-2": "maker-a", "b-1": "maker-b", "b-2": "maker-b"}


def _vectors() -> np.ndarray:
    """Два кластера: A около оси X, B около оси Y."""
    return np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.9, 0.1],
    ], dtype=np.float32)


def test_build_prototypes_empty_without_manufacturers() -> None:
    """Без данных о производителях набор прототипов пуст."""
    empty = build_prototypes(SLUGS, _vectors(), manufacturers={})
    assert len(empty) == 0
    assert empty.scores(np.array([1.0, 0.0, 0.0], dtype=np.float32)).shape == (0,)


def test_prototype_is_normalized_mean() -> None:
    """Прототип — нормированное среднее позиций винодельни."""
    prototypes = build_prototypes(SLUGS, _vectors(), manufacturers=MAKERS)
    assert len(prototypes) == 2
    norms = np.linalg.norm(prototypes.matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    names = set(prototypes.names)
    assert names == {"maker-a", "maker-b"}


def test_winery_of_maps_positions() -> None:
    """Позиция индекса отображается на своего производителя."""
    prototypes = build_prototypes(SLUGS, _vectors(), manufacturers=MAKERS)
    assert prototypes.winery_of(0) == prototypes.winery_of(1)
    assert prototypes.winery_of(2) == prototypes.winery_of(3)
    assert prototypes.winery_of(0) != prototypes.winery_of(2)
    assert prototypes.winery_of(999) == -1


def test_boost_rewards_close_wineries() -> None:
    """Кандидаты близкой винодельни получают бонус и поднимаются в выдаче."""
    prototypes = build_prototypes(SLUGS, _vectors(), manufacturers=MAKERS)
    query = np.array([1.0, 0.05, 0.0], dtype=np.float32)
    candidates = [
        {"slug": "b-1", "score": 0.80, "index": 2},
        {"slug": "a-1", "score": 0.75, "index": 0},
    ]
    boosted = boost(candidates, prototypes, query, weight=0.3, top_wineries=1)
    assert boosted[0]["slug"] == "a-1", "позиция близкой винодельни должна подняться"
    assert boosted[0]["winery_bonus"] > 0
    assert boosted[1]["winery_bonus"] == 0.0


def test_boost_skips_unknown_positions() -> None:
    """Позиции без производителя бонуса не получают."""
    prototypes = build_prototypes(SLUGS, _vectors(), manufacturers={"a-1": "maker-a"})
    candidates = [{"slug": "b-1", "score": 0.9, "index": 2}]
    boosted = boost(candidates, prototypes, np.array([1.0, 0.0, 0.0], dtype=np.float32), weight=0.5)
    assert boosted[0]["winery_bonus"] == 0.0
    assert boosted[0]["slug"] == "b-1"


def test_load_manufacturers_reads_v3(tmp_path) -> None:
    """Производители читаются из обогащённого каталога v3 по slug и manufacturer_slug."""
    target = tmp_path / "catalog_enriched_v3.jsonl"
    rows = [
        {"slug": "a-1", "manufacturer_slug": "maker-a"},
        {"slug": "b-1", "manufacturer_slug": ""},
        {"slug": "c-1"},
    ]
    target.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    mapping = load_manufacturers(target)
    assert mapping == {"a-1": "maker-a"}


def test_boost_returns_input_when_empty() -> None:
    """Пустые прототипы не меняют выдачу."""
    candidates = [{"slug": "a-1", "score": 0.5, "index": 0}]
    unchanged = boost(candidates, WineryPrototypes.empty(),
                      np.array([1.0, 0.0, 0.0], dtype=np.float32), weight=0.5)
    assert unchanged == candidates
