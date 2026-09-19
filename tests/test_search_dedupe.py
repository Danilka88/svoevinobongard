# START_MODULE_CONTRACT
# PURPOSE: Верификация дедупликации top-K поиска по slug (баг C-026: галерея давала дубликаты в выдаче)
# SCOPE: test_dedupe_keeps_max_score, test_dedupe_preserves_order, test_dedupe_truncates, test_dedupe_empty, test_pool_factor_allows_many_views
# DEPENDS: services.retrieval.app.search, pytest
# LINKS: M-RETRIEVAL, V-M-RETRIEVAL
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# test_dedupe_keeps_max_score - из дублей slug остаётся максимальный скор
# test_dedupe_preserves_order - порядок сортируется по убыванию скора
# test_dedupe_truncates - результат обрезается до лимита
# test_dedupe_empty - пустой вход даёт пустой выход
# test_pool_factor_allows_many_views - пул строк покрывает виды одного вина так, что разных slug хватает на k
# END_MODULE_MAP
"""V-M-RETRIEVAL: дедупликация выдачи."""
from __future__ import annotations

from services.retrieval.app.search import POOL_FACTOR, dedupe_by_slug

__all__ = [
    "test_dedupe_empty",
    "test_dedupe_keeps_max_score",
    "test_dedupe_preserves_order",
    "test_dedupe_truncates",
    "test_pool_factor_allows_many_views",
]


def test_dedupe_keeps_max_score() -> None:
    """Из нескольких видов одного slug остаётся максимальный скор."""
    rows = [("a", 0.3), ("b", 0.9), ("a", 0.7), ("a", 0.5)]
    assert dedupe_by_slug(rows, 10) == [("b", 0.9), ("a", 0.7)]


def test_dedupe_preserves_order() -> None:
    """Порядок — по убыванию итогового скора."""
    rows = [("x", 0.1), ("y", 0.4), ("x", 0.9), ("z", 0.5)]
    assert [slug for slug, _ in dedupe_by_slug(rows, 10)] == ["x", "z", "y"]


def test_dedupe_truncates() -> None:
    """Результат обрезается до запрошенного лимита."""
    rows = [(f"s{index}", index / 100) for index in range(20)]
    assert len(dedupe_by_slug(rows, 5)) == 5


def test_dedupe_empty() -> None:
    """Пустой вход не ломает функцию."""
    assert dedupe_by_slug([], 5) == []


def test_pool_factor_allows_many_views() -> None:
    """При шести видах на вино пул строк даёт достаточно разных slug на top-K."""
    k, views = 30, 6
    pool = max(k * POOL_FACTOR, 200)
    rows = []
    for position in range(pool // views + 1):
        for view in range(views):
            rows.append((f"wine-{position}", 1.0 - position / 1000 - view / 10000))
    for position in range(pool // views + 1):
        for view in range(views):
            rows.append((f"wine-{position}", 1.0 - position / 1000 - view / 10000))
    assert len(dedupe_by_slug(rows, k)) == k
