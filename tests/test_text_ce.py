# START_MODULE_CONTRACT
# PURPOSE: Верификация cross-encoder реранка: тексты пары, деградация без модели, бюджет и перестановка
# SCOPE: test_query_text_normalizes, test_candidate_text_joins_fields, test_rerank_none_without_model, test_rerank_none_on_empty_candidates, test_rerank_reorders_with_stub_model, test_rerank_budget_exceeded_returns_none
# DEPENDS: packages.common.text_ce, torch, pytest
# LINKS: M-TEXT, V-M-TEXT
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# CATALOG - тестовые записи каталога для пар запрос-кандидат
# test_query_text_normalizes - текст запроса приводится нормализацией к единому виду
# test_candidate_text_joins_fields - текст кандидата склеивается из полей каталога
# test_rerank_none_without_model - без весов реранк возвращает None и путь не меняется
# test_rerank_none_on_empty_candidates - пустой список кандидатов не вызывает модель
# test_rerank_reorders_with_stub_model - при доступной модели порядок меняется по её оценкам
# test_rerank_budget_exceeded_returns_none - при перерасходе бюджета результат не применяется
# END_MODULE_MAP
"""V-M-TEXT: cross-encoder реранк."""
from __future__ import annotations

import pytest
import torch

from packages.common import text_ce

__all__ = [
    "test_candidate_text_joins_fields",
    "test_query_text_normalizes",
    "test_rerank_budget_exceeded_returns_none",
    "test_rerank_none_on_empty_candidates",
    "test_rerank_none_without_model",
    "test_rerank_reorders_with_stub_model",
]

CATALOG = {
    "wine-a": {"slug": "wine-a", "name": "Шато Тамань Полусладкое", "winery": "Кубань-Вино",
               "grape": "Рислинг", "region": "Кубань", "sweetness": "полусладкое", "year": 2022},
    "wine-b": {"slug": "wine-b", "name": "Абрау Дюрсо Брют", "winery": "Абрау-Дюрсо",
               "grape": "Шардоне", "region": "Кубань"},
}


def test_query_text_normalizes() -> None:
    """Текст запроса приводится нормализацией к единому виду."""
    assert text_ce.query_text({"winery": "Chateau Tamagne", "wine_name": "semi-sweet"}) == \
        "chateau tamagne semi sweet"


def test_candidate_text_joins_fields() -> None:
    """Текст кандидата склеивается из полей каталога, включая год и сорта."""
    text = text_ce.candidate_text(CATALOG["wine-a"])
    for expected in ("Шато Тамань Полусладкое", "Кубань-Вино", "Рислинг", "2022"):
        assert expected in text


def test_rerank_none_without_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Без весов реранк возвращает None, и вызывающий оставляет путь без изменений."""
    monkeypatch.setattr(text_ce, "resolve_ce", lambda: None)
    monkeypatch.setattr(text_ce, "_cache", {})
    assert text_ce.rerank({"winery": "Абрау"}, [("wine-a", 1.0)], CATALOG) is None


def test_rerank_none_on_empty_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Пустой список кандидатов не вызывает модель."""
    monkeypatch.setattr(text_ce, "_cache", {"loaded": (object(), object(), "stub")})
    assert text_ce.rerank({"winery": "Абрау"}, [], CATALOG) is None


class _StubTokenizer:
    """Заглушка токенизатора: важно лишь, что модель получит пары."""

    def __call__(self, queries, candidates, **kwargs):  # noqa: D102
        return {"pairs": len(queries)}


class _StubOutput:
    """Заглушка выхода модели с логитами по числу пар."""

    def __init__(self, size: int) -> None:
        self.logits = torch.zeros((size, 1))


class _StubModel:
    """Заглушка модели: первая пара получает низкий логит, вторая — высокий."""

    def __call__(self, **kwargs):  # noqa: D102
        size = int(kwargs.get("pairs", 2))
        output = _StubOutput(size)
        output.logits[0, 0] = -5.0
        output.logits[1, 0] = 5.0
        return output


def test_rerank_reorders_with_stub_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """При доступной модели порядок меняется по её оценкам."""
    monkeypatch.setattr(text_ce, "_cache", {"loaded": (_StubModel(), _StubTokenizer(), "stub")})
    ranked = text_ce.rerank({"winery": "Абрау-Дюрсо"},
                            [("wine-a", 1.0), ("wine-b", 0.9)], CATALOG, budget_ms=10000.0)
    assert ranked is not None and ranked[0][0] == "wine-b"


def test_rerank_budget_exceeded_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """При перерасходе бюджета результат не применяется."""
    monkeypatch.setattr(text_ce, "_cache", {"loaded": (_StubModel(), _StubTokenizer(), "stub")})
    assert text_ce.rerank({"winery": "Абрау"}, [("wine-a", 1.0), ("wine-b", 0.9)], CATALOG,
                          budget_ms=-1.0) is None
