# START_MODULE_CONTRACT
# PURPOSE: Устойчивость цифрового сомелье: бюджет LLM по времени и гарантированный детерминированный fallback
# SCOPE: RECORDS, client, test_slow_llm_returns_deterministic_within_budget,
#        test_empty_llm_reply_falls_back_to_catalog, test_llm_failure_returns_200_not_5xx,
#        test_good_llm_reply_marks_source_llm, test_dirty_sommelier_reply_is_parsed_and_filtered,
#        test_sommelier_schema_has_degradation_fields, test_budget_and_warmup_defaults,
#        test_unknown_slug_still_404
# DEPENDS: time, pytest, fastapi.testclient, packages.common.catalog, packages.common.config,
#          packages.common.llm.errors, packages.contracts.schemas, services.gateway.app.main,
#          services.gateway.app.recommend
# LINKS: M-GATEWAY, M-WEB, M-CONTRACTS
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# RECORDS - маленький каталог для подбора без сети
# client - тестовый клиент gateway
# test_slow_llm_returns_deterministic_within_budget - медленный LLM: ответ в бюджете, детерминированные данные
# test_empty_llm_reply_falls_back_to_catalog - пустой ответ LLM: подбор по каталогу, а не ошибка
# test_llm_failure_returns_200_not_5xx - отказ LLM: HTTP 200 с данными, без 5xx
# test_good_llm_reply_marks_source_llm - успешный LLM: source=llm и llm_degraded=false
# test_dirty_sommelier_reply_is_parsed_and_filtered - «грязный» ответ разбирается и фильтруется по каталогу
# test_sommelier_schema_has_degradation_fields - схема несёт llm_degraded и source
# test_budget_and_warmup_defaults - бюджет положителен, прогрев по умолчанию выключен
# test_unknown_slug_still_404 - контракт неизвестного slug сохраняется
# END_MODULE_MAP
"""C-071: сомелье не зависит от состояния локальной text-модели.

Тесты подменяют LLM-роутер подставными объектами: медленным, пустым, отказывающим и корректным.
Ни одного сетевого вызова: каталог и официальное обогащение тоже подменяются. Проверяется главная
гарантия — эндпоинт всегда отвечает детерминированными данными в пределах бюджета и не отдаёт 5xx.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from packages.common.catalog import CatalogStore
from packages.common.config import settings
from packages.common.llm.errors import AllProvidersExhausted
from packages.contracts.schemas import SommelierResponse
from services.gateway.app import main as gw
from services.gateway.app.recommend import parse_sommelier_reply

__all__ = [
    "RECORDS",
    "client",
    "test_slow_llm_returns_deterministic_within_budget",
    "test_empty_llm_reply_falls_back_to_catalog",
    "test_llm_failure_returns_200_not_5xx",
    "test_good_llm_reply_marks_source_llm",
    "test_dirty_sommelier_reply_is_parsed_and_filtered",
    "test_sommelier_schema_has_degradation_fields",
    "test_budget_and_warmup_defaults",
    "test_unknown_slug_still_404",
]

client = TestClient(gw.app)

RECORDS = {
    "base": {"slug": "base", "name": "Кокур Сухое", "winery": "Захарьин", "region": "Крым",
             "category": "Белое", "grape": "Кокур", "sweetness": "сухое", "description": ""},
    "same-grape": {"slug": "same-grape", "name": "Кокур Резерв", "winery": "Другая", "region": "Кубань",
                   "category": "Белое", "grape": "Кокур", "sweetness": "сухое", "description": ""},
    "same-region": {"slug": "same-region", "name": "Рислинг", "winery": "Третья", "region": "Крым",
                    "category": "Белое", "grape": "Рислинг", "sweetness": "сухое", "description": ""},
    "red": {"slug": "red", "name": "Каберне", "winery": "Четвёртая", "region": "Кубань",
            "category": "Красное", "grape": "Каберне", "sweetness": "сухое", "description": ""},
}


def _store(records: dict | None = None) -> CatalogStore:
    return CatalogStore(records or RECORDS)


class _SlowRouter:
    """Медленная модель: не успевает в бюджет и не должна блокировать ответ."""

    def chat(self, *args, **kwargs):
        time.sleep(1.0)
        return SimpleNamespace(text='{"recommendations":[],"pairing":[]}')


class _EmptyRouter:
    """Модель вернула пустую строку: разобрать нечего."""

    def chat(self, *args, **kwargs):
        return SimpleNamespace(text="")


class _BrokenRouter:
    """Провайдеры исчерпаны: типизированный отказ LLM-слоя."""

    def chat(self, *args, **kwargs):
        raise AllProvidersExhausted("text_model", [])


class _GoodRouter:
    """Корректный ответ модели по кандидату из каталога."""

    def chat(self, *args, **kwargs):
        return SimpleNamespace(
            text='{"recommendations":[{"slug":"same-grape","why":"тот же сорт"}],"pairing":["Рыба"]}')


def _patch_gateway(monkeypatch, router, *, timeout: float) -> None:
    """Подменяет каталог, официальное обогащение, роутер и бюджет без сети."""
    monkeypatch.setattr(gw, "get_store", lambda: _store())
    monkeypatch.setattr(gw, "get_official", lambda: {})
    monkeypatch.setattr(gw, "LLMRouter", lambda *args, **kwargs: router)
    monkeypatch.setattr(gw, "_SOMMELIER_LLM_TIMEOUT_S", timeout)


def test_slow_llm_returns_deterministic_within_budget(monkeypatch) -> None:
    """Медленный LLM: ответ приходит в пределах бюджета и с детерминированными данными."""
    _patch_gateway(monkeypatch, _SlowRouter(), timeout=0.2)
    started = time.perf_counter()
    response = client.post("/v1/sommelier", json={"slug": "", "answers": {"style": "сухое"}})
    elapsed = time.perf_counter() - started
    assert response.status_code == 200
    body = response.json()
    assert body["recommendations"], "детерминированный подбор не должен быть пустым"
    assert body["source"] in {"deterministic", "official"}
    assert body["llm_degraded"] is True
    assert elapsed < 1.0, f"ответ должен уложиться в бюджет, вышло {elapsed:.2f}s"


def test_empty_llm_reply_falls_back_to_catalog(monkeypatch) -> None:
    """Пустой ответ модели — не ошибка: фича отдаёт подбор по каталогу."""
    _patch_gateway(monkeypatch, _EmptyRouter(), timeout=1.0)
    response = client.post("/v1/sommelier", json={"slug": "", "answers": {"style": "сухое"}})
    assert response.status_code == 200
    body = response.json()
    assert body["recommendations"]
    assert body["llm_degraded"] is True


def test_llm_failure_returns_200_not_5xx(monkeypatch) -> None:
    """Отказ LLM-слоя не превращается в 5xx: пользователь получает данные каталога."""
    _patch_gateway(monkeypatch, _BrokenRouter(), timeout=1.0)
    response = client.post("/v1/sommelier", json={"slug": "", "answers": {}})
    assert response.status_code == 200
    body = response.json()
    assert body["recommendations"]
    assert body["llm_degraded"] is True
    assert "llm unavailable" in body["note"] or body["source"] in {"deterministic", "official"}


def test_good_llm_reply_marks_source_llm(monkeypatch) -> None:
    """Успешный разбор модели: source=llm, деградация не выставляется."""
    _patch_gateway(monkeypatch, _GoodRouter(), timeout=1.0)
    response = client.post("/v1/sommelier", json={"slug": "", "answers": {"style": "сухое"}})
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "llm"
    assert body["llm_degraded"] is False
    assert [item["slug"] for item in body["recommendations"]] == ["same-grape"]
    assert body["pairing"] == ["Рыба"]


def test_dirty_sommelier_reply_is_parsed_and_filtered() -> None:
    """«Грязный» ответ (фенсы, текст вокруг, лишний slug) разбирается и фильтруется по каталогу."""
    dirty = ('Вот результат:\n```json\n{"recommendations":[{"slug":"same-grape","why":"тот же сорт"},'
             '{"slug":"hallucinated","why":"нет в каталоге"}],"pairing":["Рыба"]}\n```\nГотово.')
    parsed = parse_sommelier_reply(dirty, {"same-grape", "same-region"})
    assert [item["slug"] for item in parsed["recommendations"]] == ["same-grape"]
    assert parsed["pairing"] == ["Рыба"]


def test_sommelier_schema_has_degradation_fields() -> None:
    """Схема ответа несёт source и аддитивный llm_degraded с безопасным дефолтом."""
    response = SommelierResponse(slug="a")
    assert response.llm_degraded is False
    assert isinstance(response.source, str)


def test_budget_and_warmup_defaults() -> None:
    """Бюджет положителен и меньше лимита, прогрев text-модели по умолчанию выключен."""
    assert 0 < gw._SOMMELIER_LLM_TIMEOUT_S < 10.0
    assert 0 < settings.sommelier_llm_timeout_s < 10.0
    assert settings.sommelier_warmup_enabled is False


def test_unknown_slug_still_404(monkeypatch) -> None:
    """Существующий контракт сохранён: неизвестный slug даёт 404, а не подбор."""
    _patch_gateway(monkeypatch, _GoodRouter(), timeout=1.0)
    response = client.post("/v1/sommelier", json={"slug": "no-such-slug", "answers": {}})
    assert response.status_code == 404
