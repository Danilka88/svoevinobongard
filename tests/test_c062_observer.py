# START_MODULE_CONTRACT
# PURPOSE: Проверка режима наблюдателя: разбор наблюдения, поиск по каталогу, гибрид с визуальным топом и ветка решения за флагом
# SCOPE: CATALOG, TEST_CATALOG, test_parse_key_value_lines, test_parse_json_form, test_parse_partial_and_empty,
#        test_parse_smart_quotes, test_fields_normalise_synonyms, test_observer_candidates_find_slug,
#        test_merge_boosts_shared_candidate, test_merge_allows_observer_only_winner,
#        test_escalation_trigger_matrix, test_observer_path_wins_on_low_confidence,
#        test_observer_skipped_on_high_confidence, test_observer_disabled_by_default,
#        test_eval_predict_stays_flat
# DEPENDS: json, asyncio, numpy, pytest, fastapi.testclient, packages.common.observer, packages.contracts.schemas,
#          services.gateway.app.main, services.gateway.app.pipeline
# LINKS: M-EVAL, M-GATEWAY, M-VLM
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# CATALOG - синтетический каталог для поиска по наблюдению
# TEST_CATALOG - каталог с двумя похожими винами для гибрида
# test_parse_key_value_lines - строки «ключ: значение» разбираются в поля
# test_parse_json_form - JSON-форма наблюдения поддерживается
# test_parse_partial_and_empty - частичный и пустой ответ не ломают разбор
# test_parse_smart_quotes - «умные» кавычки и маркеры списка убираются
# test_fields_normalise_synonyms - значения нормализуются по синонимам
# test_observer_candidates_find_slug - наблюдение находит вино по всему каталогу
# test_merge_boosts_shared_candidate - совпадение с визуальным топом даёт буст
# test_merge_allows_observer_only_winner - кандидат только из наблюдения может победить
# test_escalation_trigger_matrix - триггер наблюдателя по вероятности и уверенности
# test_observer_path_wins_on_low_confidence - при низкой уверенности побеждает наблюдатель
# test_observer_skipped_on_high_confidence - при высокой уверенности наблюдатель не вызывается
# test_observer_disabled_by_default - по умолчанию режим выключен, поведение прежнее
# test_eval_predict_stays_flat - контракт организатора остаётся плоским
# OBSERVATION - пример наблюдения модели для проверок
# client - тестовый клиент gateway
# END_MODULE_MAP
"""Проверка наблюдателя на моках: без сети и моделей, через настоящий пайплайн."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from packages.common.config import settings
from packages.common.observer import (
    escalation_trigger,
    merge_candidates,
    observation_fields,
    observer_candidates,
    parse_observation,
)
from packages.contracts.schemas import ScanResponse
from services.gateway.app import main as gw
from services.gateway.app import pipeline

__all__ = [
    "CATALOG",
    "TEST_CATALOG",
    "test_parse_key_value_lines",
    "test_parse_json_form",
    "test_parse_partial_and_empty",
    "test_parse_smart_quotes",
    "test_fields_normalise_synonyms",
    "test_observer_candidates_find_slug",
    "test_merge_boosts_shared_candidate",
    "test_merge_allows_observer_only_winner",
    "test_escalation_trigger_matrix",
    "test_observer_path_wins_on_low_confidence",
    "test_observer_skipped_on_high_confidence",
    "test_observer_disabled_by_default",
    "test_eval_predict_stays_flat",
    "OBSERVATION",
    "client",
    "_FakeClients",
    "_run",
]

client = TestClient(gw.app)

CATALOG = {
    "skalistyy-bereg-shepot-czvetov": {"name": "Шёпот цветов", "winery": "Скалистый берег",
                                       "grape": "Рислинг", "year": 2024, "sweetness": "брют"},
    "nebо-rossii-glera": {"name": "Глера", "winery": "Небо России", "grape": "Глера", "year": 2022,
                          "sweetness": "сухое"},
    "massandra-muskatel": {"name": "Мускатель белый", "winery": "Массандра", "grape": "Мускатель",
                           "year": 2023, "sweetness": "сладкое"},
}
TEST_CATALOG = {
    "series-2023": {"name": "Шёпот цветов", "winery": "Скалистый берег", "year": 2023, "sweetness": "сухое"},
    "series-2024": {"name": "Шёпот цветов", "winery": "Скалистый берег", "year": 2024, "sweetness": "брют"},
}

OBSERVATION = "винодельня: Скалистый берег\nназвание: Шёпот цветов\nгод: 2024\nсладость: Брют"


class _FakeClients:
    """Подставные клиенты: наблюдатель отвечает заранее заданным наблюдением."""

    def __init__(self, items=None, observation=OBSERVATION) -> None:
        self.items = items if items is not None else [{"slug": "massandra-muskatel", "score": 0.30,
                                                       "name": "Мускатель белый", "winery": "Массандра"}]
        self.observation = observation
        self.observe_calls = 0

    async def normalize(self, data, filename="q.jpg"):
        return {"method": "identity", "crops": []}

    async def search(self, data, filename="q.png"):
        return {"top_k": list(self.items)}

    async def ocr(self, data):
        return None

    async def rerank(self, items, fields):
        return {"items": items}

    async def extract(self, data, filename="q.png"):
        return None

    async def observe(self, data, filename="q.png", timeout=None):
        self.observe_calls += 1
        return {"ok": True, "raw_text": self.observation, "model": settings.observer_model}


# --- разбор наблюдения -----------------------------------------------------------

def test_parse_key_value_lines() -> None:
    observation = parse_observation(OBSERVATION)
    assert observation["winery"] == "Скалистый берег"
    assert observation["wine_name"] == "Шёпот цветов"
    assert observation["vintage"] == "2024"
    assert observation["sweetness"] == "Брют"


def test_parse_json_form() -> None:
    observation = parse_observation('{"winery": "Массандра", "sweetness": "dry"}')
    assert observation["winery"] == "Массандра" and observation["sweetness"] == "dry"
    assert observation_fields(observation)["sweetness"] == "сухое"


def test_parse_partial_and_empty() -> None:
    assert parse_observation("") == {"raw_text": ""}
    assert parse_observation(None) == {"raw_text": ""}
    partial = parse_observation("название: Шёпот\nчастично видно MACCAHAPA и год 20__")
    assert partial["wine_name"] == "Шёпот"
    # Нераспознанные строки не теряются: наблюдение сохраняется целиком для поиска.
    assert "MACCAHAPA" in (partial.get("raw_text") or "") or "MACCAHAPA" in str(partial)


def test_parse_smart_quotes() -> None:
    observation = parse_observation("- “винодельня”: «Скалистый берег»\n* название = Шёпот цветов")
    assert observation["winery"] == "Скалистый берег"
    assert observation["wine_name"] == "Шёпот цветов"


@pytest.mark.parametrize(("raw", "expected"), [("Брют", "брют"), ("DRY", "сухое"), ("полусухое", "полусухое")])
def test_fields_normalise_synonyms(raw: str, expected: str) -> None:
    assert observation_fields({"sweetness": raw})["sweetness"] == expected


# --- поиск и гибрид -------------------------------------------------------------

def test_observer_candidates_find_slug() -> None:
    hits = observer_candidates(parse_observation(OBSERVATION), CATALOG, limit=3)
    assert hits and hits[0][0] == "skalistyy-bereg-shepot-czvetov"


def test_merge_boosts_shared_candidate() -> None:
    merged = merge_candidates([("a", 0.4)], [("a", 0.5)])
    assert merged[0][0] == "a" and merged[0][1] > 0.5  # буст за совпадение сигналов


def test_merge_allows_observer_only_winner() -> None:
    merged = merge_candidates([("new", 0.9)], [("old", 0.5)])
    assert merged[0][0] == "new"


def test_escalation_trigger_matrix() -> None:
    assert escalation_trigger(0.90, True, 0.95) is True
    assert escalation_trigger(0.99, False, 0.95) is False
    assert escalation_trigger(None, False) is True   # калибровки нет, поиск не уверен
    assert escalation_trigger(None, True) is False   # калибровки нет, поиск уверен


# --- ветка решения --------------------------------------------------------------

def _run(clients, monkeypatch, enabled=True):
    monkeypatch.setattr(pipeline, "settings", SimpleNamespace(
        **{**{name: getattr(settings, name) for name in dir(settings) if not name.startswith("_")},
           "observer_enabled": enabled, "observer_model": "qwen3-vl:2b", "observer_top_k": 10,
           "observer_threshold": 0.95, "calibration_enabled": False,
           "calibration_path": "", "pas_tiles": settings.pas_tiles,
           "vlm_num_ctx": settings.vlm_num_ctx, "vlm_reasoning_effort": settings.vlm_reasoning_effort}))
    image = np.full((64, 64, 3), 200, dtype=np.uint8)
    import io
    from PIL import Image
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return asyncio.run(pipeline.run_scan(buffer.getvalue(), gw.get_store(), clients=clients,
                                         filename="q.png"))


def test_observer_path_wins_on_low_confidence(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "_CALIBRATION_CACHE", {})
    clients = _FakeClients()
    response = _run(clients, monkeypatch)
    assert clients.observe_calls == 1
    # Ожидаемого победителя берём из фактического поиска по реальному каталогу: транслитерация
    # slug в каталоге может отличаться от синтетической, и подгонять её нельзя.
    hits = observer_candidates(parse_observation(OBSERVATION), gw.get_store().slug_to_record(), limit=3)
    assert hits, "наблюдение должно находить вино в каталоге"
    assert response.path == "observer"
    assert response.slug == hits[0][0]
    assert response.observation.get("wine_name") == "Шёпот цветов"
    assert response.not_found is False
    assert response.top5 and response.top5[0].slug == response.slug


def test_observer_skipped_on_high_confidence(monkeypatch) -> None:
    """При уверенном визуальном ответе наблюдатель не вызывается: лишних затрат нет."""
    monkeypatch.setattr(pipeline, "_CALIBRATION_CACHE", {})
    clients = _FakeClients(items=[{"slug": "massandra-muskatel", "score": 0.95,
                                   "name": "Мускатель белый", "winery": "Массандра"},
                                 {"slug": "nebо-rossii-glera", "score": 0.30, "name": "Глера",
                                  "winery": "Небо России"}])
    response = _run(clients, monkeypatch)
    assert clients.observe_calls == 0
    assert response.path != "observer"
    assert response.observation == {}


def test_observer_disabled_by_default() -> None:
    """Флаг выключен по умолчанию: режим не активен и поведение не меняется."""
    assert settings.observer_enabled is False
    assert settings.calibration_enabled is False


def test_eval_predict_stays_flat(monkeypatch) -> None:
    """Контракт организатора остаётся плоским независимо от наблюдателя."""
    async def fake_scan(*args, **kwargs):
        return ScanResponse(slug="massandra-muskatel", confidence=0.4)

    monkeypatch.setattr(gw, "run_scan", fake_scan)
    # C-067: контракт проверяется на реально декодируемом изображении — битый вход отсекается раньше.
    import io
    from PIL import Image
    buffer = io.BytesIO()
    Image.fromarray(np.full((32, 32, 3), 180, dtype=np.uint8)).save(buffer, format="PNG")
    body = client.post("/v1/eval/predict",
                       files={"image": ("q.png", buffer.getvalue(), "image/png")}).json()
    assert set(body.keys()) == {"slug"} and body["slug"] == "massandra-muskatel"
