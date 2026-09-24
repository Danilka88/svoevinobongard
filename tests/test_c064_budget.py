# START_MODULE_CONTRACT
# PURPOSE: Проверка бюджета наблюдателя: настройки, деградация без ошибки, нормализация омоглифов и неизменность латинских брендов
# SCOPE: PROBLEM_CASES, LATIN_CASES, test_homoglyphs_become_cyrillic, test_latin_brands_unchanged,
#        test_mixed_token_normalised, test_observation_fields_normalise, test_observation_fields_can_skip_normalisation,
#        test_raw_text_used_when_fields_empty, test_budget_settings, test_observe_label_degrades_without_error,
#        test_gateway_keeps_tier1_when_observer_fails, test_observer_disabled_by_default
# DEPENDS: asyncio, numpy, pytest, packages.common.config, packages.common.observer, packages.common.text_norm,
#          packages.contracts.schemas, services.gateway.app.main, services.gateway.app.pipeline, services.vlm.app.extract
# LINKS: M-VLM, M-EVAL, M-GATEWAY
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# PROBLEM_CASES - наблюдения с греческими омоглифами
# LATIN_CASES - латинские бренды, которые нельзя менять
# client - тестовый клиент gateway
# test_homoglyphs_become_cyrillic - греческие двойники переводятся в кириллицу
# test_latin_brands_unchanged - латинские бренды не портятся
# test_mixed_token_normalised - смешанный токен приводится к кириллице
# test_observation_fields_normalise - поля наблюдения нормализуются
# test_observation_fields_can_skip_normalisation - нормализацию можно отключить для сравнения
# test_raw_text_used_when_fields_empty - свободный текст идёт в поиск при пустых полях
# test_budget_settings - бюджет наблюдателя настраивается и укладывается в лимит организатора
# test_observe_label_degrades_without_error - при исчерпании бюджета возвращается неуспех без исключения
# test_gateway_keeps_tier1_when_observer_fails - gateway отвечает результатом поиска, когда наблюдатель не succeeded
# test_observer_disabled_by_default - по умолчанию режим выключен
# END_MODULE_MAP
"""Бюджет наблюдателя и омоглифы: проверки без сети, через реальные функции и подставные клиенты."""
from __future__ import annotations

import asyncio

import numpy as np
import pytest
from fastapi.testclient import TestClient

from packages.common.config import settings
from packages.common.observer import observation_fields, observer_candidates, parse_observation
from packages.common.text_norm import greek_to_cyrillic
from packages.contracts.schemas import ScanResponse
from services.gateway.app import main as gw
from services.gateway.app import pipeline
from services.vlm.app import extract as vlm_extract

client = TestClient(gw.app)

PROBLEM_CASES = ("ΛΣΚΙΦ", "ΦΑΝΑΓΟΡΙΑ", "ΛΗΚΥΘΟΣ", "ΜΑΣΣΑΝΔΡΑ")
LATIN_CASES = ("FANAGORIA", "CHATEAU PINOT", "UVA VALLIS", "MASANDRA")

__all__ = ["PROBLEM_CASES", "LATIN_CASES", "test_homoglyphs_become_cyrillic",
           "test_latin_brands_unchanged", "test_mixed_token_normalised",
           "test_observation_fields_normalise", "test_observation_fields_can_skip_normalisation",
           "test_raw_text_used_when_fields_empty", "test_budget_settings",
           "test_observe_label_degrades_without_error", "test_gateway_keeps_tier1_when_observer_fails",
           "test_observer_disabled_by_default"]


def test_homoglyphs_become_cyrillic() -> None:
    """Греческие визуальные двойники переводятся в кириллицу: иначе поиск по каталогу промахивается."""
    results = {case: greek_to_cyrillic(case) for case in PROBLEM_CASES}
    assert results["ΛΣΚΙΦ"] == "ЛСКИФ"
    assert results["ΦΑΝΑΓΟΡΙΑ"] == "ФАНАГОРИА"
    assert results["ΜΑΣΣΑΝΔΡΑ"] == "МАССАНДРА"
    assert all(not any(char in result for char in "ΛΣΚΙΦΑΗΥΘΟΜ") for result in results.values())


def test_latin_brands_unchanged() -> None:
    """Латинские бренды не трогаются: иначе мы сломали бы корректные наблюдения."""
    for case in LATIN_CASES:
        assert greek_to_cyrillic(case) == case


def test_mixed_token_normalised() -> None:
    """Токен с греческой буквой приводится целиком: «ΚΡΑСНАЯ» → «КРАСНАЯ»."""
    assert greek_to_cyrillic("ΚΡΑСНАЯ СТРЕЛА") == "КРАСНАЯ СТРЕЛА"


def test_observation_fields_normalise() -> None:
    fields = observation_fields(parse_observation("winery: ΛΣΚΙΦ\nwine_name: ΛΗΚΥΘΟΣ"))
    assert fields["winery"] == "ЛСКИФ"
    assert fields["wine_name"] == "ЛНКУООС"


def test_observation_fields_can_skip_normalisation() -> None:
    """Сравнение до/после нормализации: отключение даёт исходные греческие буквы."""
    observation = parse_observation("winery: ΛΣΚΙΦ")
    assert observation_fields(observation, normalize=False)["winery"] == "ΛΣΚΙΦ"


def test_raw_text_used_when_fields_empty() -> None:
    """Модель положила всё в свободный текст: поиск использует его, а не остаётся пустым."""
    observation = parse_observation("winery:\nwine_name:\nraw_text: СКАЛИСТЫЙ БЕРЕГ ШЁПОТ ЦВЕТОВ")
    fields = observation_fields(observation)
    # Смысл проверки: свободный текст не теряется и попадает в поиск как поле.
    assert fields.get("wine_name") == "СКАЛИСТЫЙ БЕРЕГ ШЁПОТ ЦВЕТОВ"
    catalog = {"skalistyj-bereg-shepot-czvetov": {"name": "СКАЛИСТЫЙ БЕРЕГ ШЁПОТ ЦВЕТОВ",
                                                 "winery": "Скалистый берег"}}
    hits = observer_candidates(observation, catalog, limit=1)
    assert hits and hits[0][0] == "skalistyj-bereg-shepot-czvetov"


def test_budget_settings() -> None:
    """Бюджет наблюдателя укладывается в десять секунд скрипта организатора."""
    assert 0 < settings.observer_timeout_s < 10.0
    assert settings.observer_max_side <= 640
    assert settings.observer_max_tokens <= 512
    assert settings.observer_model == "gemma4:e4b-mlx"  # выбрана по бюджету и точности бенча


def test_observe_label_degrades_without_error(monkeypatch) -> None:
    """При исчерпании бюджета возвращается неуспех без исключения, а не пустой ответ."""
    monkeypatch.setattr(vlm_extract, "_observe_native_ollama", lambda *args, **kwargs: None)
    image = np.full((64, 64, 3), 180, dtype=np.uint8)
    result = vlm_extract.observe_label([image], None, model="gemma4:e4b-mlx", timeout_s=0.1)
    assert result["ok"] is False
    assert "budget" in str(result.get("reason") or "")
    assert not result.get("raw_text")


class _SlowObserverClients:
    """Клиенты, у которых наблюдатель не успел: ответ должен остаться результатом поиска."""

    def __init__(self) -> None:
        self.observe_calls = 0

    async def normalize(self, data, filename="q.jpg"):
        return {"method": "identity", "crops": []}

    async def search(self, data, filename="q.png"):
        return {"top_k": [{"slug": "massandra-muskatel", "score": 0.30, "name": "Мускатель белый",
                           "winery": "Массандра"}]}

    async def ocr(self, data):
        return None

    async def rerank(self, items, fields):
        return {"items": items}

    async def extract(self, data, filename="q.png"):
        return None

    async def observe(self, data, filename="q.png", timeout=None):
        self.observe_calls += 1
        return {"ok": False, "reason": "observer budget exceeded"}


def test_gateway_keeps_tier1_when_observer_fails(monkeypatch) -> None:
    """Gateway обязан ответить: при неуспевшем наблюдателе отдаётся лучший результат поиска."""
    from types import SimpleNamespace
    values = {name: getattr(settings, name) for name in dir(settings) if not name.startswith("_")}
    values.update({"observer_enabled": True, "observer_model": "gemma4:e4b-mlx", "observer_top_k": 10,
                   "observer_threshold": 0.95, "calibration_enabled": False, "calibration_path": ""})
    monkeypatch.setattr(pipeline, "settings", SimpleNamespace(**values))
    monkeypatch.setattr(pipeline, "_CALIBRATION_CACHE", {})
    clients = _SlowObserverClients()
    import io
    from PIL import Image
    buffer = io.BytesIO()
    Image.fromarray(np.full((48, 48, 3), 200, dtype=np.uint8)).save(buffer, format="PNG")
    response = asyncio.run(pipeline.run_scan(buffer.getvalue(), gw.get_store(), clients=clients,
                                             filename="q.png"))
    assert clients.observe_calls == 1          # наблюдатель вызывался
    assert isinstance(response, ScanResponse)  # и ответ всё равно есть
    assert response.slug == "massandra-muskatel"
    assert response.path != "observer"
    assert response.observation == {}


def test_observer_disabled_by_default() -> None:
    assert settings.observer_enabled is False
    assert settings.calibration_enabled is False
