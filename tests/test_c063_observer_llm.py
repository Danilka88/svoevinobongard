# START_MODULE_CONTRACT
# PURPOSE: Проверка рабочего вызова наблюдателя: конфигурация моделей, промпт с примером, формат наблюдения и выбор модели бенчем
# SCOPE: PROVIDERS, PROXY_PAIRS, test_provider_models_exist_in_registry, test_observer_instruction_has_example,
#        test_parses_english_key_format, test_parses_real_e4b_observation, test_observe_label_signature,
#        test_observer_model_follows_bench, test_calibration_report_labels_distributions
# DEPENDS: json, pathlib, subprocess, pytest, packages.common.config, packages.common.observer,
#          services.vlm.app.extract
# LINKS: M-VLM, M-EVAL
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# PROVIDERS - путь к конфигурации маршрутизации моделей
# PROXY_PAIRS - путь к пер-парам proxy-eval
# REAL_E4B - реальное наблюдение слабой модели, полученное вживую
# test_provider_models_exist_in_registry - все локальные модели цепочек присутствуют в реестре
# test_observer_instruction_has_example - инструкция наблюдателя содержит пример вывода
# test_parses_english_key_format - наблюдение с английскими ключами разбирается в поля
# test_parses_real_e4b_observation - реальное наблюдение слабой модели разбирается
# test_observe_label_signature - вызов наблюдателя возвращает словарь с нужными полями
# test_observer_model_follows_bench - модель наблюдателя выбрана по результатам бенча
# test_calibration_report_labels_distributions - отчёт калибровки разделяет распределения
# END_MODULE_MAP
"""Проверка рабочего вызова наблюдателя: конфигурация, промпт, разбор и выбор модели.

Сетевые вызовы моделей не выполняются: проверяются конфигурация, разбор реальных наблюдений и
выбор модели по измеренной точности. Реестр моделей проверяется мягко: если Ollama недоступна,
тест помечается пропущенным, а не падает.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from packages.common.config import settings
from packages.common.observer import OBSERVER_INSTRUCTION, observation_fields, parse_observation
from services.vlm.app.extract import observe_label

_ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = _ROOT / "config" / "providers.yaml"
PROXY_PAIRS = _ROOT / "reports" / "baseline_pairs.jsonl"
REAL_E4B = ("winery: Скалистый берег\nwine_name: Шепот\nvintage: 2010\n"
            "grape:\nsweetness:\ncolor:\nwine_type:\n"
            "raw_text: СКАЛИСТЫЙ БЕРЕГ ОШ. 2010 ШЁПО НИВЕ ТОВ")

__all__ = ["PROVIDERS", "PROXY_PAIRS", "REAL_E4B", "test_provider_models_exist_in_registry",
           "test_observer_instruction_has_example", "test_parses_english_key_format",
           "test_parses_real_e4b_observation", "test_observe_label_signature",
           "test_observer_model_follows_bench", "test_calibration_report_labels_distributions"]


def _installed_models() -> set[str]:
    """Читает реестр моделей; при недоступности Ollama возвращает пустой набор."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=30)  # noqa: S603, S607
        return {line.split()[0] for line in result.stdout.splitlines()[1:] if line.strip()}
    except Exception:  # noqa: BLE001 - без реестра проверка пропускается
        return set()


def test_provider_models_exist_in_registry() -> None:
    """Страж конфигурации: локальные модели цепочек должны существовать в реестре.

    Именно эта ошибка ломала весь VLM-путь: в цепочке стояли имена без суффикса и вызов отвечал 404.
    """
    installed = _installed_models()
    if not installed:
        pytest.skip("реестр моделей недоступен: проверка пропущена")
    text = PROVIDERS.read_text(encoding="utf-8")
    local = [entry for entry in text.splitlines() if "provider: ollama" in entry]
    missing = [entry.split("model:")[1].strip().rstrip("}") for entry in local
               if entry.split("model:")[1].strip().rstrip("}") not in installed]
    assert missing == [], f"модели отсутствуют в реестре: {missing}"


def test_observer_instruction_has_example() -> None:
    """Инструкция обязана содержать пример вывода: без него слабые модели отвечают мусором."""
    assert "Пример ответа" in OBSERVER_INSTRUCTION
    assert "winery:" in OBSERVER_INSTRUCTION and "raw_text:" in OBSERVER_INSTRUCTION
    assert "только то, что видно" in OBSERVER_INSTRUCTION


def test_parses_english_key_format() -> None:
    observation = parse_observation("winery: Массандра\nwine_name: Мускатель\nvintage: 2023\nsweetness: dry")
    assert observation["winery"] == "Массандра"
    assert observation["wine_name"] == "Мускатель"
    assert observation_fields(observation)["sweetness"] == "сухое"


def test_parses_real_e4b_observation() -> None:
    """Реальное наблюдение слабой модели (получено вживую) разбирается в поля."""
    observation = parse_observation(REAL_E4B)
    fields = observation_fields(observation)
    assert fields["winery"] == "Скалистый берег"
    assert fields["wine_name"] == "Шепот"
    assert fields["vintage"] == "2010"
    assert "СКАЛИСТЫЙ" in observation.get("raw_text", "")


def test_observe_label_signature() -> None:
    """Вызов наблюдателя возвращает словарь наблюдения и не падает без изображений."""
    result = observe_label([], None, model="gemma4:12b-mlx")
    assert isinstance(result, dict)
    assert "ok" in result
    assert result["ok"] is False
    assert not str(result.get("raw_text") or "")  # без изображений наблюдения быть не может


def test_observer_model_follows_bench() -> None:
    """Модель наблюдателя выбрана по бюджету и точности: онлайн-наблюдатель обязан уложиться в 10 с.

    C-063 выбирал модель по точности (gemma4:12b-mlx), но C-064 измерил бюджет: 12b требует около
    минуты и не проходит лимит скрипта организатора, поэтому онлайн-наблюдателем выбран gemma4:e4b-mlx
    (identity@5 0.44, p50 4.7 с), а 12b остаётся для офлайн-разметки.
    """
    assert settings.observer_model == "gemma4:e4b-mlx"
    assert settings.observer_enabled is False  # режим по умолчанию выключен


def test_calibration_report_labels_distributions() -> None:
    """Отчёт калибровки разделяет обучающую (proxy) и отложенную (реальные фото) выборки."""
    report_path = _ROOT / "reports" / "calibration_report.json"
    if not report_path.is_file() or not PROXY_PAIRS.is_file():
        pytest.skip("отчёт калибровки или proxy-пер-пары ещё не собраны")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "fit" in report and "holdout" in report
    assert report["fit"]["pairs"] > 0
