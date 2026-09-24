# START_MODULE_CONTRACT
# PURPOSE: Проверка калибровки уверенности: монотонность, границы, порог эскалации, персист параметров и ECE
# SCOPE: PAIRS, test_probability_is_monotone, test_apply_stays_in_bounds, test_should_escalate_thresholds,
#        test_should_escalate_on_unknown, test_save_load_roundtrip, test_load_rejects_wrong_version,
#        test_fit_improves_on_synthetic, test_conservative_shrinks_when_overconfident,
#        test_ece_detects_overconfidence, test_p_correct_is_none_when_disabled,
#        test_p_correct_filled_when_enabled
# DEPENDS: json, pathlib, pytest, packages.common.calibration, services.gateway.app.pipeline
# LINKS: M-EVAL, M-GATEWAY
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# PAIRS - синтетические пары «скор, правильность»
# test_probability_is_monotone - вероятность растёт со скором
# test_apply_stays_in_bounds - вероятность остаётся в допустимых границах
# test_should_escalate_thresholds - порог эскалации работает на границах
# test_should_escalate_on_unknown - неизвестная вероятность всегда эскалируется
# test_save_load_roundtrip - параметры сохраняются и читаются
# test_load_rejects_wrong_version - чужая версия параметров не применяется
# test_fit_improves_on_synthetic - обучение разделяет правильные и неправильные ответы
# test_conservative_shrinks_when_overconfident - при переоценке включается поправка
# test_ece_detects_overconfidence - ECE видит самоуверенность
# test_p_correct_is_none_when_disabled - при выключенном флаге вероятность не заполняется
# test_p_correct_filled_when_enabled - при включённом флаге вероятность считается
# END_MODULE_MAP
"""Проверка калибровки на синтетике и через настройки: без моделей и сети."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.common.calibration import (
    CalibrationParams,
    apply,
    expected_calibration_error,
    fit,
    fit_conservative,
    load,
    save,
    should_escalate,
)
from packages.common.config import settings
from services.gateway.app import pipeline

__all__ = [
    "PAIRS",
    "test_probability_is_monotone",
    "test_apply_stays_in_bounds",
    "test_should_escalate_thresholds",
    "test_should_escalate_on_unknown",
    "test_save_load_roundtrip",
    "test_load_rejects_wrong_version",
    "test_fit_improves_on_synthetic",
    "test_conservative_shrinks_when_overconfident",
    "test_ece_detects_overconfidence",
    "test_p_correct_is_none_when_disabled",
    "test_p_correct_filled_when_enabled",
]

PAIRS = [(0.20, 0), (0.22, 0), (0.25, 0), (0.30, 1), (0.33, 1), (0.35, 1)]


def test_probability_is_monotone() -> None:
    params = fit(PAIRS)
    values = [apply(params, raw) for raw in (0.10, 0.20, 0.28, 0.35, 0.50)]
    assert values == sorted(values), values


def test_apply_stays_in_bounds() -> None:
    params = CalibrationParams()
    for raw in (-5.0, 0.0, 0.28, 1.0, 5.0):
        value = apply(params, raw)
        assert 0.0 <= value <= params.ceiling <= 0.99, (raw, value)


def test_should_escalate_thresholds() -> None:
    assert should_escalate(0.94, 0.95) is True
    assert should_escalate(0.95, 0.95) is False
    assert should_escalate(0.99, 0.95) is False


def test_should_escalate_on_unknown() -> None:
    """Без калибровки решение неизвестно, поэтому запрос уходит на проверку."""
    assert should_escalate(None) is True


def test_save_load_roundtrip(tmp_path: Path) -> None:
    params = CalibrationParams(slope=10.0, pivot=0.3, note="тест")
    path = tmp_path / "params.json"
    save(params, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] >= 1 and payload["saved_at"]
    restored = load(path)
    assert restored is not None
    assert restored.slope == params.slope and restored.pivot == params.pivot


def test_load_rejects_wrong_version(tmp_path: Path) -> None:
    path = tmp_path / "params.json"
    path.write_text(json.dumps({"version": 999, "params": {"slope": 1.0}}), encoding="utf-8")
    assert load(path) is None
    assert load(tmp_path / "absent.json") is None


def test_fit_improves_on_synthetic() -> None:
    """После обучения вероятность у верных ответов выше, чем у неверных."""
    params = fit(PAIRS)
    assert apply(params, 0.35) > apply(params, 0.20)


def test_conservative_shrinks_when_overconfident() -> None:
    """На отложенной выборке точность ниже заявленной — включается поправка и потолок."""
    params, diagnostics = fit_conservative(PAIRS, [(0.35, 0), (0.33, 0), (0.30, 0)])
    assert params.shrink < 1.0
    assert params.ceiling <= 0.6
    assert diagnostics["overconfidence"] > 0
    assert diagnostics["holdout_ece_before"] >= 0.0


def test_ece_detects_overconfidence() -> None:
    assert expected_calibration_error([0.9, 0.9], [0, 0]) > 0.5  # заявляем 90%, верно 0%
    assert expected_calibration_error([], []) == 0.0


def test_p_correct_is_none_when_disabled() -> None:
    """По умолчанию гейт выключен: поле пустое, поведение прежнее."""
    assert settings.calibration_enabled is False  # фактическое значение по умолчанию
    pipeline._CALIBRATION_CACHE.clear()
    assert pipeline._calibrated_probability(0.30) is None


def test_p_correct_filled_when_enabled(monkeypatch, tmp_path: Path) -> None:
    """При включённом флаге вероятность считается по сохранённым параметрам."""
    path = tmp_path / "params.json"
    save(CalibrationParams(slope=24.1, pivot=0.2889, shrink=0.5, ceiling=0.5), path)
    monkeypatch.setattr(pipeline, "settings",
                        SimpleNamespace(calibration_enabled=True, calibration_path=str(path)))
    pipeline._CALIBRATION_CACHE.clear()
    value = pipeline._calibrated_probability(0.30)
    assert isinstance(value, float) and 0.0 < value <= 0.5
    assert should_escalate(value, 0.95) is True  # при таком потолке гейт всё отправляет на проверку
