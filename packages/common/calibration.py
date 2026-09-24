# START_MODULE_CONTRACT
# PURPOSE: Калибровка уверенности поиска в вероятность правильного ответа: обучение, применение, консервативность и порог эскалации
# SCOPE: PARAMS_VERSION, DEFAULT_THRESHOLD, CalibrationParams, sigmoid, apply, fit, fit_conservative,
#        save, load, should_escalate, expected_calibration_error, reliability_table
# DEPENDS: json, math, datetime, pathlib, numpy, packages.common.logging
# LINKS: M-EVAL, M-GATEWAY
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# PARAMS_VERSION - версия формата параметров калибровки
# DEFAULT_THRESHOLD - порог вероятности для эскалации по умолчанию
# MAX_CEILING - предел уверенности, выше которого вероятность не поднимается
# CalibrationParams - параметры монотонного отображения скора в вероятность
# sigmoid - логистическая функция
# apply - скор превращается в вероятность правильного ответа
# fit - обучение параметров по парам «скор, правильность»
# fit_conservative - обучение с поправкой на сдвиг распределения и капом уверенности
# save - сохранение параметров с версией и датой
# load - чтение параметров
# should_escalate - триггер эскалации по вероятности
# expected_calibration_error - метрика ECE по бинам
# reliability_table - таблица надёжности для отчёта
# END_MODULE_MAP
"""Калибровка уверенности.

Сырая уверенность поиска лежит в узком диапазоне (0.21–0.35 на реальных фото) и вероятностью не
является: порог 0.95 по ней отсекает все запросы. Модуль обучает монотонное отображение
«скор → вероятность правильного ответа», проверяет его на отложенной выборке и добавляет
консервативную поправку: при переоценке на другом распределении вероятность поджимается к базовой
частоте и ограничивается сверху, чтобы система не была самоуверенной.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from packages.common.logging import log_marker

PARAMS_VERSION = 1
DEFAULT_THRESHOLD = 0.95
MAX_CEILING = 0.99

__all__ = ["PARAMS_VERSION", "DEFAULT_THRESHOLD", "MAX_CEILING", "CalibrationParams", "sigmoid",
           "apply", "fit", "fit_conservative", "save", "load", "should_escalate",
           "expected_calibration_error", "reliability_table"]


@dataclass
class CalibrationParams:
    """Параметры отображения: вероятность растёт со скором."""

    slope: float = 12.0
    pivot: float = 0.28
    base_rate: float = 0.5
    ceiling: float = MAX_CEILING
    shrink: float = 1.0
    version: int = PARAMS_VERSION
    trained_at: str = ""
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def sigmoid(value: float) -> float:
    """Логистическая функция с защитой от переполнения."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-min(value, 60.0)))
    exponent = math.exp(max(value, -60.0))
    return exponent / (1.0 + exponent)


def apply(params: CalibrationParams, raw: float) -> float:
    """Скор превращается в вероятность правильного ответа.

    Сначала считается логистика от скора, затем применяется консервативная поправка: поджатие к
    базовой частоте и ограничение сверху. Порядок важен — иначе поправка не мешает самоуверенности.
    """
    probability = sigmoid(params.slope * (float(raw) - params.pivot))
    if params.shrink < 1.0:
        probability = probability * params.shrink + params.base_rate * (1.0 - params.shrink)
    return round(min(probability, params.ceiling), 6)


def _log_loss(params: CalibrationParams, pairs: list[tuple[float, int]]) -> float:
    """Средний логарифмический убыток: критерий подбора параметров."""
    total = 0.0
    for raw, correct in pairs:
        probability = min(max(apply(params, raw), 1e-9), 1 - 1e-9)
        total += -(correct * math.log(probability) + (1 - correct) * math.log(1 - probability))
    return total / len(pairs) if pairs else float("inf")


def fit(pairs: list[tuple[float, int]], *, slope_grid: int = 40, pivot_grid: int = 60) -> CalibrationParams:
    """Обучает параметры по парам «скор, правильность» сеточным поиском.

    Сетка, а не градиент: пар мало (десятки), а сетка даёт устойчивый результат без зависимости от
    начальных условий и внешних библиотек.
    """
    if not pairs:
        return CalibrationParams(note="нет данных: отображение по умолчанию")
    scores = [float(raw) for raw, _ in pairs]
    base_rate = float(np.mean([correct for _, correct in pairs]))
    lowest, highest = min(scores), max(scores)
    span = max(highest - lowest, 0.01)
    best = CalibrationParams(base_rate=round(base_rate, 4))
    best_loss = float("inf")
    for slope in np.linspace(4.0, 60.0, slope_grid):
        for pivot in np.linspace(lowest - span * 0.5, highest + span * 0.5, pivot_grid):
            candidate = CalibrationParams(slope=float(slope), pivot=float(pivot),
                                          base_rate=round(base_rate, 4))
            loss = _log_loss(candidate, pairs)
            if loss < best_loss:
                best_loss, best = loss, candidate
    best.trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    best.note = f"обучено на {len(pairs)} парах, log-loss {best_loss:.4f}"
    return best


def expected_calibration_error(probabilities: list[float], correct: list[int], bins: int = 10) -> float:
    """ECE: средневзвешенная разница между заявленной вероятностью и фактической частотой."""
    if not probabilities:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        members = [i for i, value in enumerate(probabilities) if low <= value < high or
                   (index == bins - 1 and value == 1.0)]
        if not members:
            continue
        confidence = float(np.mean([probabilities[i] for i in members]))
        accuracy = float(np.mean([correct[i] for i in members]))
        total += (len(members) / len(probabilities)) * abs(confidence - accuracy)
    return round(total, 4)


def reliability_table(probabilities: list[float], correct: list[int], bins: int = 10) -> list[dict]:
    """Таблица надёжности: по бинам — заявленная вероятность против фактической точности."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict] = []
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        members = [i for i, value in enumerate(probabilities) if low <= value < high or
                   (index == bins - 1 and value == 1.0)]
        if not members:
            continue
        rows.append({"bin": f"{round(low, 2)}–{round(high, 2)}", "count": len(members),
                     "confidence": round(float(np.mean([probabilities[i] for i in members])), 4),
                     "accuracy": round(float(np.mean([correct[i] for i in members])), 4)})
    return rows


def fit_conservative(fit_pairs: list[tuple[float, int]], holdout_pairs: list[tuple[float, int]],
                     *, shrink_step: float = 0.1) -> tuple[CalibrationParams, dict]:
    """Обучает параметры и добавляет поправку, если на отложенной выборке видна переоценка.

    Консервативность: вероятность на отложенном распределении не должна превышать фактическую
    частоту правильных ответов заметно. Если превышает, вероятность поджимается к базовой частоте
    (шагами) и сверху ограничивается наблюдаемой точностью — лучше отправить запрос на проверку,
    чем уверенно ошибиться.
    """
    params = fit(fit_pairs)
    diagnostics: dict = {"fit_pairs": len(fit_pairs), "holdout_pairs": len(holdout_pairs)}
    if not holdout_pairs:
        diagnostics["note"] = "отложенной выборки нет: поправка не применена"
        return params, diagnostics
    raw_probabilities = [apply(params, raw) for raw, _ in holdout_pairs]
    correct = [value for _, value in holdout_pairs]
    diagnostics["holdout_accuracy"] = round(float(np.mean(correct)), 4)
    diagnostics["holdout_mean_p"] = round(float(np.mean(raw_probabilities)), 4)
    diagnostics["holdout_ece_before"] = expected_calibration_error(raw_probabilities, correct)
    overconfidence = diagnostics["holdout_mean_p"] - diagnostics["holdout_accuracy"]
    diagnostics["overconfidence"] = round(overconfidence, 4)
    if overconfidence > 0.05:
        # Поджимаем к базовой частоте, пока средняя заявленная вероятность не перестанет превышать
        # наблюдаемую точность, и ограничиваем сверху этой точностью.
        shrink = 1.0
        while shrink > 0.1:
            shrink = round(shrink - shrink_step, 4)
            candidate = CalibrationParams(slope=params.slope, pivot=params.pivot,
                                          base_rate=params.base_rate, shrink=shrink,
                                          ceiling=max(0.5, diagnostics["holdout_accuracy"] + 0.05))
            mean_p = float(np.mean([apply(candidate, raw) for raw, _ in holdout_pairs]))
            if mean_p <= diagnostics["holdout_accuracy"] + 0.02:
                params = candidate
                break
        else:
            params.shrink = 0.5
            params.ceiling = max(0.5, diagnostics["holdout_accuracy"] + 0.05)
        diagnostics["applied_shrink"] = params.shrink
        diagnostics["applied_ceiling"] = params.ceiling
        params.note = (params.note + f"; поправка на сдвиг: shrink {params.shrink}, "
                       f"потолок {params.ceiling}")
    calibrated = [apply(params, raw) for raw, _ in holdout_pairs]
    diagnostics["holdout_ece_after"] = expected_calibration_error(calibrated, correct)
    diagnostics["reliability_after"] = reliability_table(calibrated, correct, bins=5)
    log_marker("M-EVAL", "fit_conservative", "DONE", ece_before=diagnostics["holdout_ece_before"],
               ece_after=diagnostics["holdout_ece_after"])
    return params, diagnostics


def save(params: CalibrationParams, path: str | Path) -> str:
    """Сохраняет параметры с версией и датой."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": PARAMS_VERSION, "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "params": params.as_dict()}
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(target)


def load(path: str | Path) -> CalibrationParams | None:
    """Читает параметры; при несовпадении версии или отсутствии файла возвращает None."""
    target = Path(path)
    if not target.is_file():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if int(payload.get("version") or 0) != PARAMS_VERSION:
        log_marker("M-GATEWAY", "load_calibration", "VERSION_MISMATCH", version=payload.get("version"))
        return None
    data = payload.get("params") or {}
    known = {field: data[field] for field in CalibrationParams().as_dict() if field in data}
    return CalibrationParams(**known)


def should_escalate(p_correct: float | None, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """Нужно ли отправить запрос на проверку: вероятность ниже порога или неизвестна."""
    if p_correct is None:
        return True
    return float(p_correct) < float(threshold)
