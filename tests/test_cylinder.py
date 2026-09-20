# START_MODULE_CONTRACT
# PURPOSE: Верификация цилиндрического unwrap: геометрия по 6 точкам, отказ автопоиска, неизменность кадра
# SCOPE: test_unwrap_identity_points_keeps_frame, test_unwrap_output_shape, test_unwrap_bad_points_returns_source, test_find_points_none_on_noise, test_find_points_on_synthetic_label, test_unwrap_auto_off_returns_source, test_unwrap_auto_refuses_low_confidence
# DEPENDS: packages.common.cylinder, numpy, cv2, pytest
# LINKS: M-NORMALIZE, V-M-NORMALIZE
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# CORNERS - шесть маркеров синтетической области
# test_unwrap_identity_points_keeps_frame - при точках по углам кадра unwrap сохраняет размер и содержимое
# test_unwrap_output_shape - результат имеет запрошенный размер плоской этикетки
# test_unwrap_bad_points_returns_source - некорректные точки возвращают исходный кадр
# test_find_points_none_on_noise - на однородном шуме автопоиск отказывается
# test_find_points_on_synthetic_label - на синтетической этикетке точки находятся
# test_unwrap_auto_off_returns_source - при выключенном флаге кадр не меняется
# test_unwrap_auto_refuses_low_confidence - при неуверенном автопоиске возвращается исходный кадр
# END_MODULE_MAP
"""V-M-NORMALIZE: цилиндрический unwrap."""
from __future__ import annotations

import cv2
import numpy as np

from packages.common.cylinder import find_label_points, unwrap_auto, unwrap_label

__all__ = [
    "test_find_points_none_on_noise",
    "test_find_points_on_synthetic_label",
    "test_unwrap_auto_off_returns_source",
    "test_unwrap_auto_refuses_low_confidence",
    "test_unwrap_bad_points_returns_source",
    "test_unwrap_identity_points_keeps_frame",
    "test_unwrap_output_shape",
]

CORNERS = [(0.0, 0.0), (50.0, 0.0), (100.0, 0.0), (0.0, 80.0), (50.0, 80.0), (100.0, 80.0)]


def _label_frame() -> np.ndarray:
    """Синтетический кадр: белая этикетка на тёмном фоне."""
    frame = np.full((400, 400, 3), 30, dtype=np.uint8)
    cv2.rectangle(frame, (100, 120), (300, 300), (240, 240, 240), -1)
    cv2.putText(frame, "VINO", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
    return frame


def test_unwrap_identity_points_keeps_frame() -> None:
    """При точках по углам области unwrap сохраняет содержимое кадра."""
    frame = _label_frame()
    unwrapped = unwrap_label(frame, CORNERS, out_size=(100, 80))
    assert unwrapped.shape[:2] == (80, 100)
    assert float(unwrapped.mean()) > 0.0


def test_unwrap_output_shape() -> None:
    """Результат имеет запрошенный размер плоской этикетки."""
    assert unwrap_label(_label_frame(), CORNERS, out_size=(320, 240)).shape[:2] == (240, 320)


def test_unwrap_bad_points_returns_source() -> None:
    """Некорректное число точек возвращает исходный кадр без изменений."""
    frame = _label_frame()
    assert unwrap_label(frame, CORNERS[:4]) is frame


def test_find_points_none_on_noise() -> None:
    """На однородном шуме автопоиск отказывается от результата."""
    noise = np.random.default_rng(0).integers(0, 40, size=(300, 300, 3), dtype=np.uint8)
    points, confidence = find_label_points(noise)
    assert points is None, "на шуме маркеры найдены быть не должны"


def test_find_points_on_synthetic_label() -> None:
    """На синтетической этикетке маркеры находятся с приемлемой уверенностью."""
    points, confidence = find_label_points(_label_frame())
    assert points is not None and len(points) == 6
    assert confidence >= 0.35


def test_unwrap_auto_off_returns_source() -> None:
    """При выключенном флаге кадр не меняется."""
    frame = _label_frame()
    result, info = unwrap_auto(frame, enabled=False)
    assert result is frame and info["applied"] is False and info["reason"] == "flag-off"


def test_unwrap_auto_refuses_low_confidence() -> None:
    """При неуверенном автопоиске возвращается исходный кадр."""
    noise = np.random.default_rng(1).integers(0, 40, size=(300, 300, 3), dtype=np.uint8)
    result, info = unwrap_auto(noise, enabled=True)
    assert result is noise and info["applied"] is False
