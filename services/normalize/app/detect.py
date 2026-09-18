# START_MODULE_CONTRACT
# PURPOSE: Детекция этикетки/бутылки — OpenCV-эвристика (контуры, наибольший четырехугольник) как fallback
# SCOPE: Detection, detect_label, quad_area_ratio
# DEPENDS: numpy, cv2
# LINKS: M-NORMALIZE
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# Detection - результат детекции: quad, bbox, method, area_ratio
# detect_label - поиск четырехугольника этикетки или наибольшего контура
# quad_area_ratio - доля площади четырехугольника от площади кадра
# END_MODULE_MAP
"""Детекция этикетки.

Реализована детерминированная OpenCV-эвристика: поиск наибольшего выпуклого
четырехугольника достаточной площади. Нейросетевой детектор (D-FINE / RT-DETRv4,
Apache-2.0) подключается отдельным change и заменяет эвристику, сохраняя контракт
``detect_label``.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["Detection", "detect_label", "quad_area_ratio"]

MIN_AREA_RATIO = 0.05
MAX_AREA_RATIO = 0.98


@dataclass(frozen=True)
class Detection:
    """Результат детекции.

    Attributes:
        quad: 4 точки этикетки (float32, 4x2) или None.
        bbox: ограничивающий прямоугольник (x, y, w, h) или None.
        method: contour_quad | largest_bbox | none.
        area_ratio: доля площади от кадра.
    """

    quad: np.ndarray | None
    bbox: tuple[int, int, int, int] | None
    method: str
    area_ratio: float


def quad_area_ratio(quad: np.ndarray, image_shape: tuple[int, ...]) -> float:
    """Считает долю площади четырехугольника от площади кадра."""
    total = float(image_shape[0] * image_shape[1])
    if total <= 0:
        return 0.0
    return float(cv2.contourArea(quad.astype(np.float32))) / total


def _candidate_quad(contour: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray | None:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approx) != 4 or not cv2.isContourConvex(approx):
        return None
    quad = approx.reshape(4, 2).astype(np.float32)
    ratio = quad_area_ratio(quad, image_shape)
    if ratio < MIN_AREA_RATIO or ratio > MAX_AREA_RATIO:
        return None
    return quad


def detect_label(image: np.ndarray) -> Detection:
    """Ищет этикетку на изображении.

    Сначала перебирает контуры в поиске выпуклого четырехугольника достаточной
    площади (этикетка под углом), затем падает до наибольшего внешнего контура.

    Args:
        image: BGR-массив.

    Returns:
        Detection с заполненным method: contour_quad, largest_bbox или none.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    quad_candidates: list[tuple[float, np.ndarray]] = []
    bbox_candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    total = float(image.shape[0] * image.shape[1])

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if total <= 0 or area < MIN_AREA_RATIO * total:
            continue
        quad = _candidate_quad(contour, image.shape)
        if quad is not None:
            quad_candidates.append((area, quad))
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if (w * h) / total <= MAX_AREA_RATIO:
            bbox_candidates.append((area, (x, y, w, h)))

    if quad_candidates:
        area, quad = max(quad_candidates, key=lambda item: item[0])
        x, y, w, h = cv2.boundingRect(quad.astype(np.int32))
        return Detection(quad=quad, bbox=(x, y, w, h), method="contour_quad",
                         area_ratio=quad_area_ratio(quad, image.shape))

    if bbox_candidates:
        area, bbox = max(bbox_candidates, key=lambda item: item[0])
        x, y, w, h = bbox
        quad = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
        return Detection(quad=quad, bbox=bbox, method="largest_bbox", area_ratio=area / total)

    return Detection(quad=None, bbox=None, method="none", area_ratio=0.0)
