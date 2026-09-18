# START_MODULE_CONTRACT
# PURPOSE: Выпрямление перспективы этикетки по четырехугольнику
# SCOPE: order_quad, warp_quad
# DEPENDS: numpy, cv2
# LINKS: M-NORMALIZE
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# order_quad - упорядочивание 4 точек: TL, TR, BR, BL
# warp_quad - перспективное преобразование четырехугольника в прямоугольник
# END_MODULE_MAP
"""Перспективное выпрямление этикетки (гомография)."""
from __future__ import annotations

import cv2
import numpy as np

__all__ = ["order_quad", "warp_quad"]


def order_quad(quad: np.ndarray) -> np.ndarray:
    """Упорядочивает 4 точки как TL, TR, BR, BL."""
    points = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[int(np.argmin(sums))]
    ordered[2] = points[int(np.argmax(sums))]
    ordered[1] = points[int(np.argmin(diffs))]
    ordered[3] = points[int(np.argmax(diffs))]
    return ordered


def warp_quad(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Выпрямляет четырехугольник в прямоугольник."""
    top_left, top_right, bottom_right, bottom_left = order_quad(quad)
    width = int(max(np.linalg.norm(bottom_right - bottom_left),
                    np.linalg.norm(top_right - top_left)))
    height = int(max(np.linalg.norm(top_right - bottom_right),
                     np.linalg.norm(top_left - bottom_left)))
    width = max(width, 1)
    height = max(height, 1)
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(
        np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32),
        destination,
    )
    return cv2.warpPerspective(image, matrix, (width, height))
