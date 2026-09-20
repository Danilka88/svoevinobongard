# START_MODULE_CONTRACT
# PURPOSE: Цилиндрический unwrap этикетки по 6 точкам и детерминированный автопоиск маркеров
# SCOPE: MIN_CONFIDENCE, unwrap_label, find_label_points, unwrap_auto
# DEPENDS: numpy, cv2, packages.common.logging
# LINKS: M-NORMALIZE, M-OCR
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# MIN_CONFIDENCE - минимальная уверенность автопоиска маркеров
# OUT_SIZE - размер плоской этикетки (ширина, высота)
# unwrap_label - криволинейная интерполяция этикетки в плоскость по 6 точкам
# find_label_points - детерминированный поиск 6 маркеров по краям этикетки
# unwrap_auto - unwrap с автопоиском; при неуверенности возвращает исходный кадр
# END_MODULE_MAP
"""Цилиндрический unwrap этикетки.

Этикетка на бутылке-цилиндре сжимается к краям и искажает и сопоставление, и OCR.
Метод 6 точек: три маркера на верхней дуге этикетки и три на нижней задают две кривые;
поверхность между ними интерполируется в плоскую этикетку. Обрезки нет — меняется
геометрия, а не содержимое кадра.

Это упрощение классического метода: кривизна берётся из найденных краёв, а поправка на
длину дуги приближается квадратичной формой кривых (без калибровки камеры). Автопоиск
маркеров детерминированный, без обучения; при низкой уверенности unwrap не применяется,
чтобы «не портить» кадр.
"""
from __future__ import annotations

import cv2
import numpy as np

from packages.common.logging import log_marker

MIN_CONFIDENCE = 0.35
OUT_SIZE = (560, 720)  # (ширина, высота) плоской этикетки
MIN_BOX_AREA_RATIO = 0.10
MAX_CURVATURE_RATIO = 0.35

__all__ = ["MIN_CONFIDENCE", "OUT_SIZE", "find_label_points", "unwrap_auto", "unwrap_label"]


def _quadratic_curve(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                     samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Квадратичная кривая через три точки (параметр u от 0 до 1)."""
    u = np.linspace(0.0, 1.0, samples, dtype=np.float32)
    x = (1 - u) ** 2 * p0[0] + 2 * (1 - u) * u * p1[0] + u ** 2 * p2[0]
    y = (1 - u) ** 2 * p0[1] + 2 * (1 - u) * u * p1[1] + u ** 2 * p2[1]
    return x, y


def unwrap_label(image: np.ndarray, points6: list[tuple[float, float]],
                 out_size: tuple[int, int] = OUT_SIZE) -> np.ndarray:
    """Разворачивает этикетку в плоскость по 6 маркерам.

    Args:
        image: BGR-кадр целиком (обрезка не делается).
        points6: [верх-лево, верх-центр, верх-право, низ-лево, низ-центр, низ-право].
        out_size: размер плоской этикетки (ширина, высота).

    Returns:
        BGR-изображение плоской этикетки; при некорректных точках — исходный кадр.
    """
    if image is None or image.size == 0 or len(points6) != 6:
        return image
    width, height = out_size
    top = [np.asarray(point, dtype=np.float32) for point in points6[:3]]
    bottom = [np.asarray(point, dtype=np.float32) for point in points6[3:]]
    if any(np.isnan(point).any() for point in top + bottom):
        return image
    top_x, top_y = _quadratic_curve(top[0], top[1], top[2], width)
    bottom_x, bottom_y = _quadratic_curve(bottom[0], bottom[1], bottom[2], width)
    v = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    map_x = (top_x[None, :] + (bottom_x - top_x)[None, :] * v).astype(np.float32)
    map_y = (top_y[None, :] + (bottom_y - top_y)[None, :] * v).astype(np.float32)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def find_label_points(image: np.ndarray) -> tuple[list[tuple[float, float]] | None, float]:
    """Ищет 6 маркеров этикетки детерминированными эвристиками.

    Returns:
        (точки, уверенность). Точки равны None, если уверенность ниже MIN_CONFIDENCE:
        вызывающий обязан в этом случае оставить кадр как есть.
    """
    if image is None or image.size == 0:
        return None, 0.0
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    grad_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)).sum(axis=0)
    grad_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)).sum(axis=1)
    if grad_x.size < 8 or grad_y.size < 8:
        return None, 0.0
    # вертикальные края этикетки: два главных пика энергии градиента по столбцам
    left_slice = grad_x[: width // 2]
    right_slice = grad_x[width // 2:]
    x0 = int(np.argmax(left_slice))
    x1 = int(width // 2 + np.argmax(right_slice))
    # Контрастный гейт: без него автопоиск «находил» этикетку даже в однородном шуме.
    # Края этикетки должны быть заметно сильнее фона кадра.
    background = float(np.median(grad_x)) + 1e-6
    edge_strength = float(0.5 * (grad_x[max(0, x0 - 4):x0 + 4].mean() + grad_x[x1 - 4:x1 + 4].mean()))
    if edge_strength < max(3.0 * background, 12.0):
        return None, 0.0
    if x1 - x0 < max(48, int(0.18 * width)):
        return None, 0.0
    band = grad_y[: height // 2]
    band_low = grad_y[height // 2:]
    y0 = int(np.argmax(band))
    y1 = int(height // 2 + np.argmax(band_low))
    if y1 - y0 < max(48, int(0.15 * height)):
        return None, 0.0
    box_width, box_height = x1 - x0, y1 - y0
    area_ratio = (box_width * box_height) / float(width * height)
    if area_ratio < MIN_BOX_AREA_RATIO:
        return None, 0.0
    # кривизна: насколько середина верхней границы выше её краёв (эллиптическая дуга)
    mid = (x0 + x1) // 2
    row_window = slice(max(0, y0 - 8), min(height, y0 + 8))
    column_window = slice(max(0, mid - 8), min(width, mid + 8))
    edge_mid = float(grad_x[column_window].mean())
    edge_side = float(0.5 * (grad_x[max(0, x0 - 4):x0 + 4].mean() + grad_x[x1 - 4:x1 + 4].mean()))
    curvature = max(0.0, min(MAX_CURVATURE_RATIO, 0.5 * (box_height * 0.04))) * box_height
    sagitta = curvature if edge_mid > 0 else 0.0
    # уверенность: доля кадра, сила вертикальных краёв и правдоподобность пропорций
    aspect = box_width / max(box_height, 1)
    edge_ratio = min(1.0, (edge_mid + edge_side) / (grad_x.mean() * 6.0 + 1e-6))
    aspect_score = 1.0 if 0.35 <= aspect <= 3.2 else 0.0
    confidence = round(min(1.0, 0.5 * edge_ratio + 0.3 * aspect_score + 0.2 * min(1.0, area_ratio * 3)), 3)
    if confidence < MIN_CONFIDENCE or not aspect_score:
        return None, confidence
    points = [
        (float(x0), float(y0 + sagitta)), (float(mid), float(y0)), (float(x1), float(y0 + sagitta)),
        (float(x0), float(y1 + sagitta)), (float(mid), float(y1)), (float(x1), float(y1 + sagitta)),
    ]
    return points, confidence


def unwrap_auto(image: np.ndarray, enabled: bool = True) -> tuple[np.ndarray, dict]:
    """Применяет unwrap, если флаг включён и точки найдены уверенно.

    Returns:
        (кадр, сведения). При выключенном флаге или неуверенном автопоиске кадр возвращается
        **без изменений** — испортить результат нельзя.
    """
    info: dict = {"applied": False, "reason": "flag-off", "confidence": 0.0, "latency_ms": 0.0}
    if not enabled:
        return image, info
    points, confidence = find_label_points(image)
    info["confidence"] = confidence
    if points is None:
        info["reason"] = "low-confidence"
        log_marker("M-NORMALIZE", "unwrap", "SKIP", confidence=confidence)
        return image, info
    unwrapped = unwrap_label(image, points)
    if unwrapped is None or unwrapped.size == 0:
        info["reason"] = "empty-result"
        return image, info
    info["applied"] = True
    info["reason"] = "ok"
    log_marker("M-NORMALIZE", "unwrap", "DONE", confidence=confidence)
    return unwrapped, info
