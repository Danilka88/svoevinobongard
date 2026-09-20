# START_MODULE_CONTRACT
# PURPOSE: Неразрушающий препроцессинг: letterbox до квадрата, multi-scale, pan-and-scan тайлы, CLAHE и баланс белого
# SCOPE: letterbox_square, multi_scale_variants, pan_and_scan_tiles, enhance_clahe_wb, PreprocessConfig
# DEPENDS: numpy, cv2
# LINKS: M-COMMON
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# PreprocessConfig - параметры препроцессинга: тайлы, масштабы, enhance, детекция
# letterbox_square - вписывание изображения в квадрат без искажения пропорций
# multi_scale_variants - варианты по длинной стороне (по умолчанию 896 и 1280)
# pan_and_scan_tiles - перекрывающиеся тайлы по всему кадру без знания о положении этикетки
# enhance_clahe_wb - выравнивание контраста (CLAHE) и баланс белого
# END_MODULE_MAP
"""Неразрушающий препроцессинг для vision-энкодеров с фиксированным входом.

Vision-энкодер Gemma работает на 896x896 (256 токенов) и при squash неквадратного
кадра искажает текст этикетки; pan-and-scan в Ollama не реализован. Поэтому кадр
готовится без обрезки: letterbox сохраняет пропорции, multi-scale даёт деталь,
а P&S-тайлы (нарезанные по всему кадру) возвращают мелкий текст.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["PreprocessConfig", "letterbox_square", "multi_scale_variants",
           "pan_and_scan_tiles", "enhance_clahe_wb"]

DEFAULT_SCALES = (896, 1280)
DEFAULT_TILE_GRID = 2  # 2x2 = 4 тайла, 2x3 = 6 тайлов
TILE_OVERLAP = 0.15


@dataclass(frozen=True)
class PreprocessConfig:
    """Параметры подготовки кадра."""

    pas_tiles: int = 4
    enhance: bool = False
    scales: tuple[int, ...] = DEFAULT_SCALES
    crop: bool = False
    pad_color: int = 235


def letterbox_square(image: np.ndarray, size: int = 896, pad_color: int = 235) -> np.ndarray:
    """Вписывает изображение в квадрат без искажения пропорций.

    Args:
        image: BGR-изображение.
        size: сторона квадрата (по умолчанию 896 — вход vision-энкодера Gemma).
        pad_color: значение пикселя паддинга.

    Returns:
        Квадратное BGR-изображение size x size.
    """
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return np.full((size, size, 3), pad_color, dtype=np.uint8)
    scale = size / float(max(height, width))
    resized = cv2.resize(image, (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                         interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), pad_color, dtype=np.uint8)
    top = (size - resized.shape[0]) // 2
    left = (size - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas


def multi_scale_variants(image: np.ndarray, scales: tuple[int, ...] = DEFAULT_SCALES) -> list[tuple[str, np.ndarray]]:
    """Варианты по длинной стороне: 896 (вход энкодера) и 1280 (деталь для текста)."""
    variants: list[tuple[str, np.ndarray]] = []
    for size in scales:
        variants.append((f"scale_{size}", letterbox_square(image, size=size)))
    return variants


def _tile_grid(total_tiles: int) -> tuple[int, int]:
    """Подбирает сетку (строки, столбцы) с произведением, ближайшим к числу тайлов.

    Поддержаны произвольные значения: 1 -> 1x1, 2 -> 1x2, 3 -> 1x3, 4 -> 2x2,
    6 -> 2x3, 5 -> 1x5. Прежние значения (2, 4, 6) дают те же сетки, что и раньше.
    """
    target = max(1, int(total_tiles))
    best_rows, best_cols = 1, target
    # Ключ: сначала близость произведения к цели, затем предпочтение альбомной сетки
    # (столбцов не меньше строк — так кадр режется привычно), затем симметрия.
    best_key = (0, 0 if target >= 1 else 1, abs(target - 1))
    for cols in range(1, 7):
        rows = max(1, -(-target // cols))
        key = (abs(rows * cols - target), 0 if cols >= rows else 1, abs(cols - rows))
        if key < best_key:
            best_key, best_rows, best_cols = key, rows, cols
    return best_rows, best_cols


def pan_and_scan_tiles(image: np.ndarray, tiles: int = 4, overlap: float = TILE_OVERLAP,
                       size: int = 896) -> list[np.ndarray]:
    """Нарезает кадр на перекрывающиеся тайлы по всей площади.

    Тайлы считаются по всему кадру, без знания о положении этикетки: так мелкий
    текст попадает в отдельный тайл и попадает в 896x896 без squash.

    Args:
        image: BGR-изображение полного кадра.
        tiles: желаемое число тайлов (4 обычных, 6 для неуверенного случая).
        overlap: доля перекрытия между тайлами.
        size: сторона квадрата для каждого тайла.

    Returns:
        Список квадратных тайлов длиной примерно `tiles`.
    """
    height, width = image.shape[:2]
    rows, columns = _tile_grid(tiles)
    step_y = height / float(rows)
    step_x = width / float(columns)
    pad_y = step_y * overlap / 2.0
    pad_x = step_x * overlap / 2.0

    out: list[np.ndarray] = []
    for row in range(rows):
        for column in range(columns):
            top = max(0, int(row * step_y - pad_y))
            bottom = min(height, int((row + 1) * step_y + pad_y))
            left = max(0, int(column * step_x - pad_x))
            right = min(width, int((column + 1) * step_x + pad_x))
            patch = image[top:bottom, left:right]
            if patch.size == 0:
                continue
            out.append(letterbox_square(patch, size=size))
    return out


def enhance_clahe_wb(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Выравнивает контраст (CLAHE по L-каналу) и делает простой баланс белого.

    Включается флагом и применяется к P&S-набору: улучшает читаемость текста
    при бликах и неравномерном освещении.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab = cv2.merge((clahe.apply(lightness), channel_a, channel_b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    float_image = enhanced.astype(np.float32)
    means = float_image.reshape(-1, 3).mean(axis=0)
    target = float(means.mean())
    gains = np.clip(target / np.maximum(means, 1.0), 0.7, 1.3)
    balanced = np.clip(float_image * gains, 0, 255).astype(np.uint8)
    return balanced
