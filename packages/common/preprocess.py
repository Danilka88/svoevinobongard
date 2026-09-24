# START_MODULE_CONTRACT
# PURPOSE: Общий неразрушающий препроцессинг: letterbox, multi-scale, pan-and-scan тайлы, CLAHE и готовые варианты входа для энкодера и VLM
# SCOPE: PreprocessConfig, letterbox_square, multi_scale_variants, pan_and_scan_tiles, enhance_clahe_wb, build_variants, build_vlm_inputs
# DEPENDS: numpy, cv2
# LINKS: M-COMMON, M-NORMALIZE, M-VLM
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# PreprocessConfig - параметры препроцессинга: тайлы, масштабы, enhance, детекция
# letterbox_square - вписывание изображения в квадрат без искажения пропорций
# multi_scale_variants - варианты по длинной стороне (по умолчанию 896 и 1280)
# pan_and_scan_tiles - перекрывающиеся тайлы по всему кадру без знания о положении этикетки
# enhance_clahe_wb - выравнивание контраста (CLAHE) и баланс белого
# build_variants - варианты для энкодера: полный кадр, letterbox 896 и 1280, кроп только за флагом
# build_vlm_inputs - набор для VLM: letterbox-база плюс pan-and-scan тайлы всего кадра
# END_MODULE_MAP
"""Неразрушающий препроцессинг для vision-энкодеров с фиксированным входом.

Vision-энкодер Gemma работает на 896x896 (256 токенов) и при squash неквадратного
кадра искажает текст этикетки; pan-and-scan в Ollama не реализован. Поэтому кадр
готовится без обрезки: letterbox сохраняет пропорции, multi-scale даёт деталь,
а P&S-тайлы (нарезанные по всему кадру) возвращают мелкий текст.

Здесь же живут готовые наборы входа (build_variants, build_vlm_inputs), общие для
normalize и VLM: раньше они лежали в services/normalize.app.tta, из-за чего VLM
импортировал чужой сервис. Теперь общий код — в этой библиотеке, а перспективное
выпрямление кропа передаётся инъекцией (warper), чтобы packages не зависел от services.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

__all__ = ["PreprocessConfig", "letterbox_square", "multi_scale_variants",
           "pan_and_scan_tiles", "enhance_clahe_wb", "build_variants", "build_vlm_inputs"]

DEFAULT_SCALES = (896, 1280)
DEFAULT_TILE_GRID = 2  # 2x2 = 4 тайла, 2x3 = 6 тайлов
TILE_OVERLAP = 0.15
MAX_VARIANTS = 8
PAD_RATIO = 0.06


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


def _fingerprint(image: np.ndarray) -> tuple[int, int, int]:
    """Компактный отпечаток кадра для дедупликации вариантов."""
    if image.size == 0:
        return (0, 0, 0)
    small = cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return (int(gray.mean()), int(gray.std()), int(gray[0, 0]))


def _dedupe(candidates: list[tuple[str, np.ndarray]], limit: int) -> list[tuple[str, np.ndarray]]:
    """Убирает дубликаты по отпечатку и ограничивает число вариантов."""
    out: list[tuple[str, np.ndarray]] = []
    seen: set[tuple[int, int, int]] = set()
    for kind, variant in candidates:
        if variant.size == 0:
            continue
        key = _fingerprint(variant)
        if key in seen:
            continue
        seen.add(key)
        out.append((kind, variant))
        if len(out) >= limit:
            break
    return out


def build_variants(image: np.ndarray, quad: np.ndarray | None = None,
                   config: PreprocessConfig | None = None,
                   tagged: bool = False,
                   warper: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None) -> list:
    """Собирает варианты входа для энкодера.

    По умолчанию (``config.crop = False``) кадр не обрезается: первым идёт
    полный кадр, затем letterbox-масштабы 896 и 1280.

    Args:
        image: BGR-полный кадр.
        quad: четырехугольник этикетки; используется только при ``config.crop``.
        config: параметры препроцессинга.
        tagged: если True, возвращает пары (kind, изображение) для диагностики.
        warper: перспективное выпрямление четырехугольника; нужен только при
            ``config.crop`` (передаётся сервисом нормализации, чтобы библиотека
            не зависела от services).

    Returns:
        Непустой список изображений (или пар kind/изображение).
    """
    active = config or PreprocessConfig()
    candidates: list[tuple[str, np.ndarray]] = []

    if active.crop and quad is not None and warper is not None:
        warped = warper(image, quad)
        candidates.append(("crop", warped))
        height, width = warped.shape[:2]
        pad_y, pad_x = int(height * PAD_RATIO), int(width * PAD_RATIO)
        candidates.append(("crop_padded", cv2.copyMakeBorder(
            warped, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)))
        ratio = 0.85
        crop_h, crop_w = int(height * ratio), int(width * ratio)
        top, left = max(0, (height - crop_h) // 2), max(0, (width - crop_w) // 2)
        candidates.append(("crop_center", warped[top:top + crop_h, left:left + crop_w]))

    variants: list[tuple[str, np.ndarray]] = [("full", image)]
    if active.enhance:
        variants.append(("full_enhanced", enhance_clahe_wb(image)))
    base = enhance_clahe_wb(image) if active.enhance else image
    variants.extend(multi_scale_variants(base, scales=active.scales or DEFAULT_SCALES))

    unique = _dedupe(candidates + variants, MAX_VARIANTS)
    if not unique:
        unique = [("full", image)]
    if tagged:
        return unique
    return [variant for _, variant in unique]


def build_vlm_inputs(image: np.ndarray, config: PreprocessConfig | None = None,
                     base_size: int = 896) -> tuple[list[np.ndarray], list[str]]:
    """Собирает набор изображений для VLM: letterbox-база плюс P&S-тайлы.

    Правило: этикетку не обрезаем. База даёт общий вид кадра, тайлы — мелкий
    текст; нарезка идёт по всему кадру, без знания о положении этикетки.

    Args:
        image: BGR-полный кадр.
        config: параметры (число тайлов, enhance).
        base_size: сторона квадрата (896 — вход vision-энкодера).

    Returns:
        Кортеж (список изображений, список их kind-меток).
    """
    active = config or PreprocessConfig()
    source = enhance_clahe_wb(image) if active.enhance else image
    images: list[np.ndarray] = [letterbox_square(source, size=base_size)]
    kinds: list[str] = ["base_letterbox"]
    for index, tile in enumerate(pan_and_scan_tiles(source, tiles=active.pas_tiles,
                                                    size=base_size), start=1):
        images.append(tile)
        kinds.append(f"tile_{index}")
    return images, kinds
