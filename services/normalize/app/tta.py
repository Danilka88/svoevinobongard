# START_MODULE_CONTRACT
# PURPOSE: Варианты входа: неразрушающий full-frame набор для энкодера и multi-image набор для VLM
# SCOPE: build_variants, build_vlm_inputs
# DEPENDS: numpy, cv2, packages.common.preprocess
# LINKS: M-NORMALIZE
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# build_variants - варианты для энкодера: полный кадр, letterbox 896 и 1280, кроп только за флагом
# build_vlm_inputs - набор для VLM: letterbox-база плюс pan-and-scan тайлы всего кадра
# END_MODULE_MAP
"""Варианты входа.

Правило проекта: **полный кадр — база**. Обрезка этикетки овальной формы, на
цилиндре или под углом теряет текст и блики, поэтому кроп-варп включается только
флагом ``NORMALIZE_CROP=1``. Вместо обрезки кадр масштабируется без искажения
(letterbox) и нарезается на перекрывающиеся тайлы по всей площади.
"""
from __future__ import annotations

import cv2
import numpy as np

from packages.common.preprocess import (
    DEFAULT_SCALES,
    PreprocessConfig,
    enhance_clahe_wb,
    letterbox_square,
    multi_scale_variants,
    pan_and_scan_tiles,
)

__all__ = ["build_variants", "build_vlm_inputs"]

MAX_VARIANTS = 8
PAD_RATIO = 0.06


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
                   tagged: bool = False) -> list:
    """Собирает варианты входа для энкодера.

    По умолчанию (``config.crop = False``) кадр не обрезается: первым идёт
    полный кадр, затем letterbox-масштабы 896 и 1280.

    Args:
        image: BGR-полный кадр.
        quad: четырехугольник этикетки; используется только при ``config.crop``.
        config: параметры препроцессинга.
        tagged: если True, возвращает пары (kind, изображение) для диагностики.

    Returns:
        Непустой список изображений (или пар kind/изображение).
    """
    active = config or PreprocessConfig()
    candidates: list[tuple[str, np.ndarray]] = []

    if active.crop and quad is not None:
        from services.normalize.app.homography import warp_quad

        warped = warp_quad(image, quad)
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
