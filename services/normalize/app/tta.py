# START_MODULE_CONTRACT
# PURPOSE: TTA-варианты нормализованного изображения: выпрямленный кроп, паддинг, центр-кроп, полный кадр
# SCOPE: build_variants
# DEPENDS: numpy, cv2, services.normalize.app.homography
# LINKS: M-NORMALIZE
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# build_variants - список уникальных вариантов изображения для TTA
# END_MODULE_MAP
"""TTA-варианты входа.

Несколько кропов/масштабов повышают устойчивость retrieval к ракурсу и обрезке.
Гарантия контракта: список всегда непустой.
"""
from __future__ import annotations

import cv2
import numpy as np

from services.normalize.app.homography import warp_quad

__all__ = ["build_variants"]

MAX_VARIANTS = 4
PAD_RATIO = 0.06


def _fingerprint(image: np.ndarray) -> tuple[int, int, int]:
    small = cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return (int(gray.mean()), int(gray.std()), int(gray[0, 0]))


def build_variants(image: np.ndarray, quad: np.ndarray | None = None) -> list[np.ndarray]:
    """Собирает TTA-варианты изображения.

    Args:
        image: BGR-массив полного кадра.
        quad: четырехугольник этикетки или None (тогда работаем по полному кадру).

    Returns:
        Непустой список уникальных BGR-вариантов (не более MAX_VARIANTS).
    """
    candidates: list[np.ndarray] = []

    if quad is not None:
        warped = warp_quad(image, quad)
        candidates.append(warped)

        height, width = warped.shape[:2]
        pad_y = int(height * PAD_RATIO)
        pad_x = int(width * PAD_RATIO)
        padded = cv2.copyMakeBorder(
            warped, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE
        )
        candidates.append(padded)

        ratio = 0.85
        crop_h, crop_w = int(height * ratio), int(width * ratio)
        top = max(0, (height - crop_h) // 2)
        left = max(0, (width - crop_w) // 2)
        candidates.append(warped[top:top + crop_h, left:left + crop_w])

    candidates.append(image)

    unique: list[np.ndarray] = []
    seen: set[tuple[int, int, int]] = set()
    for candidate in candidates:
        if candidate.size == 0:
            continue
        key = _fingerprint(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
        if len(unique) >= MAX_VARIANTS:
            break

    if not unique:
        unique.append(image)
    return unique
