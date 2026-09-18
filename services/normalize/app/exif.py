# START_MODULE_CONTRACT
# PURPOSE: EXIF-ориентация и декодирование входного фото в BGR-массив (делегат в packages.common.images)
# SCOPE: apply_exif, load_image
# DEPENDS: packages.common.images
# LINKS: M-NORMALIZE
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# apply_exif - применение EXIF-ориентации к байтам изображения
# load_image - байты изображения в BGR-массив с учетом EXIF
# END_MODULE_MAP
"""EXIF-нормализация.

ТЗ прямо указывает: работа с нормализацией заметно улучшает F1. Портретное фото
не должно поворачиваться на бок. Реализация вынесена в общий пакет
``packages.common.images``, чтобы retrieval декодировал вход идентично.
"""
from __future__ import annotations

from PIL import Image

from packages.common.images import apply_exif as _apply_exif
from packages.common.images import load_bgr

__all__ = ["apply_exif", "load_image"]


def apply_exif(data: bytes) -> Image.Image:
    """Возвращает PIL RGB-изображение с примененной EXIF-ориентацией."""
    return _apply_exif(data)


def load_image(data: bytes) -> "object":
    """Декодирует байты в BGR-массив OpenCV с примененной EXIF-ориентацией."""
    return load_bgr(data)
