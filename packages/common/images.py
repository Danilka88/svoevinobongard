# START_MODULE_CONTRACT
# PURPOSE: Работа с изображениями: EXIF-декодирование, flatten alpha на фон, ресайз, PNG-кодирование
# SCOPE: apply_exif, load_bgr, decode_image, flatten_alpha, resize_max_side, encode_png, to_pil_rgb
# DEPENDS: numpy, cv2, PIL
# LINKS: M-COMMON
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# apply_exif - байты изображения в PIL RGB с примененной EXIF-ориентацией
# load_bgr - байты изображения в BGR ndarray с учетом EXIF
# decode_image - байты изображения в BGR ndarray без применения EXIF
# flatten_alpha - композиция прозрачности на однотонный фон
# resize_max_side - пропорциональное уменьшение по длинной стороне
# encode_png - ndarray в PNG-байты
# to_pil_rgb - ndarray в PIL RGB
# END_MODULE_MAP
"""Утилиты изображений, общие для normalize и retrieval.

Эталонные фото каталога — вырезы на прозрачном фоне; полевые фото — JPEG/PNG.
Alpha схлопывается на нейтральный фон, чтобы домены были сопоставимы.
EXIF-ориентация применяется здесь, чтобы любой сервис декодировал вход одинаково.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image, ImageOps

__all__ = [
    "apply_exif",
    "load_bgr",
    "decode_image",
    "flatten_alpha",
    "resize_max_side",
    "encode_png",
    "to_pil_rgb",
]


def apply_exif(data: bytes) -> Image.Image:
    """Возвращает PIL RGB-изображение с примененной EXIF-ориентацией.

    Raises:
        ValueError: если изображение не декодируется.
    """
    try:
        with Image.open(io.BytesIO(data)) as raw:
            transposed = ImageOps.exif_transpose(raw)
            return transposed.convert("RGB").copy()
    except Exception as error:  # noqa: BLE001 - нормализуем в доменную ошибку
        raise ValueError(f"image could not be decoded: {error}") from error


def load_bgr(data: bytes) -> np.ndarray:
    """Декодирует байты в BGR-массив OpenCV с примененной EXIF-ориентацией."""
    return cv2.cvtColor(np.asarray(apply_exif(data)), cv2.COLOR_RGB2BGR)


def decode_image(data: bytes) -> np.ndarray:
    """Декодирует байты изображения в BGR-массив OpenCV.

    Raises:
        ValueError: если изображение не декодируется.
    """
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("image could not be decoded")
    return image


def flatten_alpha(image: np.ndarray, background: int = 255) -> np.ndarray:
    """Схлопывает альфа-канал на нейтральный фон и возвращает BGR-массив."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 3:
        return image
    if image.shape[2] == 4:
        bgr = image[:, :, :3].astype(np.float32)
        alpha = (image[:, :, 3:4].astype(np.float32)) / 255.0
        background_arr = np.full_like(bgr, float(background))
        composited = bgr * alpha + background_arr * (1.0 - alpha)
        return composited.astype(np.uint8)
    raise ValueError(f"unsupported channel count: {image.shape[2]}")


def resize_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    """Пропорционально уменьшает изображение, если длинная сторона больше max_side."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side or longest == 0:
        return image
    scale = max_side / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def encode_png(image: np.ndarray) -> bytes:
    """Кодирует ndarray в PNG-байты."""
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("image could not be encoded as PNG")
    return buffer.tobytes()


def to_pil_rgb(image: np.ndarray) -> Image.Image:
    """Конвертирует BGR-массив OpenCV в PIL RGB-изображение."""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
