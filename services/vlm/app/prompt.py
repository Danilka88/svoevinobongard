# START_MODULE_CONTRACT
# PURPOSE: Промпт и JSON-схема VLM для полного кадра: весь видимый текст этикетки, короткий ответ, multi-image
# SCOPE: LABEL_FIELDS, INSTRUCTION, encode_image_data_url, build_messages
# DEPENDS: cv2, base64, numpy
# LINKS: M-VLM
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# LABEL_FIELDS - поля этикетки, включая alcohol, volume и extra_text
# INSTRUCTION - короткая инструкция «весь видимый текст» со строгой JSON-схемой
# encode_image_data_url - изображение в data URL без искажения пропорций
# build_messages - одно сообщение с несколькими изображениями: letterbox-база и P&S-тайлы
# END_MODULE_MAP
"""Промпт VLM для полного кадра.

Vision-энкодер Gemma работает на 896x896; чтобы не искажать текст, изображения
уже приходят квадратными (letterbox), поэтому здесь только кодирование. Промпт
короткий: длинные инструкции reasoning-модели игнорируют, а токены стоят времени.
Требуем весь видимый текст — любая мелочь (год, объём, крепость) помогает матчингу.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np

__all__ = ["LABEL_FIELDS", "INSTRUCTION", "encode_image_data_url", "build_messages"]

LABEL_FIELDS = ("winery", "wine_name", "vintage", "grape", "sweetness",
                "region", "alcohol", "volume", "extra_text", "confidence")

INSTRUCTION = (
    "Images: full wine label photo (letterboxed) and its zoomed tiles.\n"
    "Read ALL visible label text. Return STRICT JSON only:\n"
    '{"winery":"","wine_name":"","vintage":"","grape":"","sweetness":"",'
    '"region":"","alcohol":"","volume":"","extra_text":[],"confidence":0.0}\n'
    "Rules: copy text as printed; unknown field = empty string; put any other "
    "visible text (medals, awards, address, website) into extra_text. JSON only."
)


def encode_image_data_url(image: np.ndarray, max_side: int = 896, quality: int = 85) -> str:
    """Кодирует BGR-изображение в data URL, не искажая пропорции.

    Args:
        image: BGR-массив (обычно уже квадратный letterbox).
        max_side: максимальная длина стороны.
        quality: качество JPEG.

    Returns:
        Строка ``data:image/jpeg;base64,...``.
    """
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest > max_side:
        scale = max_side / float(longest)
        image = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))),
                           interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("image could not be encoded for VLM")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def build_messages(images: list[np.ndarray], max_side: int = 896) -> list[dict]:
    """Собирает сообщение с несколькими изображениями (база + тайлы) и коротким промптом.

    Args:
        images: список BGR-изображений; первый — letterbox-база полного кадра.
        max_side: ограничение стороны для кодирования.

    Returns:
        Список сообщений OpenAI-формата.
    """
    content: list[dict] = [{"type": "text", "text": INSTRUCTION}]
    for image in images:
        content.append({"type": "image_url",
                        "image_url": {"url": encode_image_data_url(image, max_side=max_side)}})
    return [{"role": "user", "content": content}]
