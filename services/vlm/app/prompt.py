# START_MODULE_CONTRACT
# PURPOSE: Промпт и JSON-схема полей этикетки для VLM-fallback: минимум токенов ради SLA
# SCOPE: LABEL_FIELDS, INSTRUCTION, encode_image_data_url, build_messages
# DEPENDS: cv2, base64
# LINKS: M-VLM
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# LABEL_FIELDS - поля этикетки, извлекаемые VLM
# INSTRUCTION - короткая инструкция со строгой JSON-схемой
# encode_image_data_url - изображение в data URL с ограничением стороны
# build_messages - сообщения OpenAI-формата с картинкой и опциональными кандидатами
# END_MODULE_MAP
"""Промпт VLM-fallback.

Промпт намеренно короткий: обработка prompt-токенов и генерация — основная
стоимость онлайн-пути, а SLA жёсткий. Схема полей фиксирована, ответ требуется
только в JSON (response_format=json_object), temperature=0.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np

__all__ = ["LABEL_FIELDS", "INSTRUCTION", "encode_image_data_url", "build_messages"]

LABEL_FIELDS = ("winery", "wine_name", "vintage", "wine_type", "region", "grape", "confidence")

INSTRUCTION = (
    "Ты читаешь этикетку российского вина. Верни СТРОГО JSON без пояснений:\n"
    '{"winery":"винодельня","wine_name":"название","vintage":"год или пусто",'
    '"wine_type":"белое|красное|розовое|оранжевое","region":"регион",'
    '"grape":"сорт или пусто","confidence":0.0}\n'
    "Неизвестное поле оставь пустой строкой. Только JSON."
)


def encode_image_data_url(image: np.ndarray, max_side: int = 384, quality: int = 80) -> str:
    """Кодирует BGR-изображение в data URL с ограничением длинной стороны.

    Args:
        image: BGR-массив OpenCV.
        max_side: максимальная длина стороны (меньше — быстрее промпт).
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


def build_messages(image: np.ndarray, candidates: list[dict] | None = None,
                   max_side: int = 384) -> list[dict]:
    """Собирает сообщения для VLM.

    Args:
        image: BGR-изображение этикетки.
        candidates: опциональный список кандидатов Top-K (slug и name) — помогает
            модели с near-duplicates, но удлиняет промпт.
        max_side: ограничение стороны изображения.

    Returns:
        Список сообщений OpenAI-формата.
    """
    text = INSTRUCTION
    if candidates:
        hint = "; ".join(
            f"{index + 1}. {item.get('name', '')} ({item.get('winery', '')})"
            for index, item in enumerate(candidates[:5])
        )
        text = f"{INSTRUCTION}\nКандидаты из каталога: {hint}"
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": encode_image_data_url(image, max_side=max_side)}},
        ],
    }]
