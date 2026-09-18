#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Синтетическая аугментация эталонов в «полевые» условия для proxy-метрик baseline
# SCOPE: random_background, field_augment
# DEPENDS: numpy, cv2
# LINKS: M-EVAL
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# random_background - случайный фон-градиент с шумом (полка, стол, витрина)
# field_augment - эталон в полевое фото: фон, перспектива, блюр, блик, JPEG
# END_MODULE_MAP
"""Простые «полевые» аугментации для proxy-оценки.

Полноценный пайплайн аугментации (Albumentations, очередь ревью) — scope C-006.
Здесь — детерминированный cv2-набор, достаточный для честной proxy-метрики:
фон, перспектива, размытие, блик и JPEG-сжатие.
"""
from __future__ import annotations

import cv2
import numpy as np

__all__ = ["random_background", "field_augment"]

_CANVAS = (640, 480)


def random_background(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    """Генерирует случайный фон: градиент + шум + виньетка."""
    base = rng.integers(30, 210, size=3).astype(np.float32)
    top = np.clip(base + rng.integers(-25, 25, size=3), 10, 245).astype(np.float32)
    gradient = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    canvas = top[None, None, :] * gradient + base[None, None, :] * (1.0 - gradient)
    canvas = np.repeat(canvas, width, axis=1)
    noise = rng.normal(0.0, 6.0, size=(height, width, 3)).astype(np.float32)
    canvas = np.clip(canvas + noise, 0, 255)
    return canvas.astype(np.uint8)


def field_augment(reference_rgba: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Помещает эталон в синтетическое полевое фото.

    Args:
        reference_rgba: эталонное изображение RGBA (вырез на прозрачном фоне).
        rng: генератор случайных чисел для воспроизводимости.

    Returns:
        BGR-изображение «полевого» вида.
    """
    height, width = _CANVAS
    canvas = random_background(rng, height, width).astype(np.float32)

    target_scale = float(rng.uniform(0.45, 0.9))
    label_h = max(32, int(height * target_scale))
    aspect = reference_rgba.shape[1] / max(1, reference_rgba.shape[0])
    label_w = max(32, int(label_h * aspect))

    # этикетка обязана помещаться в холст, иначе roi обрежется и broadcast сломается
    max_w = int(width * 0.92)
    max_h = int(height * 0.92)
    if label_w > max_w:
        scale = max_w / label_w
        label_w, label_h = max(32, int(label_w * scale)), max(32, int(label_h * scale))
    if label_h > max_h:
        scale = max_h / label_h
        label_w, label_h = max(32, int(label_w * scale)), max(32, int(label_h * scale))
    label = cv2.resize(reference_rgba, (label_w, label_h), interpolation=cv2.INTER_AREA)

    rgb = label[:, :, :3].astype(np.float32)
    alpha_map = (label[:, :, 3].astype(np.float32)) / 255.0

    # перспективная деформация этикетки
    jitter = float(rng.uniform(0.0, 0.12))
    src = np.float32([[0, 0], [label_w - 1, 0], [label_w - 1, label_h - 1], [0, label_h - 1]])
    dst = src + rng.normal(0.0, jitter * label_w, size=src.shape).astype(np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    rgb = cv2.warpPerspective(rgb, matrix, (label_w, label_h))
    alpha_map = cv2.warpPerspective(alpha_map, matrix, (label_w, label_h))
    alpha = alpha_map[:, :, None]

    offset_x = int(rng.integers(0, max(1, width - label_w)))
    offset_y = int(rng.integers(0, max(1, height - label_h)))
    roi = canvas[offset_y:offset_y + label_h, offset_x:offset_x + label_w]
    composited = rgb * alpha + roi * (1.0 - alpha)
    canvas[offset_y:offset_y + label_h, offset_x:offset_x + label_w] = composited

    # освещение и блик
    canvas *= float(rng.uniform(0.65, 1.25))
    if rng.random() < 0.7:
        center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        axes = (int(rng.integers(40, 140)), int(rng.integers(40, 140)))
        glare = np.zeros((height, width), dtype=np.float32)
        cv2.ellipse(glare, center, axes, 0, 0, 360, 255.0, -1)
        glare = cv2.GaussianBlur(glare, (0, 0), 25)[:, :, None] / 255.0
        canvas = canvas * (1.0 - 0.45 * glare) + 255.0 * (0.45 * glare)

    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    if rng.random() < 0.6:
        canvas = cv2.GaussianBlur(canvas, (5, 5), float(rng.uniform(0.4, 1.6)))

    quality = int(rng.integers(45, 92))
    ok, encoded = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if ok:
        canvas = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return canvas
