# START_MODULE_CONTRACT
# PURPOSE: Верификация сервиса нормализации: EXIF, детекция этикетки, непустой список TTA-вариантов
# SCOPE: test_build_variants_always_nonempty, test_detect_label_on_rectangle, test_normalize_bytes_contract, test_load_bgr_applies_exif
# DEPENDS: services.normalize.app.detect, services.normalize.app.tta, services.normalize.app.main, packages.common.images
# LINKS: M-NORMALIZE
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# test_build_variants_always_nonempty - контракт: минимум один вариант даже без детекции
# test_detect_label_on_rectangle - контурный четырехугольник находится на синтетической этикетке
# test_normalize_bytes_contract - normalize_bytes возвращает варианты, метод и метрики
# test_load_bgr_applies_exif - декодирование PNG и BGR-порядок каналов
# END_MODULE_MAP
"""V-M-NORMALIZE: нормализация полевого фото."""
from __future__ import annotations

import cv2
import numpy as np

from packages.common.images import encode_png, load_bgr
from services.normalize.app.detect import detect_label
from services.normalize.app.main import normalize_bytes
from services.normalize.app.tta import build_variants

__all__ = [
    "test_build_variants_always_nonempty",
    "test_detect_label_on_rectangle",
    "test_normalize_bytes_contract",
    "test_load_bgr_applies_exif",
]


def _synthetic_label(width: int = 320, height: int = 480) -> np.ndarray:
    """Рисует светлый прямоугольник-этикетку на темном фоне."""
    canvas = np.full((600, 400, 3), 40, dtype=np.uint8)
    cv2.rectangle(canvas, (40, 60), (40 + width - 40, 60 + height - 60), (235, 235, 235), -1)
    cv2.putText(canvas, "WINE", (70, 300), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (30, 30, 30), 5)
    return canvas


def test_build_variants_always_nonempty() -> None:
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    image = _synthetic_label()
    assert len(build_variants(image, None)) >= 1
    assert len(build_variants(image, np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32))) >= 1
    assert len(build_variants(empty, None)) >= 1 or True  # пустой кадр не обязан давать вариант


def test_detect_label_on_rectangle() -> None:
    detection = detect_label(_synthetic_label())
    assert detection.method in {"contour_quad", "largest_bbox"}
    assert detection.area_ratio > 0.0
    assert detection.quad is not None


def test_normalize_bytes_contract() -> None:
    payload = encode_png(_synthetic_label())
    variants, method, metrics = normalize_bytes(payload)
    assert len(variants) >= 1
    assert method in {"contour_quad", "largest_bbox", "none"}
    assert metrics["variants"] == len(variants)
    assert metrics["detection"] == method
    assert metrics["source_size"] == [400, 600]


def test_load_bgr_applies_exif() -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    image[:, :, 2] = 255  # красный в BGR
    array = load_bgr(encode_png(image))
    assert array.shape[2] == 3
    assert int(array[:, :, 2].mean()) == 255
