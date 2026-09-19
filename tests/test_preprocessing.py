# START_MODULE_CONTRACT
# PURPOSE: Верификация неразрушающего препроцессинга: letterbox без искажений, масштабы, P&S-тайлы, enhance, full-frame варианты
# SCOPE: test_letterbox_preserves_aspect, test_letterbox_output_is_square, test_multi_scale_sizes, test_pan_and_scan_tile_count, test_pan_and_scan_covers_frame, test_enhance_changes_contrast, test_build_variants_full_frame_default, test_build_variants_crop_off_by_default, test_build_vlm_inputs_base_plus_tiles
# DEPENDS: numpy, cv2, packages.common.preprocess, services.normalize.app.tta
# LINKS: M-NORMALIZE, M-COMMON
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# test_letterbox_preserves_aspect - letterbox не искажает пропорции: круг остается кругом
# test_letterbox_output_is_square - результат квадратный заданной стороны
# test_multi_scale_sizes - multi-scale отдает варианты 896 и 1280
# test_pan_and_scan_tile_count - число тайлов соответствует запрошенной сетке
# test_pan_and_scan_covers_frame - тайлы покрывают весь кадр, включая края
# test_enhance_changes_contrast - enhance поднимает контраст на темном кадре
# test_build_variants_full_frame_default - по умолчанию первым идет полный кадр, кропа нет
# test_build_variants_crop_off_by_default - при crop=False вариантов с kind crop нет
# test_build_vlm_inputs_base_plus_tiles - VLM-набор = letterbox-база плюс запрошенное число тайлов
# END_MODULE_MAP
"""V-M-NORMALIZE: full-frame препроцессинг (C-012)."""
from __future__ import annotations

import cv2
import numpy as np

from packages.common.preprocess import (
    PreprocessConfig,
    enhance_clahe_wb,
    letterbox_square,
    multi_scale_variants,
    pan_and_scan_tiles,
)
from services.normalize.app.tta import build_variants, build_vlm_inputs

__all__ = [
    "test_letterbox_preserves_aspect",
    "test_letterbox_output_is_square",
    "test_multi_scale_sizes",
    "test_pan_and_scan_tile_count",
    "test_pan_and_scan_covers_frame",
    "test_enhance_changes_contrast",
    "test_build_variants_full_frame_default",
    "test_build_variants_crop_off_by_default",
    "test_build_vlm_inputs_base_plus_tiles",
]


def _wide(seed: int = 0, width: int = 1200, height: int = 400) -> np.ndarray:
    """Широкий кадр с кругом и текстом — для проверки искажений."""
    rng = np.random.default_rng(seed)
    canvas = np.full((height, width, 3), 40, dtype=np.uint8)
    cv2.circle(canvas, (width // 4, height // 2), 100, (240, 240, 240), -1)
    cv2.putText(canvas, "WINE 2021", (width // 2, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                (255, 255, 255), 4)
    canvas = np.clip(canvas.astype(np.int16) + rng.integers(-5, 5, canvas.shape), 0, 255)
    return canvas.astype(np.uint8)


def test_letterbox_preserves_aspect() -> None:
    image = _wide()
    out = letterbox_square(image, size=896)
    assert out.shape[:2] == (896, 896)
    # ширина содержимого больше высоты — пропорции сохранены, появился вертикальный паддинг
    rows = np.where((out != 235).any(axis=(1, 2)))[0]
    cols = np.where((out != 235).any(axis=(0, 2)))[0]
    content_w = cols[-1] - cols[0] + 1
    content_h = rows[-1] - rows[0] + 1
    assert content_w / content_h > 2.0, "letterbox исказил пропорции"


def test_letterbox_output_is_square() -> None:
    for shape in [(300, 900, 3), (900, 300, 3), (500, 500, 3)]:
        out = letterbox_square(np.full(shape, 128, dtype=np.uint8), size=896)
        assert out.shape[:2] == (896, 896)


def test_multi_scale_sizes() -> None:
    variants = multi_scale_variants(_wide(), scales=(896, 1280))
    assert [kind for kind, _ in variants] == ["scale_896", "scale_1280"]
    assert all(image.shape[:2] == (size, size) for (_, image), size in zip(variants, (896, 1280)))


def test_pan_and_scan_tile_count() -> None:
    image = _wide()
    assert len(pan_and_scan_tiles(image, tiles=4)) == 4
    assert len(pan_and_scan_tiles(image, tiles=6)) == 6
    assert all(tile.shape[:2] == (896, 896) for tile in pan_and_scan_tiles(image, tiles=4))


def test_pan_and_scan_covers_frame() -> None:
    """Тайлы нарезаются по всему кадру: левый и правый край присутствуют."""
    image = np.full((400, 1200, 3), 30, dtype=np.uint8)
    image[:, :100] = (0, 0, 255)     # красный у левого края
    image[:, -100:] = (0, 255, 0)    # зелёный у правого края
    tiles = pan_and_scan_tiles(image, tiles=4)
    has_left = any(((tile[:, :, 2] > 200) & (tile[:, :, 1] < 60) & (tile[:, :, 0] < 60)).sum() > 50
                   for tile in tiles)
    has_right = any(((tile[:, :, 1] > 200) & (tile[:, :, 2] < 60) & (tile[:, :, 0] < 60)).sum() > 50
                    for tile in tiles)
    assert has_left and has_right, "P&S не покрыл края кадра"


def test_enhance_changes_contrast() -> None:
    dark = (_wide() * 0.35).astype(np.uint8)
    enhanced = enhance_clahe_wb(dark)
    assert enhanced.shape == dark.shape
    assert enhanced.std() >= dark.std() * 0.9


def test_build_variants_full_frame_default() -> None:
    image = _wide()
    tagged = build_variants(image, None, PreprocessConfig(crop=False), tagged=True)
    kinds = [kind for kind, _ in tagged]
    assert kinds[0] == "full", "полный кадр должен быть базой"
    assert "scale_896" in kinds and "scale_1280" in kinds
    assert tagged[0][1] is image


def test_build_variants_crop_off_by_default() -> None:
    image = _wide()
    tagged = build_variants(image, None, PreprocessConfig(crop=False), tagged=True)
    assert not any(kind.startswith("crop") for kind, _ in tagged)
    # при включённом флаге кроп появляется
    quad = np.array([[0, 0], [image.shape[1] - 1, 0],
                     [image.shape[1] - 1, image.shape[0] - 1], [0, image.shape[0] - 1]], dtype=np.float32)
    with_crop = build_variants(image, quad, PreprocessConfig(crop=True), tagged=True)
    assert any(kind.startswith("crop") for kind, _ in with_crop)


def test_build_vlm_inputs_base_plus_tiles() -> None:
    image = _wide()
    for tiles in (4, 6):
        images, kinds = build_vlm_inputs(image, PreprocessConfig(pas_tiles=tiles))
        assert kinds[0] == "base_letterbox"
        assert len(images) == tiles + 1, f"ожидалось база + {tiles} тайлов"
        assert all(image.shape[:2] == (896, 896) for image in images)
