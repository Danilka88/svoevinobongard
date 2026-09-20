# START_MODULE_CONTRACT
# PURPOSE: Верификация палитр и цветового tie-breaker: доминирующие цвета, ΔE, tie-зона, off-путь
# SCOPE: test_dominant_palette_finds_colors, test_delta_e_zero_for_same, test_delta_e_large_for_far, test_similarity_range, test_tiebreak_ignores_outside_zone, test_tiebreak_reorders_inside_zone, test_tiebreak_without_query_palette, test_palettes_roundtrip
# DEPENDS: packages.common.palette, numpy, cv2, pytest
# LINKS: M-RETRIEVAL, V-M-RETRIEVAL
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# RED - близкая к запросу палитра
# BLUE - далёкая от запроса палитра
# test_dominant_palette_finds_colors - палитра содержит оба цвета синтетического кадра
# test_delta_e_zero_for_same - у одинаковых палитр ΔE равно нулю
# test_delta_e_large_for_far - у далёких палитр ΔE большое
# test_similarity_range - близость ограничена диапазоном 0..1
# test_tiebreak_ignores_outside_zone - вне tie-зоны порядок не меняется
# test_tiebreak_reorders_inside_zone - внутри зоны кандидат с близкой палитрой поднимается
# test_tiebreak_without_query_palette - без палитры запроса порядок остаётся исходным
# test_palettes_roundtrip - артефакт палитр сохраняется и читается без потерь
# END_MODULE_MAP
"""V-M-RETRIEVAL: палитры и цветовой tie-breaker."""
from __future__ import annotations

import numpy as np

from packages.common.palette import (dominant_palette, load_palettes, palette_delta_e,
                                     palette_similarity, save_palettes, tiebreak_order)

__all__ = [
    "test_delta_e_large_for_far",
    "test_delta_e_zero_for_same",
    "test_dominant_palette_finds_colors",
    "test_palettes_roundtrip",
    "test_similarity_range",
    "test_tiebreak_ignores_outside_zone",
    "test_tiebreak_reorders_inside_zone",
    "test_tiebreak_without_query_palette",
]

RED = np.array([[50.0, 60.0, 40.0], [55.0, 58.0, 42.0]], dtype=np.float32)
BLUE = np.array([[30.0, -20.0, -40.0], [32.0, -18.0, -38.0]], dtype=np.float32)


def test_dominant_palette_finds_colors() -> None:
    """Палитра содержит оба цвета синтетического кадра."""
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[:, :40] = (0, 0, 220)   # красный в BGR
    image[:, 40:] = (220, 60, 0)  # синий
    palette = dominant_palette(image, k=2)
    assert palette is not None and palette.shape == (2, 3)
    # один из доминирующих цветов должен быть красноватым (a > 0), другой — синеватым (b < 0)
    assert palette[:, 1].max() > 0 > palette[:, 2].min()


def test_delta_e_zero_for_same() -> None:
    """У одинаковых палитр ΔE равно нулю."""
    assert palette_delta_e(RED, RED) == 0.0


def test_delta_e_large_for_far() -> None:
    """У далёких палитр ΔE большое."""
    assert palette_delta_e(RED, BLUE) > 20.0


def test_similarity_range() -> None:
    """Близость палитр лежит в 0..1, совпадение даёт единицу."""
    assert palette_similarity(RED, RED) == 1.0
    assert 0.0 <= palette_similarity(RED, BLUE) <= 1.0


def test_tiebreak_ignores_outside_zone() -> None:
    """Вне tie-зоны порядок не меняется."""
    candidates = [("far-palette", 0.90), ("near-palette", 0.50)]
    order = tiebreak_order(candidates, RED, {"far-palette": BLUE, "near-palette": RED},
                           margin=0.05, weight=0.5)
    assert order == ["far-palette", "near-palette"]


def test_tiebreak_reorders_inside_zone() -> None:
    """Внутри зоны кандидат с близкой палитрой поднимается."""
    candidates = [("far-palette", 0.700), ("near-palette", 0.699)]
    order = tiebreak_order(candidates, RED, {"far-palette": BLUE, "near-palette": RED},
                           margin=0.05, weight=0.5)
    assert order[0] == "near-palette"


def test_tiebreak_without_query_palette() -> None:
    """Без палитры запроса порядок остаётся исходным."""
    candidates = [("a", 0.7), ("b", 0.7)]
    assert tiebreak_order(candidates, None, {}) == ["a", "b"]


def test_palettes_roundtrip(tmp_path) -> None:
    """Артефакт палитр сохраняется и читается без потерь формы."""
    target = tmp_path / "palettes.json"
    save_palettes(target, {"slug-a": RED})
    loaded = load_palettes(target)
    assert loaded["slug-a"].shape == RED.shape
    assert np.allclose(loaded["slug-a"], RED, atol=0.01)
