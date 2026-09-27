# START_MODULE_CONTRACT
# PURPOSE: Тесты C-089 общего read_bgr: поведение совпадает с прежним декодированием, дублей нет
# SCOPE: test_read_bgr_decodes_rgb_like_old_helper, test_read_bgr_returns_none_on_bad_file,
#        test_read_bgr_handles_alpha, test_no_duplicate_read_image
# DEPENDS: packages.common.images, numpy, PIL
# LINKS: M-COMMON, M-EVAL
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# test_read_bgr_decodes_rgb_like_old_helper - read_bgr даёт тот же BGR, что прежний хелпер
# test_read_bgr_returns_none_on_bad_file - некодируемый файл даёт None
# test_read_bgr_handles_alpha - RGBA-изображение композитится на белый фон
# test_no_duplicate_read_image - в eval-скриптах не осталось локальных _read_image
# END_MODULE_MAP
"""Тесты общего read_bgr C-089."""
from __future__ import annotations

import pathlib

import numpy as np
from PIL import Image

from packages.common.images import flatten_alpha, read_bgr

__all__ = ["test_no_duplicate_read_image", "test_read_bgr_decodes_rgb_like_old_helper",
           "test_read_bgr_handles_alpha", "test_read_bgr_returns_none_on_bad_file"]


def test_read_bgr_decodes_rgb_like_old_helper(tmp_path) -> None:
    """read_bgr даёт тот же BGR-кадр, что прежний локальный хелпер (PIL→RGBA→flatten_alpha→BGR)."""
    array = np.zeros((8, 6, 3), dtype=np.uint8)
    array[:, :, 0] = 255  # красный
    path = tmp_path / "red.png"
    Image.fromarray(array).save(path)

    result = read_bgr(path)
    with Image.open(path) as raw:
        expected = flatten_alpha(np.asarray(raw.convert("RGBA"))[:, :, [2, 1, 0, 3]])
    assert result is not None
    assert np.array_equal(result, expected)


def test_read_bgr_returns_none_on_bad_file(tmp_path) -> None:
    """Некодируемый/отсутствующий файл даёт None, не исключение."""
    assert read_bgr(tmp_path / "missing.png") is None
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not-an-image")
    assert read_bgr(broken) is None


def test_read_bgr_handles_alpha(tmp_path) -> None:
    """RGBA-изображение композитится на фон (альфа схлопывается)."""
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255  # непрозрачный
    path = tmp_path / "rgba.png"
    Image.fromarray(rgba, mode="RGBA").save(path)
    result = read_bgr(path)
    assert result is not None and result.shape == (4, 4, 3)


def test_no_duplicate_read_image() -> None:
    """В eval-скриптах не осталось локальных _read_image (дублирование устранено)."""
    root = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "eval"
    offenders = [p.name for p in root.glob("*.py") if "def _read_image" in p.read_text(encoding="utf-8")]
    assert offenders == []
