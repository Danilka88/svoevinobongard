# START_MODULE_CONTRACT
# PURPOSE: Верификация движков OCR C-020: слияние двух распознавателей, выбор движка и graceful fallback
# SCOPE: test_fusion_lines_merges_without_duplicates, test_fusion_lines_survives_missing_cyrillic, test_resolve_engine_honours_request, test_resolve_engine_falls_back_when_unavailable, test_run_ocr_vl_falls_back_to_fusion, test_parser_takes_year_from_latin_half, test_config_default_engine_is_rapidocr_without_weights
# DEPENDS: packages.common.ocr, packages.common.ocr_vl, packages.common.config, numpy, pytest
# LINKS: M-OCR, V-M-OCR
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# FRAME - пустой кадр для проверки слияния
# test_fusion_lines_merges_without_duplicates - слияние убирает дубли и сохраняет обе выдачи
# test_fusion_lines_survives_missing_cyrillic - при сбое кириллицы слияние отдаёт латинскую половину
# test_resolve_engine_honours_request - явно запрошенный доступный движок выбирается как есть
# test_resolve_engine_falls_back_when_unavailable - недоступный движок не ломает выбор, берётся доступный
# test_run_ocr_vl_falls_back_to_fusion - сбой VL-модели не валит вызов, выдача уходит в слияние
# test_parser_takes_year_from_latin_half - год берётся из латинской половины слияния
# test_config_default_engine_is_rapidocr_without_weights - без кириллических весов поведение прежнее
# END_MODULE_MAP
"""V-M-OCR: движки OCR и их деградация."""
from __future__ import annotations

import numpy as np
import pytest

from packages.common import ocr as ocr_module
from packages.common.config import settings
from packages.common.ocr import fusion_lines, parse_fields, resolve_engine, run_ocr

__all__ = [
    "test_config_default_engine_is_rapidocr_without_weights",
    "test_fusion_lines_merges_without_duplicates",
    "test_fusion_lines_survives_missing_cyrillic",
    "test_parser_takes_year_from_latin_half",
    "test_resolve_engine_falls_back_when_unavailable",
    "test_resolve_engine_honours_request",
    "test_run_ocr_vl_falls_back_to_fusion",
]

FRAME = np.full((240, 320, 3), 240, dtype=np.uint8)


def test_parser_takes_year_from_latin_half() -> None:
    """Год берётся из латинской половины слияния, кириллица даёт слова."""
    lines = ["KoMMyHaJka", "2022 cyxoe", "Коммуналка", "Рислинг"]
    parsed = parse_fields(lines)
    assert parsed["vintage"] == "2022"
    assert parsed["sweetness"] == "сухое"
    assert any("Коммуналка" == line for line in lines)


def test_fusion_lines_merges_without_duplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Слияние объединяет выдачи двух движков и убирает дубли."""
    monkeypatch.setattr(ocr_module, "_rapid_lines", lambda image, max_side=768: ["2022 cyxoe", "KoMMyHaJka"])
    monkeypatch.setattr(ocr_module, "_cyrillic_lines",
                        lambda image, max_side=768: ["Рислинг", "Рислинг", "Коммуналка"])
    merged = fusion_lines(FRAME)
    assert "2022 cyxoe" in merged and "Коммуналка" in merged
    assert merged.count("Рислинг") == 1


def test_fusion_lines_survives_missing_cyrillic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Если кириллический движок падает, слияние всё равно отдаёт латинскую выдачу."""
    import packages.common.ocr_cyrillic as ocr_cyr

    def boom(image, max_side=768):
        raise RuntimeError("движок недоступен")

    monkeypatch.setattr(ocr_module, "_rapid_lines", lambda image, max_side=768: ["2019 brut"])
    monkeypatch.setattr(ocr_cyr, "run_cyrillic", boom)
    assert fusion_lines(FRAME) == ["2019 brut"]


def test_resolve_engine_honours_request() -> None:
    """Явно запрошенный латинский движок выбирается как есть."""
    before = settings.ocr_engine
    try:
        object.__setattr__(settings, "ocr_engine", "rapidocr")
        assert resolve_engine() == "rapidocr"
    finally:
        object.__setattr__(settings, "ocr_engine", before)


def test_resolve_engine_falls_back_when_unavailable() -> None:
    """Недоступный запрошенный движок не ломает выбор: берётся доступный."""
    before = settings.ocr_engine
    try:
        object.__setattr__(settings, "ocr_engine", "нет-такого-движка")
        assert resolve_engine() in {"rapidocr", "cyrillic", "fusion", "none"}
    finally:
        object.__setattr__(settings, "ocr_engine", before)


def test_run_ocr_vl_falls_back_to_fusion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Сбой VL-модели не валит вызов: выдача деградирует в слияние."""
    import packages.common.ocr_vl as ocr_vl

    monkeypatch.setattr(ocr_vl, "run_vl", lambda image, prompt="OCR:", max_new_tokens=None: ([], 0.0))
    called: list[int] = []
    original = ocr_module.fusion_lines

    def spy(image, max_side=768):
        called.append(1)
        return ["2021 cyxoe"]

    monkeypatch.setattr(ocr_module, "fusion_lines", spy)
    try:
        result = run_ocr(FRAME, engine="paddleocr_vl")
        assert called, "слияние должно быть вызвано как fallback"
        assert result.lines == ["2021 cyxoe"]
    finally:
        monkeypatch.setattr(ocr_module, "fusion_lines", original)


def test_config_default_engine_is_rapidocr_without_weights() -> None:
    """Без кириллических весов и без OCR_ENGINE движок остаётся прежним латинским."""
    before_engine = settings.ocr_engine
    before_model = settings.ocr_rec_model
    try:
        object.__setattr__(settings, "ocr_engine", "")
        object.__setattr__(settings, "ocr_rec_model", "")
        assert resolve_engine() == "rapidocr"
    finally:
        object.__setattr__(settings, "ocr_engine", before_engine)
        object.__setattr__(settings, "ocr_rec_model", before_model)
