# START_MODULE_CONTRACT
# PURPOSE: Верификация C-066: кириллический OCR как off-flag — выбор движка, graceful fallback,
#          чтение года и сладости из кириллицы и неизменность поведения при выключенном флаге
# SCOPE: FLAG_NAMES, _engine_flag, test_default_engine_is_rapidocr_when_flag_off,
#        test_cyrillic_selected_when_requested_and_available, test_fusion_selected_when_available,
#        test_cyrillic_falls_back_when_unavailable, test_resolve_cyrillic_none_when_files_missing,
#        test_parse_fields_reads_cyrillic_year_and_sweetness, test_parse_fields_reads_cyrillic_roman_year,
#        test_run_ocr_cyrillic_without_engine_is_graceful, test_run_ocr_cyrillic_error_does_not_raise
# DEPENDS: contextlib, numpy, pytest, packages.common.config, packages.common.ocr, packages.common.ocr_cyrillic
# LINKS: M-OCR, V-M-OCR
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# FLAG_NAMES - имена флагов движка, которые тесты временно переопределяют
# FRAME - пустой кадр для вызовов run_ocr без движка
# test_default_engine_is_rapidocr_when_flag_off - при пустом флаге движок прежний, кириллица не выбирается
# test_cyrillic_selected_when_requested_and_available - явный запрос кириллицы при доступных весах выбирает её
# test_fusion_selected_when_available - запрос слияния при доступной кириллице выбирает fusion
# test_cyrillic_falls_back_when_unavailable - недоступная кириллица не ломает выбор: берётся доступный движок
# test_resolve_cyrillic_none_when_files_missing - без весов и словаря движок честно недоступен, без исключения
# test_parse_fields_reads_cyrillic_year_and_sweetness - год и сладость читаются из кириллических строк
# test_parse_fields_reads_cyrillic_roman_year - римский год из кириллической выдачи распознаётся
# test_run_ocr_cyrillic_without_engine_is_graceful - пустая выдача кириллицы даёт ok без исключения
# test_run_ocr_cyrillic_error_does_not_raise - сбой кириллического движка возвращает ok=false, не падая
# END_MODULE_MAP
"""V-M-OCR: кириллический движок за флагом (C-066).

Тесты не требуют vendored-весов PP-OCRv5: доступность движка подменяется на уровне
`resolve_cyrillic`/`run_cyrillic`, поэтому проверка работает и в чистом окружении.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pytest

import packages.common.ocr_cyrillic as ocr_cyr
from packages.common.config import settings
from packages.common.ocr import parse_fields, resolve_engine, run_ocr

__all__ = [
    "FLAG_NAMES",
    "FRAME",
    "test_cyrillic_falls_back_when_unavailable",
    "test_cyrillic_selected_when_requested_and_available",
    "test_default_engine_is_rapidocr_when_flag_off",
    "test_fusion_selected_when_available",
    "test_parse_fields_reads_cyrillic_roman_year",
    "test_parse_fields_reads_cyrillic_year_and_sweetness",
    "test_resolve_cyrillic_none_when_files_missing",
    "test_run_ocr_cyrillic_error_does_not_raise",
    "test_run_ocr_cyrillic_without_engine_is_graceful",
]

FLAG_NAMES = ("ocr_engine", "ocr_rec_model", "ocr_rec_keys")

FRAME = np.full((240, 320, 3), 240, dtype=np.uint8)


@contextmanager
def _engine_flag(**values):
    """Временно выставляет поля настроек движка и возвращает прежние значения."""
    previous = {name: getattr(settings, name) for name in FLAG_NAMES}
    try:
        for name, value in values.items():
            object.__setattr__(settings, name, value)
        yield
    finally:
        for name, value in previous.items():
            object.__setattr__(settings, name, value)


def test_default_engine_is_rapidocr_when_flag_off() -> None:
    """Пустой флаг: движок прежний (RapidOCR), кириллица не выбирается сама по себе."""
    with _engine_flag(ocr_engine=""):
        assert resolve_engine() == "rapidocr"


def test_cyrillic_selected_when_requested_and_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Явный запрос кириллицы при доступных весах выбирает её."""
    monkeypatch.setattr(ocr_cyr, "resolve_cyrillic", lambda: ("model.onnx", "dict.txt"))
    with _engine_flag(ocr_engine="cyrillic"):
        assert resolve_engine() == "cyrillic"


def test_fusion_selected_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Запрос слияния при доступной кириллице выбирает fusion, а не откатывается молча."""
    monkeypatch.setattr(ocr_cyr, "resolve_cyrillic", lambda: ("model.onnx", "dict.txt"))
    with _engine_flag(ocr_engine="fusion"):
        assert resolve_engine() == "fusion"


def test_cyrillic_falls_back_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Недоступная кириллица не ломает выбор: возвращается доступный движок."""
    monkeypatch.setattr(ocr_cyr, "resolve_cyrillic", lambda: None)
    with _engine_flag(ocr_engine="cyrillic"):
        assert resolve_engine() in {"rapidocr", "easyocr", "tesseract", "none"}


def test_resolve_cyrillic_none_when_files_missing() -> None:
    """Без весов и словаря движок честно недоступен, без исключения."""
    with _engine_flag(ocr_rec_model="/нет/такого.onnx", ocr_rec_keys="/нет/такого.txt"):
        assert ocr_cyr.resolve_cyrillic() is None


def test_parse_fields_reads_cyrillic_year_and_sweetness() -> None:
    """Год и сладость читаются из кириллических строк без латинской подстановки."""
    parsed = parse_fields(["ВИНОДЕЛЬНЯ", "Коммуналка", "2021 полусухое"])
    assert parsed["vintage"] == "2021"
    assert parsed["sweetness"] == "полусухое"


def test_parse_fields_reads_cyrillic_roman_year() -> None:
    """Римский год из кириллической выдачи распознаётся тем же разбором, что и раньше."""
    parsed = parse_fields(["Коммуналка", "MMXXII", "сухое"])
    assert parsed["vintage"] == "2022"
    assert parsed["sweetness"] == "сухое"


def test_run_ocr_cyrillic_without_engine_is_graceful(monkeypatch: pytest.MonkeyPatch) -> None:
    """Пустая выдача кириллицы (нет весов) даёт ok без исключения и без ложных полей."""
    monkeypatch.setattr(ocr_cyr, "run_cyrillic", lambda image, max_side=768: [])
    result = run_ocr(FRAME, engine="cyrillic")
    assert result.ok is True
    assert result.lines == []
    assert result.vintage == "" and result.sweetness == ""


def test_run_ocr_cyrillic_error_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Сбой кириллического движка возвращает ok=false с причиной, не валя пайплайн."""

    def boom(image, max_side=768):
        raise RuntimeError("движок недоступен")

    monkeypatch.setattr(ocr_cyr, "run_cyrillic", boom)
    result = run_ocr(FRAME, engine="cyrillic")
    assert result.ok is False
    assert "RuntimeError" in result.reason
