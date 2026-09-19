# START_MODULE_CONTRACT
# PURPOSE: Верификация OCR-сигнала: парсинг полей, выбор движка, graceful fallback и стабильность на синтетике
# SCOPE: test_parse_fields_year, test_parse_fields_sweetness_with_lookalikes, test_parse_fields_empty, test_resolve_engine_available, test_run_ocr_on_rendered_text, test_run_ocr_degrades_without_engine, test_ocr_result_as_fields, test_roman_to_int_basic, test_roman_to_int_rejects_garbage, test_roman_year_accepts_label_years, test_roman_year_dirty_variants, test_roman_year_range_guard, test_parse_fields_prefers_arabic_and_adds_roman, test_parse_fields_roman_does_not_break_empty
# DEPENDS: packages.common.ocr, numpy, cv2
# LINKS: M-OCR
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# test_parse_fields_year - год извлекается из строк OCR, включая год внутри названия
# test_parse_fields_sweetness_with_lookalikes - сладость находится с учетом латинских подмен кириллицы
# test_parse_fields_empty - пустой ввод не дает ложных полей
# test_resolve_engine_available - движок определяется и является одним из допустимых
# test_run_ocr_on_rendered_text - OCR читает год с синтетической этикетки и укладывается в бюджет
# test_run_ocr_degrades_without_engine - при недоступном движке возвращается ok=false без исключения
# test_ocr_result_as_fields - результат конвертируется в поля, совместимые с matcher
# test_roman_to_int_basic - корректная римская запись переводится в число
# test_roman_to_int_rejects_garbage - некорректная запись не превращается в число
# test_roman_year_accepts_label_years - римские годы этикеток распознаются
# test_roman_year_dirty_variants - грязные OCR-варианты (l, 1, потерянная M, хвостовой Y)
# test_roman_year_range_guard - годы вне диапазона каталога не принимаются
# test_parse_fields_prefers_arabic_and_adds_roman - арабский год приоритетнее, римский добавляется
# test_parse_fields_roman_does_not_break_empty - без года поле остаётся пустым
# END_MODULE_MAP
"""V-M-OCR: OCR-сигнал."""
from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from packages.common.ocr import (
    ENGINE_NAMES,
    OcrResult,
    parse_fields,
    reset_engine,
    resolve_engine,
    run_ocr,
)

__all__ = [
    "test_parse_fields_year",
    "test_parse_fields_sweetness_with_lookalikes",
    "test_parse_fields_empty",
    "test_resolve_engine_available",
    "test_run_ocr_on_rendered_text",
    "test_run_ocr_degrades_without_engine",
    "test_ocr_result_as_fields",
]

_ENGINE = resolve_engine()


def test_parse_fields_year() -> None:
    assert parse_fields(["Aligote Barrel", "2024"])["vintage"] == "2024"
    assert parse_fields(["Riesling 2022"])["vintage"] == "2022"
    assert parse_fields(["no year here"])["vintage"] == ""


def test_parse_fields_sweetness_with_lookalikes() -> None:
    assert parse_fields(["Вино сухое белое"])["sweetness"] == "сухое"
    assert parse_fields(["полусладкое красное"])["sweetness"] == "полусладкое"
    assert parse_fields(["брют"])["sweetness"] == "брют"
    # латинские подмены кириллицы: "cyxoe" -> "сухое"
    assert parse_fields(["CYxoe bеlое"])["sweetness"] == "сухое"


def test_parse_fields_empty() -> None:
    fields = parse_fields([])
    assert fields["vintage"] == "" and fields["sweetness"] == ""


def test_resolve_engine_available() -> None:
    assert resolve_engine() in ENGINE_NAMES + ("none",)


def test_run_ocr_degrades_without_engine() -> None:
    reset_engine()
    result = run_ocr(np.full((64, 64, 3), 255, dtype=np.uint8), engine="none")
    assert result.ok is False
    assert "no OCR engine" in result.reason
    reset_engine()


@pytest.mark.skipif(_ENGINE == "none", reason="no OCR engine installed")
def test_run_ocr_on_rendered_text() -> None:
    canvas = np.full((160, 640, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, "ALIGOTE BARREL 2024", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
    started = time.perf_counter()
    result = run_ocr(canvas, max_side=768)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    assert result.ok is True
    assert result.engine in ENGINE_NAMES
    assert "2024" in result.text.replace(" ", "")
    assert result.vintage == "2024"
    assert elapsed_ms < 5000, f"OCR слишком медленный: {elapsed_ms:.0f}ms"


def test_ocr_result_as_fields() -> None:
    result = OcrResult(ok=True, engine="rapidocr", lines=["Chateau 2019"],
                       vintage="2019", sweetness="сухое", winery_hint="Chateau")
    fields = result.as_fields()
    assert fields["vintage"] == "2019"
    assert fields["wine_type"] == "сухое"
    assert fields["winery"] == "Chateau"


# --- римские годы (C-015) ---
from packages.common.ocr import roman_to_int, roman_year  # noqa: E402


def test_roman_to_int_basic() -> None:
    """Корректная римская запись переводится в число."""
    assert roman_to_int("MMXXII") == 2022
    assert roman_to_int("MMXXIV") == 2024
    assert roman_to_int("MCMXCVIII") == 1998
    assert roman_to_int("XIV") == 14


def test_roman_to_int_rejects_garbage() -> None:
    """Некорректная запись не превращается в число."""
    assert roman_to_int("") == 0
    assert roman_to_int("ABC") == 0
    assert roman_to_int("MMMM") == 0


def test_roman_year_accepts_label_years() -> None:
    """Римские годы этикеток распознаются."""
    assert roman_year("MMXXII") == "2022"
    assert roman_year("урожай MMXXIV года") == "2024"
    assert roman_year("MCMXCVIII") == "1998"


def test_roman_year_dirty_variants() -> None:
    """Грязные OCR-варианты: l/1 вместо I и потерянная M, прочитанная как C."""
    assert roman_year("MMXXIl") == "2022"
    assert roman_year("cMXXIY") == "2022"
    assert roman_year("MMXX1V") == "2024"


def test_roman_year_range_guard() -> None:
    """Годы вне диапазона каталога не принимаются: MMM, XIV, MMXLIX."""
    assert roman_year("MMM") == ""
    assert roman_year("XIV") == ""
    assert roman_year("MMXLIX") == ""


def test_parse_fields_prefers_arabic_and_adds_roman() -> None:
    """Арабский год приоритетнее; римский добавляется, когда арабского нет."""
    arabic = parse_fields(["Chateau 2019", "MMXXII"])
    assert arabic["vintage"] == "2019"
    roman_only = parse_fields(["ПИНО НУАР", "урожай MMXXII", "сухое"])
    assert roman_only["vintage"] == "2022"
    assert roman_only["sweetness"] == "сухое"


def test_parse_fields_roman_does_not_break_empty() -> None:
    """Без года ни арабского, ни римского поле остаётся пустым."""
    assert parse_fields(["ПИНО НУАР", "РОЗОВОЕ БРЮТ"])["vintage"] == ""
