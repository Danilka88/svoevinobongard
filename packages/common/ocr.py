# START_MODULE_CONTRACT
# PURPOSE: Дешёвый OCR-сигнал: текст с кропа этикетки и нормализованные поля (год, сладость) с graceful fallback движка
# SCOPE: OcrResult, resolve_engine, reset_engine, run_ocr, parse_fields
# DEPENDS: numpy, cv2, packages.common.logging
# LINKS: M-OCR
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# OcrResult - результат OCR: строки, текст, год, сладость, движок, latency
# resolve_engine - выбор доступного OCR-движка (rapidocr, easyocr, tesseract или none)
# reset_engine - сброс кэша движка (тесты, смена конфигурации)
# run_ocr - распознавание текста и нормализация полей на кропе этикетки
# parse_fields - извлечение года и сладости из строк OCR
# END_MODULE_MAP
"""OCR-сигнал для дизамбигуации near-duplicates.

Заменяет локальный VLM, который не читает этикетки (C-004): OCR стоит сотни
миллисекунд, детерминирован и не требует генерации. Основной движок —
RapidOCR (Apache-2.0, ONNXRuntime, без paddle). Кириллица у стандартной
модели читается как латиница, поэтому год (цифры) распознается надежно, а
сладость — best-effort: дополнительно применяется обратная замена
латиница→кириллица по визуально похожим символам.
"""
from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field

import cv2
import numpy as np

from packages.common.logging import log_marker

__all__ = ["OcrResult", "resolve_engine", "reset_engine", "run_ocr", "parse_fields"]

ENGINE_NAMES = ("rapidocr", "easyocr", "tesseract")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
SWEETNESS = ("экстра брют", "брют натюр", "брют", "полусладкое", "полусухое", "сладкое", "сухое")

# Визуально похожие латинские символы, которыми OCR подменяет кириллицу.
_LOOKALIKE = str.maketrans({
    "c": "с", "y": "у", "x": "х", "o": "о", "e": "е", "a": "а", "p": "р",
    "h": "н", "b": "в", "m": "м", "t": "т", "k": "к", "3": "з", "n": "п",
})

_engine_cache: dict[str, object] = {}


@dataclass
class OcrResult:
    """Результат OCR-прогона."""

    ok: bool = False
    engine: str = "none"
    lines: list[str] = field(default_factory=list)
    vintage: str = ""
    sweetness: str = ""
    winery_hint: str = ""
    latency_ms: float = 0.0
    reason: str = ""

    @property
    def text(self) -> str:
        """Все распознанные строки, склеенные пробелом."""
        return " ".join(self.lines)

    def as_fields(self) -> dict:
        """Поля в формате matcher (совместимо с полями VLM)."""
        fields = {"winery": self.winery_hint, "vintage": self.vintage}
        if self.sweetness:
            fields["wine_type"] = self.sweetness
        return fields


def resolve_engine() -> str:
    """Определяет доступный OCR-движок.

    Returns:
        rapidocr | easyocr | tesseract | none.
    """
    try:
        from packages.common.ocr_cyrillic import resolve_cyrillic  # noqa: PLC0415

        if resolve_cyrillic() is not None:
            return "cyrillic"
    except Exception:  # noqa: BLE001 - кириллический движок опционален
        pass
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return "rapidocr"
    except Exception:  # noqa: BLE001
        pass
    try:
        import easyocr  # noqa: F401
        return "easyocr"
    except Exception:  # noqa: BLE001
        pass
    try:
        import shutil

        import pytesseract  # noqa: F401

        if shutil.which("tesseract"):
            return "tesseract"
    except Exception:  # noqa: BLE001
        pass
    return "none"


def reset_engine() -> None:
    """Сбрасывает кэш движка."""
    _engine_cache.clear()


def _get_engine(name: str):
    """Ленивая инициализация движка (создание модели дорогое)."""
    if name in _engine_cache:
        return _engine_cache[name]
    if name == "rapidocr":
        from rapidocr_onnxruntime import RapidOCR

        engine = RapidOCR()
    elif name == "easyocr":
        import easyocr

        engine = easyocr.Reader(["ru", "en"], gpu=False, verbose=False)
    elif name == "tesseract":
        import pytesseract

        engine = pytesseract
    else:
        engine = None
    _engine_cache[name] = engine
    return engine


def _resize(image: np.ndarray, max_side: int) -> np.ndarray:
    """Ограничивает длинную сторону изображения."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side or longest == 0:
        return image
    scale = max_side / float(longest)
    return cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))),
                      interpolation=cv2.INTER_AREA)


def _delookalike(text: str) -> str:
    """Обратная замена визуально похожих латинских символов на кириллицу."""
    return text.translate(_LOOKALIKE)


def parse_fields(lines: list[str]) -> dict:
    """Извлекает год и сладость из строк OCR.

    Args:
        lines: распознанные строки.

    Returns:
        Словарь с ключами vintage и sweetness (пустые строки, если не найдено).
    """
    raw = " ".join(lines)
    folded = unicodedata.normalize("NFKC", raw).lower()
    years = YEAR_RE.findall(folded)
    vintage = ""
    match = re.search(r"\b((?:19|20)\d{2})\b", folded)
    if match:
        vintage = match.group(1)
    sweetness = ""
    for candidate in (folded, _delookalike(folded)):
        for token in SWEETNESS:
            if token in candidate:
                sweetness = token
                break
        if sweetness:
            break
    return {"vintage": vintage, "sweetness": sweetness, "years_found": years}


def run_ocr(image: np.ndarray, max_side: int = 768, engine: str | None = None) -> OcrResult:
    """Распознает текст на кропе этикетки и нормализует поля.

    Args:
        image: BGR-изображение (нормализованный кроп этикетки).
        max_side: ограничение длинной стороны (бюджет времени).
        engine: явный движок; по умолчанию — автоопределение.

    Returns:
        OcrResult; при отсутствии движка или сбое ok=False с причиной.
    """
    name = engine or resolve_engine()
    if name == "none":
        return OcrResult(ok=False, engine="none", reason="no OCR engine available")
    started = time.perf_counter()
    prepared = _resize(image, max_side)
    try:
        active = None if name == "cyrillic" else _get_engine(name)
        if name == "cyrillic":
            from packages.common.ocr_cyrillic import run_cyrillic  # noqa: PLC0415

            lines = [str(text).strip() for _, text, _ in run_cyrillic(prepared, max_side=max_side)]
        elif name == "rapidocr":
            result, _ = active(prepared)  # type: ignore[operator]
            lines = [str(item[1]).strip() for item in (result or []) if len(item) > 1]
        elif name == "easyocr":
            lines = [str(part).strip() for part in active.readtext(prepared, detail=0)]  # type: ignore[attr-defined]
        else:
            lines = [line.strip() for line in active.image_to_string(prepared, lang="rus+eng").splitlines()]
    except Exception as error:  # noqa: BLE001 - сбой движка не должен валить пайплайн
        log_marker("M-OCR", "run_ocr", "ENGINE_FAILED", engine=name, error=type(error).__name__)
        return OcrResult(ok=False, engine=name, reason=f"{type(error).__name__}: {error}"[:200])

    lines = [line for line in lines if line]
    fields = parse_fields(lines)
    latency_ms = (time.perf_counter() - started) * 1000.0
    log_marker("M-OCR", "run_ocr", "DONE", engine=name, lines=len(lines),
               vintage=fields["vintage"], sweetness=fields["sweetness"],
               latency_ms=round(latency_ms, 1))
    return OcrResult(
        ok=True,
        engine=name,
        lines=lines,
        vintage=fields["vintage"],
        sweetness=fields["sweetness"],
        winery_hint=lines[0] if lines else "",
        latency_ms=round(latency_ms, 1),
    )
