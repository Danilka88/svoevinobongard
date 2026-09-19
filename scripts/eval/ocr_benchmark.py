#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Микробенчмарк OCR-движков: синтетические текстовые линии и эталоны каталога с ground truth
# SCOPE: load_font, make_synthetic_label, synthetic_cases, catalog_cases, engine_lines, evaluate, main
# DEPENDS: numpy, PIL, rapidfuzz, packages.common.catalog, packages.common.config, packages.common.images,
#          packages.common.logging, packages.common.ocr, packages.common.ocr_cyrillic, packages.common.ocr_vl
# LINKS: M-EVAL, M-OCR
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# DEFAULT_CATALOG - сколько эталонов каталога брать по умолчанию
# DEFAULT_SYNTHETIC - сколько синтетических случаев брать по умолчанию
# FONT_CANDIDATES - системные шрифты с кириллицей для синтетики
# NAMES - названия вин для синтетических линий
# SWEETNESS - термины сладости для синтетических линий
# WINERIES - кириллические названия виноделен для синтетических линий
# GRAPES - кириллические сорта для синтетических линий
# YEARS - годы для проверки чтения цифр
# load_font - первый доступный шрифт с кириллицей
# make_synthetic_label - рисует синтетическую этикетку с заданными строками
# synthetic_cases - наборы синтетических случаев с известным текстом
# catalog_cases - эталоны каталога с ground truth из официального обогащения
# engine_lines - строки текста от выбранного движка
# evaluate - метрики движка по набору случаев
# main - прогон по движкам и отчёт
# END_MODULE_MAP
"""Микробенчмарк OCR с ground truth.

Два источника данных:
- **синтетика** — этикетка рисуется шрифтом, поэтому истина известна точно; отдельно
  проверяются арабские цифры (год), кириллические слова (винодельня, сорт) и «грязные»
  варианты с размытием и поворотом;
- **эталоны каталога** — настоящие изображения с ground truth из официального
  обогащения v3 (год, сладость, винодельня, сорт).

Метрики одинаковы для всех движков: чтение года, точность года, совпадение кириллических
слов, сладость, число кириллических слов и latency.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rapidfuzz import fuzz

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from packages.common.catalog import load_catalog, load_official_v3  # noqa: E402
from packages.common.config import settings  # noqa: E402
from packages.common.images import flatten_alpha  # noqa: E402
from packages.common.logging import log_marker  # noqa: E402
from packages.common.ocr import parse_fields, run_ocr  # noqa: E402

FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Verdana.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
)
WINERIES = ("Коммуналка", "Абрау-Дюрсо", "Фанагория", "Кубань-Вино", "Мысхако", "Золотая Балка")
NAMES = ("Пино Нуар", "Рислинг", "Саперави", "Шардоне", "Каберне Совиньон", "Сира Нуво")
GRAPES = ("Рислинг", "Мерло", "Алиготе", "Глера", "Красностоп")
YEARS = ("2019", "2020", "2021", "2022", "2023", "2024")
SWEETNESS = ("сухое", "полусухое", "полусладкое", "брют")
DEFAULT_SYNTHETIC = 60
DEFAULT_CATALOG = 40

__all__ = ["FONT_CANDIDATES", "GRAPES", "NAMES", "SWEETNESS", "WINERIES", "YEARS",
           "catalog_cases", "engine_lines", "evaluate", "load_font",
           "make_synthetic_label", "main", "synthetic_cases"]


def load_font(size: int = 46) -> ImageFont.FreeTypeFont:
    """Первый доступный системный шрифт с кириллицей."""
    for path in FONT_CANDIDATES:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def make_synthetic_label(lines: list[str], *, size: tuple[int, int] = (900, 1200),
                         blur: float = 0.0, angle: float = 0.0,
                         rng: np.random.Generator | None = None) -> np.ndarray:
    """Рисует синтетическую этикетку: белая карточка на тёмном фоне кадра.

    Args:
        lines: строки текста этикетки (ground truth).
        size: размер кадра.
        blur: радиус размытия (имитация расфокуса).
        angle: поворот этикетки в градусах.
        rng: генератор для фона.

    Returns:
        BGR-кадр целиком, как приходит с полевого фото.
    """
    rng = rng or np.random.default_rng(0)
    frame = np.full((size[1], size[0], 3), 34, dtype=np.uint8)
    frame += rng.integers(0, 18, size=frame.shape, dtype=np.uint8)
    card_w, card_h = int(size[0] * 0.72), int(size[1] * 0.62)
    card = Image.new("RGB", (card_w, card_h), (247, 245, 240))
    draw = ImageDraw.Draw(card)
    font = load_font()
    y = int(card_h * 0.12)
    for line in lines:
        draw.text((int(card_w * 0.08), y), line, fill=(22, 22, 26), font=font)
        y += int(card_h * 0.17)
    if angle:
        card = card.rotate(angle, expand=True, fillcolor=(247, 245, 240))
    if blur:
        from PIL import ImageFilter  # noqa: PLC0415

        card = card.filter(ImageFilter.GaussianBlur(blur))
    x = (size[0] - card.width) // 2
    y = (size[1] - card.height) // 2
    frame[y:y + card.height, x:x + card.width] = np.asarray(card)[:, :, ::-1]
    return frame


def synthetic_cases(count: int, seed: int = 42) -> list[dict]:
    """Синтетические случаи: этикетка с известным текстом, годом и сладостью."""
    rng = np.random.default_rng(seed)
    cases: list[dict] = []
    for index in range(count):
        winery = str(rng.choice(WINERIES))
        name = str(rng.choice(NAMES))
        grape = str(rng.choice(GRAPES))
        year = str(rng.choice(YEARS))
        sweet = str(rng.choice(SWEETNESS))
        lines = [winery, name, grape, f"{year}  {sweet}"]
        dirty = index % 5 == 4  # каждый пятый случай — «грязный»: размытие и наклон
        image = make_synthetic_label(lines, blur=1.1 if dirty else 0.0,
                                     angle=6.0 if dirty else 0.0, rng=rng)
        cases.append({
            "id": f"synthetic-{index:03d}",
            "image": image,
            "truth": {"year": year, "sweetness": sweet, "winery": winery, "name": name, "grape": grape},
            "dirty": dirty,
        })
    return cases


def _year_in_slug(slug: str) -> str:
    """Год из slug: только четырёхзначные числа в диапазоне каталога.

    Крепость в slug двузначная (…-13), поэтому четырёхзначное число — это год.
    """
    import re  # noqa: PLC0415

    for match in re.findall(r"(?:19|20)\d{2}", slug):
        if 1990 <= int(match) <= 2035:
            return match
    return ""


def catalog_cases(count: int, uploads: Path, seed: int = 42) -> list[dict]:
    """Эталоны каталога с ground truth из официального обогащения v3."""
    catalog = load_catalog(_ROOT / settings.catalog_jsonl)
    official = load_official_v3(_ROOT / "data" / "catalog" / "catalog_enriched_v3.jsonl")
    rng = np.random.default_rng(seed)
    candidates = [
        slug for slug, record in official.items()
        if record.get("api_match") and record.get("alcohol_pct") is not None
        and (uploads / (catalog.get(slug, {}).get("image") or "")).is_file()
    ]
    if len(candidates) > count:
        candidates = sorted(rng.choice(candidates, size=count, replace=False).tolist())
    cases: list[dict] = []
    for slug in candidates:
        path = uploads / (catalog[slug].get("image") or "")
        try:
            with Image.open(path) as raw:
                image = flatten_alpha(np.asarray(raw.convert("RGBA"))[:, :, [2, 1, 0, 3]])
        except Exception:  # noqa: BLE001
            continue
        record = official[slug]
        grapes = record.get("grapes") or []
        cases.append({
            "id": slug,
            "image": image,
            "truth": {
                "year": str(record.get("year") or _year_in_slug(slug) or ""),
                "sweetness": str(record.get("sweetness") or ""),
                "winery": str(record.get("manufacturer_name") or catalog[slug].get("winery") or ""),
                "name": str(catalog[slug].get("name") or ""),
                "grape": str(grapes[0]) if grapes else str(catalog[slug].get("grape") or ""),
            },
            "dirty": False,
        })
    return cases


def engine_lines(image: np.ndarray, engine: str) -> tuple[list[str], float]:
    """Строки текста от движка и его latency в мс.

    Движок fusion последовательно прогоняет два распознавателя (латиница+цифры и
    кириллица) и склеивает выдачу построчно без дублей.
    """
    started = time.perf_counter()
    if engine == "fusion":
        from rapidocr_onnxruntime import RapidOCR  # noqa: PLC0415

        base = RapidOCR()
        result, _ = base(image)
        lines = [str(item[1]).strip() for item in (result or []) if len(item) > 1]
        cyr = run_ocr(image, engine="cyrillic")
        latency = (time.perf_counter() - started) * 1000.0
        merged: list[str] = []
        seen: set[str] = set()
        for line in lines + list(cyr.lines):
            key = " ".join(line.lower().split())
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(line)
        return merged, latency
    result = run_ocr(image, engine=engine)
    latency = (time.perf_counter() - started) * 1000.0
    if not result.ok and not result.lines:
        # движок VLM возвращает уже разобранные поля, а не строки
        text = getattr(result, "text", "") or ""
        return [line for line in text.splitlines() if line.strip()], latency
    return list(result.lines), latency


def _clean(value: str) -> str:
    """Нормализация для сравнения строк."""
    return " ".join(str(value or "").lower().replace("ё", "е").split())


def evaluate(engine: str, cases: list[dict]) -> dict:
    """Считает метрики движка по набору случаев."""
    rows: list[dict] = []
    latencies: list[float] = []
    for case in cases:
        truth = case["truth"]
        lines, latency = engine_lines(case["image"], engine)
        latencies.append(latency)
        text = " ".join(lines)
        parsed = parse_fields(lines)
        year_truth = str(truth.get("year") or "")
        has_year_truth = bool(year_truth)
        winery_truth, name_truth = truth.get("winery") or "", truth.get("name") or ""
        grape_truth = truth.get("grape") or ""
        cyrillic_words = [word for word in text.split() if len(word) >= 3 and any("а" <= ch.lower() <= "я" for ch in word)]
        rows.append({
            "id": case["id"],
            "dirty": case.get("dirty", False),
            "year_read": bool(parsed["vintage"]),
            "year_ok": has_year_truth and parsed["vintage"] == year_truth,
            "year_truth": has_year_truth,
            "sweetness_ok": bool(truth.get("sweetness")) and parsed["sweetness"] == _clean(truth.get("sweetness")),
            "winery_match": max((fuzz.token_set_ratio(_clean(line), _clean(winery_truth)) for line in lines),
                                default=0.0) if winery_truth else 0.0,
            "name_match": max((fuzz.token_set_ratio(_clean(line), _clean(name_truth)) for line in lines),
                              default=0.0) if name_truth else 0.0,
            "grape_match": max((fuzz.token_set_ratio(_clean(line), _clean(grape_truth)) for line in lines),
                               default=0.0) if grape_truth else 0.0,
            "cyrillic_words": len(cyrillic_words),
            "lines": len(lines),
            "sample": " | ".join(lines[:3])[:160],
        })

    def share(key: str) -> float:
        return round(sum(1 for row in rows if row[key]) / len(rows), 4) if rows else 0.0

    def above(key: str, threshold: float) -> float:
        return round(sum(1 for row in rows if row[key] >= threshold) / len(rows), 4) if rows else 0.0

    ordered = sorted(latencies)

    def percentile(p: float) -> float:
        if not ordered:
            return 0.0
        return round(ordered[min(len(ordered) - 1, int(round(p / 100.0 * (len(ordered) - 1))))], 1)

    with_year = [row for row in rows if row["year_read"]]
    with_truth = [row for row in rows if row["year_truth"]]
    return {
        "engine": engine,
        "cases": len(rows),
        "year_read": share("year_read"),
        "year_ok": share("year_ok"),
        "year_truth_cases": len(with_truth),
        "year_ok_of_truth": round(sum(1 for row in with_truth if row["year_ok"]) / len(with_truth), 4) if with_truth else 0.0,
        "year_precision": round(sum(1 for row in with_year if row["year_ok"]) / len(with_year), 4) if with_year else 0.0,
        "sweetness_ok": share("sweetness_ok"),
        "winery_ge60": above("winery_match", 60.0),
        "name_ge60": above("name_match", 60.0),
        "grape_ge60": above("grape_match", 60.0),
        "avg_cyrillic_words": round(sum(row["cyrillic_words"] for row in rows) / len(rows), 2) if rows else 0.0,
        "avg_lines": round(sum(row["lines"] for row in rows) / len(rows), 2) if rows else 0.0,
        "latency_p50": percentile(50.0),
        "latency_p95": percentile(95.0),
        "rows": rows,
    }


def _ensure_cyrillic_env() -> str:
    """Включает кириллический движок, если его веса и словарь уже скачаны.

    Без этого замер cyrillic молча давал нули: движок недоступен, а метрики
    выглядят как «плохое качество» вместо «не измерено».
    """
    import os  # noqa: PLC0415

    model = _ROOT / "data" / "models" / "ocr" / "cyrillic_PP-OCRv5_rec.onnx"
    keys = _ROOT / "data" / "models" / "ocr" / "cyrillic_dict.txt"
    if not (model.is_file() and keys.is_file()):
        return "НЕДОСТУПЕН: нет весов или словаря"
    # settings — frozen dataclass, созданный при импорте: одного os.environ мало,
    # поэтому значение проставляется прямо в объект настроек
    object.__setattr__(settings, "ocr_rec_model", str(model))
    object.__setattr__(settings, "ocr_rec_keys", str(keys))
    return "включён из data/models/ocr"


def main(argv: list[str] | None = None) -> int:
    """Прогоняет бенчмарк по движкам и пишет отчёт."""
    parser = argparse.ArgumentParser(description="Микробенчмарк OCR-движков")
    parser.add_argument("--engines", default="rapidocr,cyrillic,fusion")
    parser.add_argument("--synthetic", type=int, default=DEFAULT_SYNTHETIC)
    parser.add_argument("--catalog", type=int, default=DEFAULT_CATALOG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="reports/ocr_benchmark.json")
    parser.add_argument("--vl-cases", type=int, default=3,
                        help="PaddleOCR-VL на MPS идёт ~3 минуты на кадр, поэтому только подвыборка")
    parser.add_argument("--uploads", default=str(
        _ROOT.parent / "материалы для конкурса" / "Датасет" / "prod-svoe-vino-strapi"
        / "prod-svoe-vino" / "strapi" / "uploads"))
    args = parser.parse_args(argv)

    print(f"кириллический движок: {_ensure_cyrillic_env()}")
    synthetic = synthetic_cases(args.synthetic, args.seed)
    catalog_set = catalog_cases(args.catalog, Path(args.uploads), args.seed)
    engines = [name.strip() for name in args.engines.split(",") if name.strip()]

    report: dict = {"synthetic_cases": len(synthetic), "catalog_cases": len(catalog_set),
                    "vl_cases": args.vl_cases, "engines": {}}
    for engine in engines:
        try:
            if engine == "paddleocr_vl":
                # VL-модель декодирует кадр минутами: честно ограничиваем подвыборкой
                subset_syn = synthetic[: args.vl_cases]
                subset_cat = catalog_set[: max(1, args.vl_cases - 1)]
                report["engines"][engine] = {
                    "synthetic": evaluate(engine, subset_syn),
                    "catalog": evaluate(engine, subset_cat),
                    "note": f"VL измерен на подвыборке {len(subset_syn)} синт. + {len(subset_cat)} эталонов",
                }
                continue
            report["engines"][engine] = {
                "synthetic": evaluate(engine, synthetic),
                "catalog": evaluate(engine, catalog_set),
            }
        except Exception as error:  # noqa: BLE001 - движок может быть недоступен
            report["engines"][engine] = {"error": f"{type(error).__name__}: {error}"[:200]}
            log_marker("M-EVAL", "ocr_benchmark", "ENGINE_FAILED", engine=engine,
                       error=type(error).__name__)

    target = _ROOT / args.out
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== БЕНЧМАРК OCR ===")
    print(f"синтетика: {len(synthetic)} · эталоны каталога: {len(catalog_set)}")
    print(f"{'движок':<10} {'набор':<9} {'год чит.':>8} {'год верен':>9} {'год из изв.':>11} "
          f"{'сладость':>8} {'winery≥60':>9} {'кир.слов':>8} {'p50мс':>7} {'p95мс':>7}")
    for engine, data in report["engines"].items():
        if "error" in data:
            print(f"{engine:<14} ОШИБКА: {data['error'][:70]}")
            continue
        for name in ("synthetic", "catalog"):
            block = data[name]
            print(f"{engine:<14} {name:<10} {block['year_read']:>9} {block['year_ok']:>10} "
                  f"{block['year_ok_of_truth']:>8} {block['sweetness_ok']:>9} {block['winery_ge60']:>10} "
                  f"{block['avg_cyrillic_words']:>12} {block['latency_p50']:>8} {block['latency_p95']:>8}")
    print(f"report: {target.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
