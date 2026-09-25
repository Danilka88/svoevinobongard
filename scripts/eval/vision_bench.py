#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Офлайн-бенч читателей этикеток без бюджетного ограничения: качество чтения и сквозной результат
# SCOPE: DEFAULT_FIELD_DIR, DEFAULT_TIMING_DIR, DEFAULT_TIMING_LIMIT, DEFAULT_CATALOG, DEFAULT_GOLD,
#        DEFAULT_OUT, DEFAULT_MD, DEFAULT_READERS, READER_LABELS, DEEPSEEK_OCR_PROMPT, OLLAMA_PROMPTS,
#        OCR_ENGINE_READERS, PADDLE_READER, READER_TOKENS, READER_TIMEOUT_S, TIMING_TIMEOUT_S, TOP_K,
#        IMAGE_SUFFIXES, QUALITY_LIMITS, TIMING_LIMITS, NOT_MEASURED_REASONS, settings, percentile,
#        load_image_bgr, observe_image, observe_ocr,
#        observe_vl, run_reader,
#        observer_hits_indexed, expected_winery, reading_quality, e2e_for_reader, run_quality, run_timing,
#        pick_best_worst, example_rows, build_conclusions, build_markdown, save_reports, render_console, main
# DEPENDS: argparse, json, sys, time, dataclasses, pathlib, numpy, packages.common.catalog,
#          packages.common.config, packages.common.images, packages.common.observer, packages.common.ocr,
#          packages.common.ocr_vl, packages.common.preprocess, packages.common.text_norm,
#          packages.common.text_rank, services.normalize.app.tta, services.vlm.app.extract,
#          scripts.eval.field21_abc, scripts.eval.field21_gold
# LINKS: M-EVAL, M-OCR, M-VLM, M-COMMON
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# DEFAULT_FIELD_DIR - каталог 21 проверенного полевого фото с эталонным названием
# DEFAULT_TIMING_DIR - подвыборка 100 реальных фото организатора для замера времени
# DEFAULT_TIMING_LIMIT - сколько фото подвыборки брать по умолчанию
# DEFAULT_CATALOG - catalog.jsonl источник slug
# DEFAULT_GOLD - эталонный маппинг фото → slug, построенный field21_gold
# DEFAULT_OUT - путь JSON-отчёта бенча
# DEFAULT_MD - путь Markdown-отчёта бенча
# DEFAULT_READERS - читатели по умолчанию
# READER_LABELS - человекочитаемые названия читателей
# DEEPSEEK_OCR_PROMPT - OCR-промпт для deepseek-ocr
# OLLAMA_PROMPTS - промпты отдельных читателей Ollama
# OCR_ENGINE_READERS - читатели поверх движков packages/common/ocr.py
# PADDLE_READER - имя читателя PaddleOCR-VL
# READER_TOKENS - лимит токенов читателя
# READER_TIMEOUT_S - таймаут читателя без бюджетного ограничения (щедрый)
# TIMING_TIMEOUT_S - таймаут читателя на подвыборке времени
# TOP_K - сколько кандидатов брать из текстового поиска
# IMAGE_SUFFIXES - расширения изображений
# QUALITY_LIMITS - уменьшенные подвыборки качества для очень медленных читателей
# TIMING_LIMITS - уменьшенные подвыборки времени для очень медленных читателей
# NOT_MEASURED_REASONS - причины, по которым читатель честно помечается «не измерен»
# settings - настройки проекта (пороги решения и бюджет наблюдателя, читаются, не меняются)
# percentile - перцентиль по ближайшему рангу
# load_image_bgr - чтение изображения в BGR через PIL (webp включительно)
# observe_image - наблюдение Ollama-моделью без бюджетного ограничения
# observe_ocr - чтение OCR-движком (cyrillic/fusion)
# observe_vl - чтение PaddleOCR-VL
# run_reader - запуск читателя по имени
# observer_hits_indexed - поиск по наблюдению с предвычисленным индексом каталога
# expected_winery - ожидаемая винодельня эталона для фото
# reading_quality - качество чтения: похожесть на название и прочтение бренда
# e2e_for_reader - сквозной результат наблюдение → поиск → slug по фото
# run_quality - прогон качества чтения и сквозных метрик на 21 фото
# run_timing - замер времени на подвыборке
# pick_best_worst - лучший и худший читатель по сквозной точности
# example_rows - примеры «фото → чтение → кандидаты → slug»
# build_conclusions - честные выводы по бюджетам
# build_markdown - Markdown-отчёт бенча
# save_reports - сохранение JSON и Markdown
# render_console - сводка в консоль
# main - CLI бенча
# END_MODULE_MAP
"""Бенч читателей этикеток.

Два слоя измерения на одном эталоне (21 проверенное полевое фото):

* **качество чтения** — похожесть прочитанного текста на эталонное название и отдельно факт
  прочтения бренда/винодельни каталога (похожесть через транслит/омоглифы);
* **сквозной результат** — наблюдение → нормализация → текстовый поиск по **всему** каталогу → slug:
  accuracy@1/@5 и честная доля not_found.

Время снимается без бюджетного ограничения: на 21 фото и на подвыборке 100 реальных фото
(``data/samples/real100``). Бенч офлайн: не пишет в лог flywheel, не меняет онлайн-дефолты config,
не создаёт коммит. Читатели: Ollama-модели (gemma4:e4b/12b/26b, deepseek-ocr), PaddleOCR-VL и
OCR-движки проекта (кириллица/слияние).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import packages.common.config as config_module  # noqa: E402
from packages.common.catalog import load_catalog  # noqa: E402
from packages.common.observer import observation_fields, parse_observation  # noqa: E402
from packages.common.ocr import run_ocr  # noqa: E402
from packages.common.ocr_vl import run_vl  # noqa: E402
from packages.common.preprocess import PreprocessConfig  # noqa: E402
from packages.common.text_norm import similarity  # noqa: E402
from packages.common.text_rank import text_index, text_search  # noqa: E402
from scripts.eval.field21_abc import acceptable_slugs, compute_metrics, is_evaluable  # noqa: E402
from scripts.eval.field21_gold import load_gold  # noqa: E402
from services.normalize.app.tta import build_vlm_inputs  # noqa: E402
from services.vlm.app.extract import observe_label  # noqa: E402

settings = config_module.settings

DEFAULT_FIELD_DIR = _ROOT / "data" / "samples" / "real_field"
DEFAULT_TIMING_DIR = _ROOT / "data" / "samples" / "real100"
DEFAULT_TIMING_LIMIT = 100
DEFAULT_CATALOG = _ROOT / "data" / "catalog" / "catalog.jsonl"
DEFAULT_GOLD = _ROOT / "reports" / "field21_gold.json"
DEFAULT_OUT = _ROOT / "reports" / "vision_bench.json"
DEFAULT_MD = _ROOT / "reports" / "vision_bench.md"
DEFAULT_READERS = ("gemma4:e4b-mlx", "gemma4:12b-mlx", "gemma4:26b-mlx", "deepseek-ocr:latest",
                   "paddleocr-vl", "ocr-cyrillic")
READER_LABELS = {
    "gemma4:e4b-mlx": "gemma4:e4b-mlx (LLM-зрение, онлайн)",
    "gemma4:12b-mlx": "gemma4:12b-mlx (LLM-зрение)",
    "gemma4:26b-mlx": "gemma4:26b-mlx (LLM-зрение, офлайн-учитель)",
    "deepseek-ocr:latest": "deepseek-ocr (чистый OCR)",
    "paddleocr-vl": "PaddleOCR-VL (vendored)",
    "ocr-cyrillic": "OCR кириллица (PP-OCRv5 rec)",
    "ocr-fusion": "OCR слияние (RapidOCR+кириллица)",
}
DEEPSEEK_OCR_PROMPT = "Free OCR."
OLLAMA_PROMPTS = {"deepseek-ocr:latest": DEEPSEEK_OCR_PROMPT}
OCR_ENGINE_READERS = {"ocr-cyrillic": "cyrillic", "ocr-fusion": "fusion"}
PADDLE_READER = "paddleocr-vl"
READER_TOKENS = 256
READER_TIMEOUT_S = 180.0
TIMING_TIMEOUT_S = 120.0
TOP_K = 10
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
# C-073: PaddleOCR-VL на MPS читает фото ~100-230 с (замерено), поэтому 100 фото заняли бы часы.
# Для него берутся уменьшенные подвыборки качества и времени, и фактическое n честно попадает в отчёт.
QUALITY_LIMITS = {"paddleocr-vl": 6}
TIMING_LIMITS = {"paddleocr-vl": 4}
# C-073: читатели, которые честно не измеряются в прогоне, не должны выглядеть как «нулевые»:
# строка отчёта помечается «не измерен» с причиной, а не None. PaddleOCR-VL — офлайн-инструмент.
NOT_MEASURED_REASONS = {
    "paddleocr-vl": "не измерен: ≈226 с/фото на MPS (21 фото ≈80 мин), оставлен офлайн-инструментом",
}

__all__ = ["DEFAULT_CATALOG", "DEFAULT_FIELD_DIR", "DEFAULT_GOLD", "DEFAULT_MD", "DEFAULT_OUT",
           "DEFAULT_READERS", "DEFAULT_TIMING_DIR", "DEFAULT_TIMING_LIMIT", "DEEPSEEK_OCR_PROMPT",
           "IMAGE_SUFFIXES", "NOT_MEASURED_REASONS", "OCR_ENGINE_READERS", "OLLAMA_PROMPTS",
           "PADDLE_READER", "READER_LABELS",
           "READER_TIMEOUT_S", "READER_TOKENS", "TIMING_TIMEOUT_S", "TOP_K", "build_conclusions",
           "build_markdown", "e2e_for_reader", "example_rows", "expected_winery", "load_image_bgr",
           "main", "observe_image", "observe_ocr", "observe_vl", "observer_hits_indexed",
           "percentile", "pick_best_worst", "reading_quality", "render_console", "run_quality",
           "run_reader", "run_timing", "save_reports"]


def percentile(values: list[float], p: float) -> float:
    """Перцентиль по ближайшему рангу (единый метод с остальными отчётами проекта)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(float(ordered[0]), 1)
    k = (len(ordered) - 1) * p
    low, high = int(k), min(int(k) + 1, len(ordered) - 1)
    return round(float(ordered[low] + (ordered[high] - ordered[low]) * (k - low)), 1)


def load_image_bgr(path: str | Path) -> np.ndarray:
    """Читает изображение в BGR через PIL (поддерживает webp подвыборки)."""
    with Image.open(path) as handle:
        array = np.asarray(handle.convert("RGB"))
    return array[:, :, ::-1].copy()


def observe_image(path: str | Path, model: str, *, prompt: str | None = None,
                  tokens: int = READER_TOKENS, timeout_s: float = READER_TIMEOUT_S) -> dict:
    """Наблюдение Ollama-моделью без бюджетного ограничения.

    Настройки подменяются на время вызова (щедрый таймаут), затем возвращаются: онлайн-дефолты
    config не меняются. ``prompt`` задаёт инструкцию читателя (по умолчанию — штатная наблюдателя).
    """
    started = time.perf_counter()
    frame = load_image_bgr(path)
    images, _ = build_vlm_inputs(frame, PreprocessConfig(pas_tiles=settings.pas_tiles))
    original = config_module.settings
    patched = replace(original, observer_timeout_s=max(timeout_s, 1.0), observer_max_tokens=max(tokens, 1))
    config_module.settings = patched
    try:
        result = observe_label(images, None, model=model, max_tokens=tokens, timeout_s=timeout_s,
                               instruction=prompt)
    finally:
        config_module.settings = original
    result["wall_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    return result


def observe_ocr(path: str | Path, engine: str) -> dict:
    """Чтение OCR-движком проекта (cyrillic/fusion) на полном кадре."""
    started = time.perf_counter()
    frame = load_image_bgr(path)
    result = run_ocr(frame, engine=engine)
    return {"ok": bool(result.ok), "raw_text": "\n".join(result.lines), "engine": result.engine,
            "latency_ms": float(result.latency_ms), "wall_ms": round((time.perf_counter() - started) * 1000.0, 1),
            "reason": result.reason}


def observe_vl(path: str | Path) -> dict:
    """Чтение PaddleOCR-VL (vendored, packages/common/ocr_vl.py) на полном кадре."""
    started = time.perf_counter()
    frame = load_image_bgr(path)
    lines, latency = run_vl(frame)
    return {"ok": bool(lines), "raw_text": "\n".join(lines), "engine": "paddleocr_vl",
            "latency_ms": float(latency), "wall_ms": round((time.perf_counter() - started) * 1000.0, 1),
            "reason": "" if lines else "no lines"}


def run_reader(reader: str, path: str | Path, *, tokens: int = READER_TOKENS,
               timeout_s: float = READER_TIMEOUT_S) -> dict:
    """Запускает читателя по имени и возвращает единый результат."""
    if reader in OCR_ENGINE_READERS:
        return observe_ocr(path, OCR_ENGINE_READERS[reader])
    if reader == PADDLE_READER:
        return observe_vl(path)
    return observe_image(path, reader, prompt=OLLAMA_PROMPTS.get(reader), tokens=tokens,
                         timeout_s=timeout_s)


def observer_hits_indexed(observation: dict, catalog: dict, index: dict,
                          *, limit: int = TOP_K) -> list[tuple[str, float]]:
    """Поиск по наблюдению с предвычисленным индексом каталога (без перестройки форм)."""
    fields = observation_fields(observation)
    if not fields:
        return []
    hits = text_search(fields, catalog, limit=limit, index=index)
    return [(slug, float(score)) for slug, score in hits]


def expected_winery(record: dict | None, catalog: dict) -> str:
    """Ожидаемая винодельня эталона: для matched — по slug, для ambiguous — первый кандидат."""
    data = record or {}
    slug = data.get("expected_slug")
    if not slug and data.get("candidates"):
        slug = (data["candidates"][0] or {}).get("slug")
    if not slug:
        return ""
    return str((catalog.get(str(slug)) or {}).get("winery") or "")


def reading_quality(rows: list[dict], gold: dict, catalog: dict) -> dict:
    """Качество чтения: похожесть прочитанного на название и доля прочитанных брендов.

    Args:
        rows: записи читателя (``file``, ``raw_text``).
        gold: эталон field21_gold (``files`` → human_label/статус/slug).
        catalog: записи каталога ``slug → {winery, name, ...}``.

    Returns:
        Метрики первого слоя: средняя похожесть текста на название, доля прочитанных брендов и др.
    """
    files = (gold or {}).get("files", {})
    title_scores: list[float] = []
    winery_scores: list[float] = []
    brand_hits = 0
    nonempty = 0
    for row in rows:
        record = files.get(row.get("file"), {})
        label = str(record.get("human_label") or "")
        text = str(row.get("raw_text") or "")
        if text.strip():
            nonempty += 1
        title_scores.append(similarity(text, label) if label else 0.0)
        winery = expected_winery(record, catalog)
        if winery:
            score = similarity(text, winery)
            winery_scores.append(score)
            if score >= 0.6:
                brand_hits += 1

    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    return {
        "samples": len(rows),
        "observations_nonempty": nonempty,
        "title_similarity_mean": mean(title_scores),
        "winery_similarity_mean": mean(winery_scores),
        "brand_read_rate": round(brand_hits / len(winery_scores), 4) if winery_scores else 0.0,
        "brand_evaluated": len(winery_scores),
    }


def e2e_for_reader(reader: str, readings: dict[str, dict], gold: dict, catalog: dict,
                   index: dict) -> list[dict]:
    """Сквозной результат: наблюдение → нормализация → поиск по каталогу → slug."""
    results: list[dict] = []
    for file_name, reading in readings.items():
        raw_text = str(reading.get("raw_text") or "")
        observation = parse_observation(raw_text)
        hits = observer_hits_indexed(observation, catalog, index, limit=TOP_K)
        top5 = [slug for slug, _ in hits[:5]]
        results.append({
            "file": file_name,
            "reader": reader,
            "top1": top5[0] if top5 else None,
            "top5": top5,
            "not_found": not top5,
            "confident": bool(top5),
            "latency_ms": round(float(reading.get("latency_ms") or reading.get("wall_ms") or 0.0), 1),
            "reader_ms": round(float(reading.get("wall_ms") or reading.get("latency_ms") or 0.0), 1),
            "path": "reader" if top5 else "reader:empty",
            "scores": [round(score, 4) for _, score in hits[:5]],
            "observation": raw_text[:400],
            "fields": observation_fields(observation),
            "ok": bool(reading.get("ok")),
            "reason": str(reading.get("reason") or ""),
        })
    return results


def run_quality(gold: dict, catalog: dict, field_dir: Path, readers: list[str], *,
                tokens: int = READER_TOKENS, timeout_s: float = READER_TIMEOUT_S,
                index: dict | None = None, on_reader=None) -> dict:
    """Прогоняет читателей на 21 фото: качество чтения и сквозные метрики.

    ``on_reader`` вызывается после каждого читателя (сырые чтения, качество, сквозные метрики):
    это позволяет сохранять отчёт инкрементально и не терять результат долгого прогона.
    """
    index = index if index is not None else text_index(catalog)
    files = gold.get("files", {})
    names = [name for name in files if (field_dir / name).is_file()]
    readings: dict[str, dict] = {}
    quality: dict[str, dict] = {}
    e2e: dict[str, dict] = {}
    for reader in readers:
        per_reader: dict[str, dict] = {}
        started = time.perf_counter()
        subset = names[:QUALITY_LIMITS.get(reader, len(names))]
        for name in subset:
            result = run_reader(reader, field_dir / name, tokens=tokens, timeout_s=timeout_s)
            per_reader[name] = {"file": name, "raw_text": str(result.get("raw_text") or ""),
                                "ok": bool(result.get("ok")), "latency_ms": float(result.get("latency_ms") or 0.0),
                                "wall_ms": float(result.get("wall_ms") or 0.0),
                                "reason": str(result.get("reason") or "")}
        readings[reader] = per_reader
        quality[reader] = reading_quality(list(per_reader.values()), gold, catalog)
        results = e2e_for_reader(reader, per_reader, gold, catalog, index)
        e2e[reader] = compute_metrics(results, gold)
        print(f"  [quality] {reader:<22} {time.perf_counter() - started:6.1f}с  "
              f"title_sim={quality[reader]['title_similarity_mean']} "
              f"brand={quality[reader]['brand_read_rate']} "
              f"acc@1={e2e[reader]['accuracy_at_1']} acc@5={e2e[reader]['accuracy_at_5']} "
              f"nf={e2e[reader]['not_found_rate']}", flush=True)
        if on_reader is not None:
            on_reader(reader, per_reader, quality[reader], e2e[reader])
    return {"readings": readings, "quality": quality, "e2e": e2e}


def run_timing(timing_dir: Path, readers: list[str], *, limit: int = DEFAULT_TIMING_LIMIT,
               tokens: int = READER_TOKENS, timeout_s: float = TIMING_TIMEOUT_S,
               on_reader=None) -> dict:
    """Замеряет время читателей на подвыборке реальных фото (без эталона)."""
    files = sorted(path for path in timing_dir.iterdir()
                   if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    timing: dict[str, dict] = {}
    for reader in readers:
        subset = files[:TIMING_LIMITS.get(reader, limit)]
        latencies: list[float] = []
        failures = 0
        for path in subset:
            result = run_reader(reader, path, tokens=tokens, timeout_s=timeout_s)
            latency = float(result.get("wall_ms") or result.get("latency_ms") or 0.0)
            latencies.append(latency)
            if not result.get("ok"):
                failures += 1
        timing[reader] = {
            "n": len(latencies),
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "mean": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
            "failures": failures,
        }
        print(f"  [timing]  {reader:<22} n={len(latencies)} "
              f"p50={timing[reader]['p50']} p95={timing[reader]['p95']} "
              f"mean={timing[reader]['mean']} fail={failures}", flush=True)
        if on_reader is not None:
            on_reader(reader)
    return timing


def pick_best_worst(e2e: dict[str, dict], quality: dict[str, dict]) -> tuple[str, str]:
    """Лучший и худший читатель по сквозной точности, затем по качеству чтения."""
    if not e2e:
        return "", ""
    ordered = sorted(e2e, key=lambda reader: (e2e[reader].get("accuracy_at_1", 0.0),
                                              quality.get(reader, {}).get("title_similarity_mean", 0.0)),
                     reverse=True)
    return ordered[0], ordered[-1]


def example_rows(reader: str, readings: dict[str, dict], results: list[dict], gold: dict,
                 *, limit: int = 5) -> list[dict]:
    """Примеры «фото → чтение → кандидаты → slug» для читателя."""
    files = (gold or {}).get("files", {})
    by_file = {row["file"]: row for row in results}
    examples: list[dict] = []
    for name, reading in readings.items():
        row = by_file.get(name, {})
        record = files.get(name, {})
        examples.append({
            "file": name,
            "reader": reader,
            "status": record.get("status"),
            "expected": record.get("expected_slug") or record.get("status"),
            "human_label": record.get("human_label", ""),
            "observation": str(reading.get("raw_text") or "")[:300],
            "fields": observation_fields(parse_observation(str(reading.get("raw_text") or ""))),
            "candidates": list(row.get("top5") or [])[:3],
            "top1": row.get("top1"),
            "not_found": row.get("not_found"),
            "ok": reading.get("ok"),
        })
        if len(examples) >= limit:
            break
    return examples


def build_conclusions(payload: dict) -> list[str]:
    """Формулирует честные выводы по бюджетам, не подгоняя их под желаемое."""
    gold = payload.get("gold", {})
    counts = gold.get("counts", {})
    quality = payload.get("quality", {})
    e2e = payload.get("e2e", {})
    timing = payload.get("timing", {})
    notes = [f"Эталон: {counts.get('total', 0)} фото, вычислимых {counts.get('evaluable', 0)} "
             f"(однозначных {counts.get('matched', 0)}, неоднозначных {counts.get('ambiguous', 0)}), "
             f"вне каталога {counts.get('not_found', 0)}."]
    online: list[tuple[float, str, float]] = []
    for reader in payload.get("readers", []):
        q = quality.get(reader, {})
        metrics = e2e.get(reader, {})
        if not q and not metrics:
            # C-073: читатель честно не измерен — не выдаём None за «нулевой результат».
            notes.append(f"{READER_LABELS.get(reader, reader)}: "
                         f"{NOT_MEASURED_REASONS.get(reader, 'не измерен в этом прогоне')}.")
            continue
        latency = metrics.get("latency_ms", {})
        speed = timing.get(reader, {})
        p95 = speed.get("p95") if speed else latency.get("p95")
        notes.append(f"{READER_LABELS.get(reader, reader)}: чтение title_sim "
                     f"{q.get('title_similarity_mean')}, бренд {q.get('brand_read_rate')} "
                     f"({q.get('brand_evaluated')} фото), acc@1 {metrics.get('accuracy_at_1')}, "
                     f"acc@5 {metrics.get('accuracy_at_5')}, not_found {metrics.get('not_found_rate')}, "
                     f"p50/p95 21 фото {latency.get('p50')}/{latency.get('p95')} мс, "
                     f"p50/p95 100 фото {speed.get('p50')}/{speed.get('p95')} мс.")
        if p95 is not None and float(p95) <= 10000.0:
            online.append((float(metrics.get("accuracy_at_1", 0.0)), reader, float(p95)))
    best_offline = max(e2e, key=lambda reader: e2e[reader].get("accuracy_at_1", 0.0)) if e2e else ""
    if online:
        best_online = max(online, key=lambda item: (item[0], -item[2]))
        notes.append(f"Лучший онлайн (p95 ≤ 10 с): {READER_LABELS.get(best_online[1], best_online[1])} — "
                     f"acc@1 {best_online[0]}, p95 {best_online[2]} мс.")
    else:
        notes.append("Ни один читатель не уложился в онлайн-бюджет 10 с: онлайн-наблюдатель требует "
                     "отдельного решения.")
    if best_offline:
        notes.append(f"Лучший офлайн (без ограничения бюджета): "
                     f"{READER_LABELS.get(best_offline, best_offline)} — "
                     f"acc@1 {e2e[best_offline].get('accuracy_at_1')}, "
                     f"p95 100 фото {timing.get(best_offline, {}).get('p95')} мс.")
        best_online_acc = max((value for value, _, _ in online), default=0.0)
        offline_acc = float(e2e[best_offline].get("accuracy_at_1", 0.0))
        if online and best_offline != max(online, key=lambda item: item[0])[1] and offline_acc > best_online_acc + 0.05:
            notes.append(f"Офлайн-модель существенно точнее онлайн (+{round(offline_acc - best_online_acc, 4)}): "
                         f"зафиксирована как офлайн-учитель для разметки (задел C-076).")
    return notes


def build_markdown(payload: dict) -> str:
    """Markdown-отчёт: таблица читателей, примеры лучшего/худшего и честные выводы."""
    gold = payload.get("gold", {})
    counts = gold.get("counts", {})
    quality = payload.get("quality", {})
    e2e = payload.get("e2e", {})
    timing = payload.get("timing", {})
    lines = [
        "# Бенч читателей этикеток (C-073)",
        "",
        f"- фото: **{counts.get('total', 0)}** · однозначных {counts.get('matched', 0)} · "
        f"неоднозначных {counts.get('ambiguous', 0)} · ненайденных {counts.get('not_found', 0)} "
        f"(вычислимых {counts.get('evaluable', 0)})",
        f"- читатели: {', '.join(payload.get('readers', []))}",
        f"- подвыборка времени: {payload.get('timing_limit')} фото из {payload.get('timing_dir')}",
        "- качество чтения — похожесть прочитанного текста на эталонное название и отдельно доля",
        "  прочитанных брендов/виноделен; сквозной результат — наблюдение → нормализация → поиск по",
        "  всему каталогу → slug (accuracy@1/@5, честный not_found). Бюджет не ограничивался.",
        "",
        "## Читатели: чтение, сквозной результат, время",
        "",
        "| Читатель | title_sim | бренд | acc@1 | acc@5 | not_found | p50/p95 (21) | p50/p95 (100) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    unmeasured: list[tuple[str, str]] = []
    for reader in payload.get("readers", []):
        q = quality.get(reader, {})
        metrics = e2e.get(reader, {})
        if not q and not metrics:
            # C-073: честная строка «не измерен» вместо None-метрик.
            unmeasured.append((READER_LABELS.get(reader, reader),
                               NOT_MEASURED_REASONS.get(reader, "не измерен в этом прогоне")))
            lines.append(f"| {READER_LABELS.get(reader, reader)} | — | — | — | — | — | — | — |")
            continue
        latency = metrics.get("latency_ms", {})
        speed = timing.get(reader, {})
        speed_text = (f"{speed.get('p50')} / {speed.get('p95')}" if speed else "—")
        lines.append(
            f"| {READER_LABELS.get(reader, reader)} | {q.get('title_similarity_mean')} | "
            f"{q.get('brand_read_rate')} ({q.get('brand_evaluated')}) | {metrics.get('accuracy_at_1')} | "
            f"{metrics.get('accuracy_at_5')} | {metrics.get('not_found_rate')} | "
            f"{latency.get('p50')} / {latency.get('p95')} | {speed_text} |")
    if unmeasured:
        lines += ["", "Читатели без измерения в этом прогоне: "
                  + "; ".join(f"{label} — {reason}" for label, reason in unmeasured) + "."]
    best, worst = payload.get("best_reader", ""), payload.get("worst_reader", "")
    for label, reader in (("Лучший читатель", best), ("Худший читатель", worst)):
        if not reader:
            continue
        lines += ["", f"## {label}: {READER_LABELS.get(reader, reader)}", ""]
        for example in example_rows(reader, payload.get("readings", {}).get(reader, {}),
                                    payload.get("results", {}).get(reader, []), gold):
            lines += [
                f"### {example['file']}",
                f"- эталон: `{example['expected']}` ({example['status']}) · {example['human_label']}",
                f"- чтение: {example['observation'] or '— (пусто)'}",
                f"- поля: `{json.dumps(example['fields'], ensure_ascii=False)}`" if example["fields"] else "- поля: —",
                f"- кандидаты: {', '.join(example['candidates']) or '—'}",
                f"- итог: `{example['top1'] or '—'}` · not_found={example['not_found']}",
                "",
            ]
    lines += ["## Честные выводы", ""] + [f"- {item}" for item in payload.get("conclusions", [])]
    return "\n".join(lines) + "\n"


def save_reports(payload: dict, out: Path, md: Path) -> None:
    """Сохраняет JSON и Markdown отчёты.

    JSON содержит и сырые чтения (``readings``), чтобы отчёт можно было пересчитать офлайн без
    повторного вызова моделей; Markdown строится из полного payload (примеры читателей).
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md.write_text(build_markdown(payload), encoding="utf-8")


def render_console(payload: dict) -> None:
    """Печатает сводку бенча в консоль."""
    print("===== БЕНЧ ЧИТАТЕЛЕЙ (21 фото) =====")
    quality = payload.get("quality", {})
    e2e = payload.get("e2e", {})
    timing = payload.get("timing", {})
    for reader in payload.get("readers", []):
        q = quality.get(reader, {})
        metrics = e2e.get(reader, {})
        speed = timing.get(reader, {})
        print(f"  {reader:<22} title_sim {q.get('title_similarity_mean')} · "
              f"бренд {q.get('brand_read_rate')} · acc@1 {metrics.get('accuracy_at_1')} · "
              f"acc@5 {metrics.get('accuracy_at_5')} · nf {metrics.get('not_found_rate')} · "
              f"p50/p95(21) {metrics.get('latency_ms', {}).get('p50')}/{metrics.get('latency_ms', {}).get('p95')} · "
              f"p50/p95(100) {speed.get('p50')}/{speed.get('p95')}")
    print("  выводы:")
    for note in payload.get("conclusions", []):
        print(f"    - {note}")


def main(argv: list[str] | None = None) -> int:
    """CLI бенча читателей: прогон качества и времени, запись JSON и Markdown отчётов."""
    parser = argparse.ArgumentParser(description="Бенч читателей этикеток (C-073)")
    parser.add_argument("--dir", default=str(DEFAULT_FIELD_DIR))
    parser.add_argument("--timing-dir", default=str(DEFAULT_TIMING_DIR))
    parser.add_argument("--timing-limit", type=int, default=DEFAULT_TIMING_LIMIT)
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--readers", default=",".join(DEFAULT_READERS))
    parser.add_argument("--reader-tokens", type=int, default=READER_TOKENS)
    parser.add_argument("--reader-timeout", type=float, default=READER_TIMEOUT_S)
    parser.add_argument("--timing-timeout", type=float, default=TIMING_TIMEOUT_S)
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md", default=str(DEFAULT_MD))
    args = parser.parse_args(argv)

    readers = [reader.strip() for reader in args.readers.split(",") if reader.strip()]
    gold = load_gold(args.gold)
    if not gold.get("files"):
        print(f"эталон не найден: {args.gold}. Сначала: scripts/eval/field21_gold.py", file=sys.stderr)
        return 2
    catalog = load_catalog(args.catalog)
    out, md = Path(args.out), Path(args.md)
    existing: dict = {}
    if out.is_file():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
        except ValueError:
            existing = {}
    payload: dict = {
        # C-073: список читателей объединяется с уже сохранённым — дополнительные прогоны
        # (например, только PaddleOCR-VL) не теряют ранее измеренных читателей.
        "readers": list(dict.fromkeys([*(existing.get("readers") or []), *readers])),
        "gold": gold,
        "field_dir": args.dir,
        "timing_dir": args.timing_dir,
        "timing_limit": args.timing_limit,
        "readings": existing.get("readings", {}),
        "results": existing.get("results", {}),
        "quality": existing.get("quality", {}),
        "e2e": existing.get("e2e", {}),
        "timing": existing.get("timing", {}),
    }
    if not args.skip_quality:
        index = text_index(catalog)

        def _on_quality(reader: str, readings: dict, qual: dict, metrics: dict) -> None:
            """Сохраняет отчёт после каждого читателя: долгий прогон не теряется при остановке."""
            payload["readings"][reader] = readings
            payload["quality"][reader] = qual
            payload["e2e"][reader] = metrics
            payload["results"][reader] = e2e_for_reader(reader, readings, gold, catalog, index)
            save_reports(payload, out, md)

        run_quality(gold, catalog, Path(args.dir), readers, tokens=args.reader_tokens,
                    timeout_s=args.reader_timeout, index=index, on_reader=_on_quality)
        save_reports(payload, out, md)
        print(f"  качество сохранено: {out}")
    if not args.skip_timing and Path(args.timing_dir).is_dir():
        def _persist(_reader: str) -> None:
            save_reports(payload, out, md)

        payload["timing"] = {**payload.get("timing", {}),
                             **run_timing(Path(args.timing_dir), readers, limit=args.timing_limit,
                                          tokens=args.reader_tokens, timeout_s=args.timing_timeout,
                                          on_reader=_persist)}
        save_reports(payload, out, md)
        print(f"  время сохранено: {out}")
    payload["best_reader"], payload["worst_reader"] = pick_best_worst(payload["e2e"], payload["quality"])
    payload["conclusions"] = build_conclusions(payload)
    save_reports(payload, out, md)
    render_console(payload)
    print(f"  отчёты: {out} и {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
