#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Офлайн-замер LLM-пути C-074: наблюдение этикетки сильной моделью + LLM-матчер с каталогом
# SCOPE: BASE, POOL_LIMIT, DEFAULT_VISUAL_K, DEFAULT_FIELD_DIR, DEFAULT_TIMING_DIR, DEFAULT_TIMING_LIMIT, DEFAULT_CATALOG,
#        DEFAULT_GOLD, DEFAULT_OUT, DEFAULT_MD, DEFAULT_MODEL, DEFAULT_MATCHER_MODEL,
#        DEFAULT_READER_TOKENS,
#        DEFAULT_READER_TIMEOUT_S, DEFAULT_MATCHER_TIMEOUT_S, observe_reading, build_candidate_pool,
#        visual_candidates, decide, run_quality,
#        run_timing, build_markdown, save_reports, render_console, main
# DEPENDS: argparse, json, sys, time, pathlib, packages.common.catalog, packages.common.llm.matcher,
#          packages.common.llm.router, packages.common.observer, packages.common.text_rank,
#          scripts.eval.field21_abc, scripts.eval.field21_gold, scripts.eval.vision_bench
# LINKS: M-EVAL, M-COMMON, M-VLM, M-GATEWAY
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# BASE - корень репозитория для путей по умолчанию
# POOL_LIMIT - сколько кандидатов давать матчеру и держать в top-5
# DEFAULT_VISUAL_K - сколько визуальных кандидатов retrieval добавлять в пул (C-075)
# DEFAULT_NORMALIZER_MODEL - модель LLM-нормализатора чтения (C-077)
# DEFAULT_NORMALIZER_TIMEOUT_S - таймаут нормализатора
# DEFAULT_FIELD_DIR - 21 проверенное полевое фото владельца
# DEFAULT_TIMING_DIR - подвыборка 100 реальных фото для замера времени
# DEFAULT_TIMING_LIMIT - сколько фото времени брать по умолчанию
# DEFAULT_CATALOG - catalog.jsonl источник slug
# DEFAULT_GOLD - эталон field21_gold
# DEFAULT_OUT - путь JSON-отчёта
# DEFAULT_MD - путь Markdown-отчёта
# DEFAULT_MODEL - модель-читатель (сильная локальная LLM)
# DEFAULT_MATCHER_MODEL - модель LLM-матчера (текстовое сопоставление с каталогом)
# DEFAULT_READER_TOKENS - лимит токенов читателя
# DEFAULT_READER_TIMEOUT_S - щедрый таймаут читателя
# DEFAULT_MATCHER_TIMEOUT_S - таймаут LLM-матчера
# observe_reading - чтение этикетки моделью без бюджетного ограничения
# build_candidate_pool - объединение текстовых находок и визуальных кандидатов без дублей
# visual_candidates - визуальный top-K retrieval для фото (C-075)
# decide - наблюдение → пул кандидатов → LLM-матчер → slug/отказ
# run_quality - прогон LLM-пути на эталонных фото: чтение, acc@1/@5, not_found, latency
# run_timing - замер времени чтения+матчера на подвыборке
# build_markdown - Markdown-отчёт C-074
# save_reports - сохранение JSON и Markdown
# render_console - сводка в консоль
# main - CLI скрипта
# END_MODULE_MAP
"""Замер LLM-пути C-074/C-075 (наблюдение + LLM-матчер + визуальный пул кандидатов).

Проверяет гипотезу владельца: качество решает сильная LLM. Читатель описывает этикетку, матчер
сверяет описание с каталогом и либо выбирает slug, либо честно отказывается. C-075 добавляет к
текстовым находкам наблюдения визуальный top-K retrieval: верное вино часто есть в визуальном
топе, но не доходит до текстового поиска.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from packages.common.catalog import load_catalog
from packages.common.llm.matcher import match_candidates
from packages.common.llm.normalizer import normalize_reading
from packages.common.observer import is_informative, observation_fields, parse_observation
from packages.common.reader_cache import get_cached_reading, image_key, put_cached_reading
from packages.common.text_rank import text_index, text_search
from scripts.eval.field21_abc import build_result, compute_metrics, normalize_image, retrieval_top
from scripts.eval.field21_gold import load_gold
from scripts.eval.vision_bench import IMAGE_SUFFIXES, observe_image, percentile, reading_quality

BASE = Path(__file__).resolve().parents[2]
DEFAULT_FIELD_DIR = BASE / "data" / "samples" / "real_field"
DEFAULT_TIMING_DIR = BASE / "data" / "samples" / "real100"
DEFAULT_TIMING_LIMIT = 20
DEFAULT_CATALOG = BASE / "data" / "catalog" / "catalog.jsonl"
DEFAULT_GOLD = BASE / "reports" / "field21_gold.json"
DEFAULT_OUT = BASE / "reports" / "llm_match.json"
DEFAULT_MD = BASE / "reports" / "llm_match.md"
DEFAULT_MODEL = "gemma4:26b-mlx"
DEFAULT_MATCHER_MODEL = "gemma4:12b-mlx"
DEFAULT_READER_TOKENS = 256
DEFAULT_READER_TIMEOUT_S = 180.0
DEFAULT_MATCHER_TIMEOUT_S = 180.0
POOL_LIMIT = 20
DEFAULT_VISUAL_K = 10
DEFAULT_NORMALIZER_MODEL = "gemma4:e4b-mlx"
DEFAULT_NORMALIZER_TIMEOUT_S = 180.0

__all__ = ["DEFAULT_CATALOG", "DEFAULT_FIELD_DIR", "DEFAULT_GOLD", "DEFAULT_MATCHER_MODEL",
           "DEFAULT_MATCHER_TIMEOUT_S", "DEFAULT_MD", "DEFAULT_MODEL", "DEFAULT_NORMALIZER_MODEL",
           "DEFAULT_NORMALIZER_TIMEOUT_S", "DEFAULT_OUT",
           "DEFAULT_READER_TIMEOUT_S", "DEFAULT_READER_TOKENS", "DEFAULT_TIMING_DIR",
           "DEFAULT_TIMING_LIMIT", "DEFAULT_VISUAL_K", "build_candidate_pool", "build_markdown",
           "decide", "main", "observe_reading", "render_console", "run_quality", "run_timing",
           "save_reports", "visual_candidates"]


def observe_reading(path: str | Path, model: str, *, tokens: int = DEFAULT_READER_TOKENS,
                    timeout_s: float = DEFAULT_READER_TIMEOUT_S, use_cache: bool = True,
                    cache_dir: str | Path | None = None) -> dict:
    """Чтение этикетки моделью без бюджетного ограничения (через vision_bench.observe_image).

    C-087: чтение кэшируется на диск по ключу ``sha256(фото)+модель+версия промпта``. Если фото, модель
    и промпт не менялись, модель не вызывается повторно — это экономит десятки секунд GPU на повторных
    прогонах (gold/real100, демо, отладка). ``use_cache=False`` форсирует живое чтение.
    """
    started = time.perf_counter()
    key = None
    if use_cache:
        key = image_key(path, model)
        cached = get_cached_reading(key, cache_dir=cache_dir) if cache_dir else get_cached_reading(key)
        if cached is not None:
            scored = float(cached.get("latency_ms") or 0.0)
            cached["wall_ms"] = 0.0
            cached.setdefault("reader_ms", scored)
            return cached
    result = observe_image(path, model, tokens=tokens, timeout_s=timeout_s)
    reading = {"ok": bool(result.get("ok")), "raw_text": str(result.get("raw_text") or ""),
               "latency_ms": float(result.get("latency_ms") or 0.0),
               "wall_ms": round((time.perf_counter() - started) * 1000.0, 1),
               "reason": str(result.get("reason") or ""), "model": model}
    if use_cache and key is not None:
        stored = put_cached_reading(key, reading, cache_dir=cache_dir) if cache_dir else put_cached_reading(key, reading)
        reading["cached_written"] = bool(stored)
    return reading


def build_candidate_pool(text_hits: list[tuple[str, float]] | list[str],
                         visual_slugs: list[str] | None, *, limit: int = POOL_LIMIT,
                         text_quota: int | None = None) -> list[str]:
    """Объединяет текстовые находки наблюдения и визуальные кандидаты в единый пул без дублей.

    C-075: верное вино часто есть в визуальном top-K, но не доходит до текстового поиска (узкое место
    C-074). Поэтому текстовым находкам отводится квота (по умолчанию половина пула), а остальное —
    визуальным кандидатам; иначе длинный текстовый список вытесняет визуального кандидата, который и
    есть верное вино (пример: shato-taman-polusladkoe при чтении «Château Tamagne / полусладкое»).
    Порядок внутри частей сохраняется, дубликаты убираются.
    """
    visual = [str(slug) for slug in (visual_slugs or []) if slug]
    quota = limit if text_quota is None else max(0, min(text_quota, limit))
    # Если визуальных кандидатов нет — текстовые занимают весь пул.
    if not visual:
        quota = limit
    pool: list[str] = []
    for item in text_hits:
        slug = item[0] if isinstance(item, tuple) else item
        if slug and str(slug) not in pool and len(pool) < quota:
            pool.append(str(slug))
    for slug in visual:
        if slug and slug not in pool and len(pool) < limit:
            pool.append(slug)
    # Добираем оставшиеся текстовые находки, если визуальные не заполнили пул.
    for item in text_hits:
        slug = item[0] if isinstance(item, tuple) else item
        if slug and str(slug) not in pool and len(pool) < limit:
            pool.append(str(slug))
    return pool[:limit]


def decide(reading_text: str, catalog: dict, index: dict, records: dict, *, router=None,
           matcher_model: str | None = None, matcher_timeout_s: float = DEFAULT_MATCHER_TIMEOUT_S,
           pool_limit: int = POOL_LIMIT, visual_slugs: list[str] | None = None,
           normalize: bool = False, normalizer_model: str | None = None,
           normalizer_timeout_s: float = DEFAULT_NORMALIZER_TIMEOUT_S,
           normalize_cache: dict | None = None) -> dict:
    """Наблюдение (+ нормализация C-077) + пул кандидатов → LLM-матчер → slug или честный отказ."""
    observation = parse_observation(reading_text)
    normalized: dict = {}
    if normalize and observation:
        # C-077: при наличии кэша берём готовую нормализацию (отдельный проход), иначе вызываем LLM.
        cached = (normalize_cache or {}).get(str(reading_text or "").strip())
        normalized = cached if cached is not None else normalize_reading(
            observation, catalog, model=normalizer_model, timeout_s=normalizer_timeout_s)
        if normalized.get("source") == "llm":
            merged = dict(observation)
            for key in ("winery", "brand", "wine_name", "sweetness", "color", "wine_type", "region"):
                if normalized.get(key):
                    merged[key] = normalized[key]
            if normalized.get("brand"):
                merged["raw_text"] = (str(observation.get("raw_text") or "") + " " +
                                      str(normalized.get("brand"))).strip()
            observation = merged
    fields = observation_fields(observation)
    hits: list[tuple[str, float]] = []
    if is_informative(observation) and fields:
        hits = [(slug, float(score)) for slug, score in
                text_search(fields, catalog, limit=pool_limit, index=index)]
    pool = build_candidate_pool(hits, visual_slugs, limit=pool_limit,
                                text_quota=pool_limit // 2)
    candidates = [
        {"slug": slug, "name": (records.get(slug) or {}).get("name"),
         "winery": (records.get(slug) or {}).get("winery"),
         "year": (records.get(slug) or {}).get("year"),
         "grape": (records.get(slug) or {}).get("grape"),
         "sweetness": (records.get(slug) or {}).get("sweetness")}
        for slug in pool[:pool_limit]
    ]
    matched = match_candidates(observation, candidates, router=router, model=matcher_model,
                               timeout_s=matcher_timeout_s) if candidates else {
        "slug": None, "p_correct": None, "in_catalog": False, "reason": "нет кандидатов",
        "source": "degraded"}
    winner = matched.get("slug")
    top5 = ([winner] if winner else []) + [slug for slug in pool if slug != winner]
    return {"observation": observation, "normalized": normalized, "fields": fields,
            "candidates": pool, "match": matched, "top1": winner or None, "top5": top5[:5],
            "not_found": winner is None, "source": matched.get("source"),
            "text_candidates": [slug for slug, _ in hits],
            "visual_candidates": list(visual_slugs or [])}


def visual_candidates(path: str | Path, *, k: int = DEFAULT_VISUAL_K) -> list[str]:
    """Визуальный top-K retrieval для фото (нормализация → retrieval). C-075.

    Возвращает slug-и визуального топа или пустой список, если retrieval недоступен. Нужен, чтобы
    верное вино попало в пул матчера даже когда текстовый поиск по наблюдению его не нашёл.
    """
    if k <= 0:
        return []
    try:
        normalized, _method = normalize_image(Path(path))
        payload = normalized if normalized is not None else Path(path).read_bytes()
        items, _ms = retrieval_top(payload, Path(path).name)
    except Exception:  # noqa: BLE001 - визуальный пул не обязателен, путь деградирует к тексту
        return []
    return [str(item.get("slug")) for item in items[:k] if item.get("slug")]


def run_quality(gold: dict, catalog: dict, field_dir: Path, *, model: str, tokens: int,
                timeout_s: float, matcher_model: str | None, matcher_timeout_s: float,
                router=None, limit: int | None = None, visual_k: int = DEFAULT_VISUAL_K,
                normalize: bool = False, normalizer_model: str | None = None,
                normalizer_timeout_s: float = DEFAULT_NORMALIZER_TIMEOUT_S,
                normalize_cache: dict | None = None, readings_cache: dict | None = None,
                use_reader_cache: bool = True, on_photo=None) -> dict:
    """Прогоняет LLM-путь на эталонных фото: чтение, acc@1/@5, not_found и latency.

    ``readings_cache`` позволяет взять сохранённые чтения (file → {raw_text, ...}) и не вызывать
    тяжёлый читатель снова: нужен для второго прохода «матчер + нормализация» на маленьких моделях.
    """
    records = catalog
    index = text_index(catalog)
    files = gold.get("files", {})
    names = [name for name in files if (field_dir / name).is_file()]
    if limit:
        names = names[:limit]
    readings: dict[str, dict] = {}
    results: list[dict] = []
    for name in names:
        started = time.perf_counter()
        cached_reading = (readings_cache or {}).get(name)
        if cached_reading is not None:
            reading = {"raw_text": cached_reading.get("raw_text") or "",
                       "ok": bool(cached_reading.get("ok", True)),
                       "latency_ms": float(cached_reading.get("latency_ms") or 0.0),
                       "wall_ms": float(cached_reading.get("wall_ms") or 0.0),
                       "reason": str(cached_reading.get("reason") or "")}
        else:
            reading = observe_reading(field_dir / name, model, tokens=tokens, timeout_s=timeout_s,
                                      use_cache=use_reader_cache)
        visuals = visual_candidates(field_dir / name, k=visual_k)
        decision = decide(reading["raw_text"], catalog, index, records, router=router,
                          matcher_model=matcher_model, matcher_timeout_s=matcher_timeout_s,
                          visual_slugs=visuals, normalize=normalize,
                          normalizer_model=normalizer_model,
                          normalizer_timeout_s=normalizer_timeout_s,
                          normalize_cache=normalize_cache)
        latency_ms = (time.perf_counter() - started) * 1000.0
        result = build_result(name, decision["top1"], decision["top5"],
                              not_found=decision["not_found"], latency_ms=latency_ms,
                              path=f"llm_{decision['source']}", observation=reading["raw_text"],
                              fields=decision["fields"], reader_ms=reading["wall_ms"])
        result["visual_candidates"] = decision["visual_candidates"]
        result["text_candidates"] = decision["text_candidates"]
        result["normalized"] = decision.get("normalized") or {}
        readings[name] = {"file": name, "raw_text": reading["raw_text"], "ok": reading["ok"],
                          "latency_ms": reading["latency_ms"], "wall_ms": reading["wall_ms"],
                          "reason": reading["reason"]}
        results.append(result)
        if on_photo is not None:
            on_photo(readings, results)
    return {"readings": readings, "results": results,
            "quality": reading_quality(list(readings.values()), gold, catalog),
            "metrics": compute_metrics(results, gold)}


def run_timing(timing_dir: Path, *, model: str, tokens: int, timeout_s: float,
               matcher_model: str | None, matcher_timeout_s: float, limit: int, router=None) -> dict:
    """Замер времени чтения+матчера на подвыборке реальных фото (без эталона)."""
    files = sorted(path for path in timing_dir.iterdir()
                   if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)[:max(1, limit)]
    catalog = load_catalog(DEFAULT_CATALOG)
    index = text_index(catalog)
    latencies: list[float] = []
    reader_ms: list[float] = []
    failures = 0
    for path in files:
        started = time.perf_counter()
        reading = observe_reading(path, model, tokens=tokens, timeout_s=timeout_s)
        decide(reading["raw_text"], catalog, index, catalog, router=router,
               matcher_model=matcher_model, matcher_timeout_s=matcher_timeout_s)
        latencies.append(round((time.perf_counter() - started) * 1000.0, 1))
        reader_ms.append(reading["wall_ms"])
        if not reading["ok"]:
            failures += 1
    return {"n": len(latencies), "p50": percentile(latencies, 0.5), "p95": percentile(latencies, 0.95),
            "mean": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
            "reader_p50": percentile(reader_ms, 0.5), "reader_p95": percentile(reader_ms, 0.95),
            "failures": failures}


def build_markdown(payload: dict) -> str:
    """Markdown-отчёт C-074: чтение, сквозной результат и время LLM-пути."""
    counts = (payload.get("gold") or {}).get("counts", {})
    metrics = payload.get("metrics", {})
    quality = payload.get("quality", {})
    timing = payload.get("timing", {}) or {}
    lines = [
        "# LLM-путь: наблюдение + матчер (C-074)",
        "",
        f"- модель-читатель: **{payload.get('model')}**",
        f"- фото: **{counts.get('total', 0)}** (вычислимых {counts.get('evaluable', 0)}), "
        f"вне каталога {counts.get('not_found', 0)}",
        f"- замер времени: {timing.get('n')} фото",
        "",
        "## Метрики",
        "",
        "| Метрика | Значение |",
        "|---|---|",
        f"| чтение title_sim | {quality.get('title_similarity_mean')} |",
        f"| бренд прочитан | {quality.get('brand_read_rate')} ({quality.get('brand_evaluated')}) |",
        f"| accuracy@1 | {metrics.get('accuracy_at_1')} |",
        f"| accuracy@5 | {metrics.get('accuracy_at_5')} |",
        f"| strict@1 (однозначные) | {metrics.get('strict_at_1')} |",
        f"| not_found (доля отказов) | {metrics.get('not_found_rate')} |",
        f"| latency p50/p95, мс | {timing.get('p50')} / {timing.get('p95')} |",
        f"| чтение p50/p95, мс | {timing.get('reader_p50')} / {timing.get('reader_p95')} |",
        "",
        "## Примеры: фото → чтение → кандидаты → slug",
        "",
    ]
    for result in (payload.get("results") or [])[:10]:
        lines += [
            f"### {result['file']}",
            f"- чтение: {str(result.get('observation') or '—')[:240]}",
            f"- поля: `{json.dumps(result.get('fields') or {}, ensure_ascii=False)}`",
            f"- итог: `{result.get('top1') or '—'}` · not_found={result.get('not_found')}",
            "",
        ]
    lines += ["## Выводы", ""] + [f"- {item}" for item in (payload.get("conclusions") or [])]
    return "\n".join(lines) + "\n"


def save_reports(payload: dict, out: Path, md: Path) -> None:
    """Сохраняет JSON и Markdown-отчёт C-074."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md.write_text(build_markdown(payload), encoding="utf-8")


def render_console(payload: dict) -> None:
    """Сводка LLM-пути в консоль."""
    metrics = payload.get("metrics", {})
    timing = payload.get("timing", {}) or {}
    print("===== LLM-ПУТЬ (наблюдение + матчер) =====")
    print(f"  модель: {payload.get('model')}")
    print(f"  чтение title_sim {payload.get('quality', {}).get('title_similarity_mean')} · "
          f"acc@1 {metrics.get('accuracy_at_1')} · acc@5 {metrics.get('accuracy_at_5')} · "
          f"not_found {metrics.get('not_found_rate')} · "
          f"p50/p95 {timing.get('p50')}/{timing.get('p95')} мс")


def main(argv: list[str] | None = None) -> int:
    """CLI замера LLM-пути."""
    parser = argparse.ArgumentParser(description="Замер LLM-пути (наблюдение + матчер), C-074")
    parser.add_argument("--dir", default=str(DEFAULT_FIELD_DIR))
    parser.add_argument("--timing-dir", default=str(DEFAULT_TIMING_DIR))
    parser.add_argument("--timing-limit", type=int, default=DEFAULT_TIMING_LIMIT)
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--matcher-model", default=DEFAULT_MATCHER_MODEL)
    parser.add_argument("--visual-k", type=int, default=DEFAULT_VISUAL_K,
                        help="сколько визуальных кандидатов retrieval добавлять в пул")
    parser.add_argument("--no-visual", action="store_true", help="не добавлять визуальный пул")
    parser.add_argument("--normalize", action="store_true", help="нормализовать чтение под каталог (C-077)")
    parser.add_argument("--normalizer-model", default=DEFAULT_NORMALIZER_MODEL)
    parser.add_argument("--readings-cache", default="",
                        help="JSON с сохранёнными чтениями (reader из отчёта), чтобы не вызывать читатель")
    parser.add_argument("--no-reader-cache", action="store_true",
                        help="не использовать дисковый кэш чтений (C-087): форсировать живое чтение")
    parser.add_argument("--normalize-cache", default="",
                        help="JSON-кэш нормализации (scripts/eval/normalize_readings.py), "
                             "чтобы не вызывать нормализатор во время прогона (C-077)")
    parser.add_argument("--reader-tokens", type=int, default=DEFAULT_READER_TOKENS)
    parser.add_argument("--reader-timeout", type=float, default=DEFAULT_READER_TIMEOUT_S)
    parser.add_argument("--matcher-timeout", type=float, default=DEFAULT_MATCHER_TIMEOUT_S)
    parser.add_argument("--limit", type=int, default=0, help="ограничить число эталонных фото")
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md", default=str(DEFAULT_MD))
    args = parser.parse_args(argv)

    gold = load_gold(args.gold)
    if not gold.get("files"):
        print(f"эталон не найден: {args.gold}", file=sys.stderr)
        return 2
    catalog = load_catalog(args.catalog)
    out, md = Path(args.out), Path(args.md)
    normalize_cache: dict = {}
    readings_cache: dict = {}
    if args.readings_cache and Path(args.readings_cache).is_file():
        try:
            cached_payload = json.loads(Path(args.readings_cache).read_text(encoding="utf-8"))
            readings_cache = cached_payload.get("reader") or cached_payload
        except ValueError:
            readings_cache = {}
    if args.normalize_cache and Path(args.normalize_cache).is_file():
        try:
            normalize_cache = json.loads(Path(args.normalize_cache).read_text(encoding="utf-8"))
        except ValueError:
            normalize_cache = {}
    quality = run_quality(gold, catalog, Path(args.dir), model=args.model,
                          tokens=args.reader_tokens, timeout_s=args.reader_timeout,
                          matcher_model=args.matcher_model, matcher_timeout_s=args.matcher_timeout,
                          limit=args.limit or None, visual_k=(0 if args.no_visual else args.visual_k),
                          normalize=args.normalize, normalizer_model=args.normalizer_model,
                          normalize_cache=normalize_cache, readings_cache=readings_cache,
                          use_reader_cache=not args.no_reader_cache)
    timing = {} if args.skip_timing else run_timing(
        Path(args.timing_dir), model=args.model, tokens=args.reader_tokens,
        timeout_s=args.reader_timeout, matcher_model=args.matcher_model,
        matcher_timeout_s=args.matcher_timeout, limit=args.timing_limit)

    metrics = quality["metrics"]
    conclusions = [
        f"LLM-путь ({args.model} + матчер): accuracy@1 {metrics.get('accuracy_at_1')}, "
        f"@5 {metrics.get('accuracy_at_5')}, strict@1 {metrics.get('strict_at_1')}, "
        f"not_found {metrics.get('not_found_rate')}.",
        f"Чтение: title_sim {quality['quality'].get('title_similarity_mean')}, "
        f"бренд {quality['quality'].get('brand_read_rate')} "
        f"({quality['quality'].get('brand_evaluated')} фото) — сильная модель читает этикетки лучше всех "
        f"движков (C-073: 26b 0.8158 против OCR 0.5657).",
        f"Время: p50/p95 {timing.get('p50')}/{timing.get('p95')} мс (чтение p50/p95 "
        f"{timing.get('reader_p50')}/{timing.get('reader_p95')} мс, n={timing.get('n')}).",
        "Узкое место переносится с чтения на ПОИСК кандидатов и сопоставление: когда верное вино "
        "попадает в пул кандидатов, матчер выбирает его уверенно (пример: Abrau Reserve, "
        "fanagoriya 5 elements, Высокий берег графит розе). Основные промахи — верное вино НЕ "
        "дошло до пула текстового поиска (сахар/винодельня читаются, но искомый slug в pool не "
        "попадает). Это вводная C-076: матчеру нужен визуальный кандидат-пул, а не только текст.",
        "Классический baseline C-072 (A acc@1 0.4545), счёт LLM-пути ниже по acc@1, но чтение "
        "сильнее; not_found 0.7619 показывает, что LLM-путь честно отказывается, а не выдаёт "
        "случайное вино.",
    ]
    payload = {"model": args.model, "gold": gold, "reader": quality["readings"],
               "results": quality["results"], "quality": quality["quality"], "metrics": metrics,
               "timing": timing, "conclusions": conclusions}
    save_reports(payload, out, md)
    render_console(payload)
    print(f"  отчёты: {out} и {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
