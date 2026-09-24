#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Реальный измерительный стенд A(классика)/B(LLM-зрение)/C(каскад) на 21 проверенном полевом фото
# SCOPE: DEFAULT_FIELD_DIR, DEFAULT_CATALOG, DEFAULT_GOLD, DEFAULT_OUT, DEFAULT_MD, DEFAULT_MODELS,
#        DEFAULT_READER_TIMEOUT_S, DEFAULT_READER_TOKENS, RETRIEVAL_TOP_K, IMAGE_SUFFIXES, MODE_LABELS,
#        NORMALIZE_URL, RETRIEVAL_URL, settings,
#        percentile, load_image_bgr, post_multipart, normalize_image, retrieval_top, observer_hits_indexed,
#        observe_image, acceptable_slugs, is_evaluable, build_result, mode_a_result, mode_b_result,
#        cascade_decision, mode_c_result, compute_metrics, example_rows, build_markdown, run_stand,
#        stringify_key, render_console, main
# DEPENDS: argparse, json, statistics, sys, time, pathlib, urllib, dataclasses, numpy, PIL,
#          packages.common.catalog, packages.common.config, packages.common.images, packages.common.observer,
#          packages.common.preprocess, packages.common.text_rank, services.normalize.app.tta,
#          services.vlm.app.extract
# LINKS: M-EVAL, M-RETRIEVAL, M-VLM
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# DEFAULT_FIELD_DIR - каталог 21 проверенного полевого фото
# DEFAULT_CATALOG - catalog.jsonl источник slug
# DEFAULT_GOLD - эталонный маппинг, построенный field21_gold
# DEFAULT_OUT - путь JSON-отчёта стенда
# DEFAULT_MD - путь Markdown-отчёта стенда
# DEFAULT_MODELS - модели-читатели по умолчанию (задел под C-073)
# DEFAULT_READER_TIMEOUT_S - бюджет модели-читателя (онлайн-наблюдатель — 6.5 с)
# DEFAULT_READER_TOKENS - лимит токенов наблюдения
# RETRIEVAL_TOP_K - сколько кандидатов брать из retrieval
# IMAGE_SUFFIXES - расширения изображений
# MODE_LABELS - человекочитаемые названия режимов A/B/C
# NORMALIZE_URL - адрес сервиса нормализации
# RETRIEVAL_URL - адрес сервиса retrieval
# settings - настройки проекта (пороги решения и бюджет наблюдателя)
# percentile - перцентиль по ближайшему рангу
# load_image_bgr - чтение изображения в BGR
# post_multipart - POST multipart/form-data с одним изображением
# normalize_image - нормализация через сервис normalize (первый кроп)
# retrieval_top - Top-K retrieval+geo через сервис retrieval
# observer_hits_indexed - поиск по наблюдению с предвычисленным индексом каталога
# observe_image - наблюдение этикетки моделью-читателем
# acceptable_slugs - приемлемые slug эталона (matched — один, ambiguous — набор)
# is_evaluable - вычислима ли метрика точности для фото
# build_result - запись результата одного фото в одном режиме
# mode_a_result - режим A: классика retrieval+geo
# mode_b_result - режим B: только LLM-зрение
# cascade_decision - выбор режима C: классика или эскалация к LLM-зрению
# mode_c_result - режим C: каскад классики и LLM-зрения
# compute_metrics - accuracy@1/@5, strict@1, not_found, latency, распределение путей
# example_rows - примеры «фото → чтение LLM → кандидаты → итоговый slug»
# build_markdown - Markdown-отчёт стенда
# run_stand - прогон A и B/C по всем моделям на 21 фото
# stringify_key - ключ режима отчёта (A или B:model / C:model)
# render_console - вывод сводки в консоль
# main - CLI стенда
# END_MODULE_MAP
"""Стенд A/B/C на 21 проверенном полевом фото.

Три режима на одном эталоне:

* **A — классика.** normalize → retrieval+geo, как в проде, observer выключен.
* **B — только LLM-зрение.** модель-читатель описывает этикетку, наблюдение нормализуется
  и ищется текстом по **всему** каталогу; визуальный поиск не участвует.
* **C — каскад.** сначала классика; при ``not_found``/неуверенности/``should_escalate``
  подключается LLM-зрение и кандидаты сливаются с визуальным топом.

Стенд офлайн: он не пишет в лог flywheel (иначе тестовые фото утекут в active learning) и не
меняет онлайн-дефолты. Список моделей-читателей задаётся ``--models`` — это задел под C-073.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import packages.common.config as config_module  # noqa: E402
from packages.common.catalog import load_catalog  # noqa: E402
from packages.common.images import load_bgr  # noqa: E402
from packages.common.observer import (escalation_trigger, merge_candidates,  # noqa: E402
                                      observation_fields, parse_observation)
from packages.common.preprocess import PreprocessConfig  # noqa: E402
from packages.common.text_rank import text_index, text_search  # noqa: E402
from services.normalize.app.tta import build_vlm_inputs  # noqa: E402
from services.vlm.app.extract import observe_label  # noqa: E402
from scripts.eval.field21_gold import load_gold  # noqa: E402

settings = config_module.settings

DEFAULT_FIELD_DIR = _ROOT / "data" / "samples" / "real_field"
DEFAULT_CATALOG = _ROOT / "data" / "catalog" / "catalog.jsonl"
DEFAULT_GOLD = _ROOT / "reports" / "field21_gold.json"
DEFAULT_OUT = _ROOT / "reports" / "field21_abc.json"
DEFAULT_MD = _ROOT / "reports" / "field21_abc.md"
DEFAULT_MODELS = ("gemma4:e4b-mlx",)
DEFAULT_READER_TIMEOUT_S = 6.5
DEFAULT_READER_TOKENS = 256
RETRIEVAL_TOP_K = 10
NORMALIZE_URL = "http://127.0.0.1:8081/normalize"
RETRIEVAL_URL = "http://127.0.0.1:8082/search"
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
MODE_LABELS = {"A": "A · классика (retrieval+geo)", "B": "B · только LLM-зрение",
               "C": "C · каскад (классика → LLM-зрение)"}

__all__ = ["DEFAULT_CATALOG", "DEFAULT_FIELD_DIR", "DEFAULT_GOLD", "DEFAULT_MD", "DEFAULT_MODELS",
           "DEFAULT_OUT", "DEFAULT_READER_TIMEOUT_S", "DEFAULT_READER_TOKENS", "IMAGE_SUFFIXES",
           "MODE_LABELS", "NORMALIZE_URL", "RETRIEVAL_TOP_K", "RETRIEVAL_URL", "acceptable_slugs",
           "build_markdown", "build_result", "cascade_decision", "compute_metrics", "example_rows",
           "is_evaluable", "load_image_bgr", "main", "mode_a_result", "mode_b_result", "mode_c_result",
           "normalize_image", "observe_image", "observer_hits_indexed", "percentile", "post_multipart",
           "render_console", "retrieval_top", "run_stand", "settings", "stringify_key"]


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
    """Читает изображение в BGR через numpy/PIL (тот же поворот каналов, что в сервисах)."""
    with Image.open(path) as handle:
        array = np.asarray(handle.convert("RGB"))
    return array[:, :, ::-1].copy()


def post_multipart(url: str, filename: str, data: bytes, *, timeout: float = 180.0) -> tuple[int, dict]:
    """POST multipart/form-data с одним изображением; HTTP-ошибка возвращается как код и тело."""
    boundary = "----svoeVinoField21"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'.encode(),
        b"Content-Type: application/octet-stream\r\n\r\n", data, b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ])
    request = urllib_request.Request(url, data=body,
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                                     method="POST")
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        return exc.code, {"error": exc.read().decode("utf-8", errors="replace")[:200]}
    except urllib_error.URLError as exc:
        return 0, {"error": f"{type(exc).__name__}: {exc}"[:200]}


def normalize_image(path: str | Path, *, base_url: str = NORMALIZE_URL) -> tuple[bytes | None, str]:
    """Нормализует фото через сервис normalize и возвращает байты первого кропа и метод.

    Как в проде (gateway): берётся первый нормализованный вариант; при недоступности сервиса
    возвращается исходное изображение — пайплайн не должен падать из-за нормализации.
    """
    data = Path(path).read_bytes()
    status, payload = post_multipart(base_url, Path(path).name, data)
    if status != 200:
        return None, "unavailable"
    crops = payload.get("crops") or []
    if not crops:
        return None, str(payload.get("method") or "none")
    import base64  # noqa: PLC0415 - локальный импорт, чтобы не тянуть base64 в модульный скоуп
    try:
        return base64.b64decode(crops[0]["png_base64"]), str(payload.get("method") or "unknown")
    except (KeyError, ValueError):
        return None, "decode_failed"


def retrieval_top(image_bytes: bytes, filename: str, *, base_url: str = RETRIEVAL_URL,
                  k: int = RETRIEVAL_TOP_K) -> tuple[list[dict], float]:
    """Возвращает Top-K retrieval+geo и задержку HTTP-вызова в миллисекундах."""
    started = time.perf_counter()
    status, payload = post_multipart(base_url, filename, image_bytes)
    latency_ms = (time.perf_counter() - started) * 1000.0
    if status != 200:
        return [], latency_ms
    return list(payload.get("top_k") or [])[:k], latency_ms


def observer_hits_indexed(observation: dict, catalog: dict, index: dict, *, limit: int = 10) -> list[tuple[str, float]]:
    """Поиск по наблюдению с предвычисленным индексом каталога.

    Это тело ``observer_candidates`` без повторного построения форм всего каталога: без
    индекса каждый вызов перестраивает ~8000 форм (десятки секунд), что неприемлемо для
    стенда на 21 фото × модели. Нормализация наблюдения — та же (``observation_fields``).
    """
    fields = observation_fields(observation)
    if not fields:
        return []
    hits = text_search(fields, catalog, limit=limit, index=index)
    return [(slug, float(score)) for slug, score in hits]


def observe_image(path: str | Path, model: str, *, tokens: int = DEFAULT_READER_TOKENS,
                  timeout_s: float = DEFAULT_READER_TIMEOUT_S) -> dict:
    """Наблюдение этикетки моделью-читателем через штатный ``observe_label``.

    Онлайн-бюджет наблюдателя (6.5 с) сохраняется по умолчанию. Для офлайн-сравнения
    тяжёлых моделей бюджет можно увеличить: настройки подменяются на время вызова и
    возвращаются обратно, поэтому онлайн-дефолты не меняются.
    """
    frame = load_image_bgr(path)
    images, _ = build_vlm_inputs(frame, PreprocessConfig(pas_tiles=settings.pas_tiles))
    original = config_module.settings
    patched = replace(original, observer_timeout_s=max(timeout_s, 1.0),
                      observer_max_tokens=max(tokens, 1))
    started = time.perf_counter()
    config_module.settings = patched
    try:
        result = observe_label(images, None, model=model, max_tokens=tokens, timeout_s=timeout_s)
    finally:
        config_module.settings = original
    result["wall_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    return result


def acceptable_slugs(gold_record: dict | None) -> set[str]:
    """Приемлемые slug эталона: matched — один ожидаемый, ambiguous — набор кандидатов."""
    record = gold_record or {}
    status = record.get("status")
    if status == "matched" and record.get("expected_slug"):
        return {str(record["expected_slug"])}
    if status == "ambiguous":
        return {str(item.get("slug")) for item in (record.get("candidates") or []) if item.get("slug")}
    return set()


def is_evaluable(gold_record: dict | None) -> bool:
    """Вычислима ли метрика точности: вино есть в каталоге (matched или ambiguous)."""
    return (gold_record or {}).get("status") in {"matched", "ambiguous"}


def build_result(file_name: str, top1: str | None, top5: list[str], *, not_found: bool,
                 latency_ms: float, path: str, scores: list[float] | None = None,
                 observation: str = "", fields: dict | None = None, reader_ms: float = 0.0,
                 confident: bool | None = None) -> dict:
    """Запись результата одного фото в одном режиме (единый формат для метрик и примеров)."""
    return {
        "file": file_name,
        "top1": top1,
        "top5": [slug for slug in top5 if slug],
        "not_found": bool(not_found),
        "confident": (not bool(not_found)) if confident is None else bool(confident),
        "latency_ms": round(float(latency_ms), 1),
        "reader_ms": round(float(reader_ms), 1),
        "path": path,
        "scores": [round(float(score), 4) for score in (scores or [])],
        "observation": observation[:400],
        "fields": fields or {},
    }


def mode_a_result(image_path: Path) -> tuple[dict, dict]:
    """Режим A: нормализация и retrieval+geo (классика как в проде, observer off).

    Уверенность считается как в gateway: top-1 выше ``SCORE_TAU`` и отрыв от top-2 выше
    ``MARGIN_DELTA``. Именно этот признак решает, эскалировать ли в режиме C.
    """
    started = time.perf_counter()
    normalized, method = normalize_image(image_path)
    payload = normalized if normalized is not None else image_path.read_bytes()
    items, retrieval_ms = retrieval_top(payload, image_path.name)
    latency_ms = (time.perf_counter() - started) * 1000.0
    top5 = [str(item.get("slug") or "") for item in items[:5] if item.get("slug")]
    scores = [float(item.get("score") or 0.0) for item in items[:5]]
    top1 = top5[0] if top5 else None
    best = scores[0] if scores else 0.0
    second = scores[1] if len(scores) > 1 else 0.0
    margin = round(best - second, 6)
    confident = bool(top1) and best >= float(settings.score_tau) and margin >= float(settings.margin_delta)
    not_found = best < float(settings.score_tau)
    result = build_result(image_path.name, top1, top5, not_found=not_found, latency_ms=latency_ms,
                          path=f"classic:{method}", scores=scores, confident=confident)
    return result, {"retrieval_ms": retrieval_ms, "normalize_method": method, "margin": margin,
                    "best_score": best}


def mode_b_result(image_path: Path, model: str, catalog: dict, index: dict, *, tokens: int,
                  timeout_s: float) -> dict:
    """Режим B: только LLM-зрение — наблюдение → нормализация → поиск по всему каталогу."""
    observed = observe_image(image_path, model, tokens=tokens, timeout_s=timeout_s)
    raw_text = str(observed.get("raw_text") or "")
    observation = parse_observation(raw_text)
    hits = observer_hits_indexed(observation, catalog, index, limit=10)
    top5 = [slug for slug, _ in hits[:5]]
    top1 = top5[0] if top5 else None
    reader_ms = float(observed.get("wall_ms") or observed.get("latency_ms") or 0.0)
    fields = observation_fields(observation)
    return build_result(image_path.name, top1, top5, not_found=not top5, latency_ms=reader_ms,
                        path=("observer" if top5 else "observer:empty"), scores=[score for _, score in hits[:5]],
                        observation=raw_text, fields=fields, reader_ms=reader_ms, confident=bool(top5))


def cascade_decision(classic: dict, observer_hits: list[tuple[str, float]], *,
                     p_correct: float | None = None, threshold: float = 0.95) -> dict:
    """Решает режим C: оставить классику или эскалировать к LLM-зрению.

    Эскалация — при неуверенности классики или ``should_escalate`` (когда доступна калибровка).
    Если наблюдение дало кандидата вне визуального топа, он получает право победить — именно так
    окупается LLM-зрение на перерисованных этикетках, выпавших из галереи эталонов.
    """
    classic_slugs = list(classic.get("top5") or [])
    classic_scores = list(classic.get("scores") or [])
    confident = bool(classic.get("confident", not classic.get("not_found")))
    if not escalation_trigger(p_correct, confident, threshold):
        return {"escalated": False, "route": "classic", "top1": classic.get("top1"),
                "top5": classic_slugs, "scores": classic_scores, "merged": []}
    if not observer_hits:
        return {"escalated": True, "route": "classic:observer_empty", "top1": classic.get("top1"),
                "top5": classic_slugs, "scores": classic_scores, "merged": []}
    padded = classic_scores + [0.0] * max(0, len(classic_slugs) - len(classic_scores))
    image_topk = [(slug, float(score)) for slug, score in zip(classic_slugs, padded)]
    merged = merge_candidates(observer_hits, image_topk)
    merged_slugs = [slug for slug, _ in merged]
    winner = merged_slugs[0] if merged_slugs else classic.get("top1")
    route = "observer" if winner not in classic_slugs else "classic+observer"
    return {"escalated": True, "route": route, "top1": winner,
            "top5": merged_slugs[:5] or classic_slugs, "scores": [score for _, score in merged[:5]],
            "merged": merged[:10]}


def mode_c_result(image_path: Path, classic: dict, model: str, catalog: dict, index: dict, *,
                  tokens: int, timeout_s: float, p_correct: float | None = None,
                  b_result: dict | None = None) -> dict:
    """Режим C: классика, при неуверенности — LLM-зрение и слияние кандидатов.

    Наблюдение берётся из результата B, если он передан: это тот же вызов читателя, и повторять
    его нельзя — иначе стенд платит за модель дважды и завышает задержку каскада.
    """
    if not escalation_trigger(p_correct, bool(classic.get("confident")), float(settings.observer_threshold)):
        # Классика уверена: LLM-зрение не вызывается (экономия бюджета, как в проде).
        return build_result(image_path.name, classic.get("top1"), list(classic.get("top5") or []),
                            not_found=bool(classic.get("not_found")),
                            latency_ms=float(classic.get("latency_ms") or 0.0),
                            path="classic", scores=list(classic.get("scores") or []),
                            confident=bool(classic.get("confident")))
    if b_result is not None:
        raw_text = str(b_result.get("observation") or "")
        hits = list(zip(b_result.get("top5") or [], b_result.get("scores") or []))
        reader_ms = float(b_result.get("reader_ms") or b_result.get("latency_ms") or 0.0)
    else:
        observed = observe_image(image_path, model, tokens=tokens, timeout_s=timeout_s)
        raw_text = str(observed.get("raw_text") or "")
        reader_ms = float(observed.get("wall_ms") or observed.get("latency_ms") or 0.0)
        observation = parse_observation(raw_text)
        hits = observer_hits_indexed(observation, catalog, index, limit=10)
    decision = cascade_decision(classic, hits)
    total_ms = float(classic.get("latency_ms") or 0.0) + reader_ms
    observation = parse_observation(raw_text)
    return build_result(image_path.name, decision.get("top1"), list(decision.get("top5") or []),
                        not_found=not decision.get("top1"), latency_ms=total_ms,
                        path=str(decision.get("route") or "classic"),
                        scores=list(decision.get("scores") or []), observation=raw_text,
                        fields=observation_fields(observation), reader_ms=reader_ms)


def compute_metrics(results: list[dict], gold: dict) -> dict:
    """Считает accuracy@1/@5 (вычислимые), strict@1 (однозначные), not_found, latency и пути.

    Точность на вычислимом наборе (matched + ambiguous) не завышается: для ambiguous
    правильным считается попадание в набор кандидатов каталога, а сам факт неоднозначности
    показывается отдельно. Позиции вне каталога (not_found) в точность не входят — для них
    считается корректность отказа.
    """
    files = gold.get("files", {})
    rows: list[dict] = []
    for result in results:
        record = files.get(result["file"], {})
        acceptable = acceptable_slugs(record)
        evaluable = is_evaluable(record)
        top5 = [slug for slug in (result.get("top5") or []) if slug]
        top1 = result.get("top1")
        hit1 = bool(evaluable and top1 in acceptable)
        hit5 = bool(evaluable and any(slug in acceptable for slug in top5))
        strict = record.get("status") == "matched"
        strict_hit = bool(strict and top1 == record.get("expected_slug"))
        absent = record.get("status") == "not_found"
        row = {**result, "status": record.get("status"), "acceptable": sorted(acceptable),
               "evaluable": evaluable, "hit1": hit1, "hit5": hit5,
               "strict": strict, "strict_hit": strict_hit, "expected_absent": absent,
               "refusal_correct": bool(absent and result.get("not_found"))}
        rows.append(row)
    evaluable_rows = [row for row in rows if row["evaluable"]]
    strict_rows = [row for row in rows if row["strict"]]
    absent_rows = [row for row in rows if row["expected_absent"]]
    latencies = [float(row.get("latency_ms") or 0.0) for row in rows]
    paths: dict[str, int] = {}
    for row in rows:
        key = str(row.get("path") or "")
        paths[key] = paths.get(key, 0) + 1

    def ratio(part: int, total: int) -> float:
        return round(part / total, 4) if total else 0.0

    return {
        "samples": len(rows),
        "evaluable": len(evaluable_rows),
        "matched_only": len(strict_rows),
        "expected_absent": len(absent_rows),
        "accuracy_at_1": ratio(sum(1 for row in evaluable_rows if row["hit1"]), len(evaluable_rows)),
        "accuracy_at_5": ratio(sum(1 for row in evaluable_rows if row["hit5"]), len(evaluable_rows)),
        "strict_at_1": ratio(sum(1 for row in strict_rows if row["strict_hit"]), len(strict_rows)),
        "not_found_rate": ratio(sum(1 for row in rows if row.get("not_found")), len(rows)),
        "refusal_correct_rate": ratio(sum(1 for row in absent_rows if row["refusal_correct"]), len(absent_rows)),
        "observations_nonempty": sum(1 for row in rows if str(row.get("observation") or "").strip()),
        "fields_extracted": sum(1 for row in rows if row.get("fields")),
        "latency_ms": {"p50": percentile(latencies, 0.50), "p95": percentile(latencies, 0.95)},
        "paths": paths,
        "rows": rows,
    }


def example_rows(result: dict, gold_record: dict, *, limit_observation: int = 260) -> dict:
    """Пример «фото → чтение LLM → кандидаты → итоговый slug» для отчёта."""
    return {
        "file": result["file"],
        "human_label": (gold_record or {}).get("human_label", ""),
        "expected": (gold_record or {}).get("expected_slug") or (gold_record or {}).get("status"),
        "status": (gold_record or {}).get("status"),
        "observation": str(result.get("observation") or "")[:limit_observation],
        "fields": result.get("fields") or {},
        "candidates": list((result.get("top5") or []))[:3],
        "top1": result.get("top1"),
        "not_found": result.get("not_found"),
    }


def build_markdown(payload: dict) -> str:
    """Markdown-отчёт: таблица A/B/C, примеры наблюдений LLM и честные выводы."""
    gold = payload.get("gold", {})
    counts = gold.get("counts", {})
    lines = [
        "# Стенд A/B/C на 21 проверенном полевом фото (C-072)",
        "",
        f"- фото: **{counts.get('total', 0)}** · однозначных {counts.get('matched', 0)} · "
        f"неоднозначных {counts.get('ambiguous', 0)} · ненайденных {counts.get('not_found', 0)} "
        f"(вычислимых {counts.get('evaluable', 0)})",
        f"- модели-читатели: {', '.join(payload.get('models', []))}",
        "- точность — на вычислимых фото (вино есть в каталоге); для неоднозначных верным считается",
        "  попадание в набор кандидатов линейки; ненайденные (зарубежные/отсутствующие) считаются отдельно;",
        "- режим A — retrieval+geo как в проде (observer off); B — только LLM-зрение; C — каскад.",
        "",
        "## A/B/C: accuracy, отказ, задержка",
        "",
        "| Режим | accuracy@1 | accuracy@5 | strict@1 | not_found | отказ верен | p50 / p95, мс |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, label in (("A", MODE_LABELS["A"]),):
        metrics = payload["modes"].get(key)
        if metrics:
            lines.append(_metrics_row(label, metrics))
    for model in payload.get("models", []):
        for key, label in (("B", MODE_LABELS["B"]), ("C", MODE_LABELS["C"])):
            metrics = payload["modes"].get(f"{key}:{model}")
            if metrics:
                lines.append(_metrics_row(f"{label} · {model}", metrics))
    lines += ["", "### Распределение путей", ""]
    for key in ["A"] + [f"{m}:{model}" for model in payload.get("models", []) for m in ("B", "C")]:
        metrics = payload["modes"].get(key)
        if metrics:
            windows = ", ".join(f"{name}×{count}" for name, count in sorted(metrics["paths"].items()))
            lines.append(f"- **{key}**: {windows}")
    lines += ["", "## Примеры: фото → чтение LLM → кандидаты → итоговый slug", ""]
    for example in payload.get("examples", []):
        lines += [
            f"### {example['file']}",
            f"- эталон: `{example['expected']}` ({example['status']})",
            f"- чтение LLM: {example['observation'] or '— (пусто)'}",
            f"- поля: `{json.dumps(example['fields'], ensure_ascii=False)}`" if example["fields"] else "- поля: —",
            f"- кандидаты B: {', '.join(example['candidates']) or '—'}",
            f"- итог B: `{example.get('b_top1') or '—'}` · итог C: `{example.get('c_top1') or '—'}`",
            "",
        ]
    lines += ["## Честные выводы", ""] + [f"- {item}" for item in payload.get("conclusions", [])]
    return "\n".join(lines) + "\n"


def _metrics_row(label: str, metrics: dict) -> str:
    """Строка таблицы метрик для отчёта."""
    latency = metrics["latency_ms"]
    return (f"| {label} | {metrics['accuracy_at_1']} | {metrics['accuracy_at_5']} | "
            f"{metrics['strict_at_1']} | {metrics['not_found_rate']} | "
            f"{metrics['refusal_correct_rate']} | {latency['p50']} / {latency['p95']} |")


def run_stand(gold: dict, catalog: dict, field_dir: Path, models: list[str], *,
              only: list[str] | None = None, tokens: int = DEFAULT_READER_TOKENS,
              timeout_s: float = DEFAULT_READER_TIMEOUT_S, a_results: dict | None = None) -> dict:
    """Прогоняет A и B/C по всем моделям; возвращает блоки метрик, примеры и выводы."""
    index = text_index(catalog)
    files = gold.get("files", {})
    names = [name for name in files if (field_dir / name).is_file()]
    if only:
        wanted = set(only)
        names = [name for name in names if name in wanted]
    a_metrics_cache = a_results or {}
    mode_metrics: dict[str, dict] = {}
    mode_results: dict[str, list[dict]] = {}

    classic_rows: list[dict] = []
    for name in names:
        if name in a_metrics_cache:
            classic_rows.append(a_metrics_cache[name])
            continue
        result, _ = mode_a_result(field_dir / name)
        classic_rows.append(result)
        print(f"  [A] {name[:36]:<38} top1={result['top1']} "
              f"{result['latency_ms']:.0f} мс", flush=True)
    mode_results["A"] = classic_rows
    mode_metrics["A"] = compute_metrics(classic_rows, gold)

    for model in models:
        b_rows: list[dict] = []
        c_rows: list[dict] = []
        for name, classic in zip(names, classic_rows):
            b_result = mode_b_result(field_dir / name, model, catalog, index, tokens=tokens, timeout_s=timeout_s)
            b_rows.append(b_result)
            c_result = mode_c_result(field_dir / name, classic, model, catalog, index,
                                     tokens=tokens, timeout_s=timeout_s, b_result=b_result)
            c_rows.append(c_result)
            print(f"  [{model}] {name[:28]:<30} B={b_result['top1'] or '—'} "
                  f"C={c_result['top1'] or '—'} ({c_result['path']}) {b_result['reader_ms']:.0f} мс", flush=True)
        mode_results[stringify_key("B", model)] = b_rows
        mode_results[stringify_key("C", model)] = c_rows
        mode_metrics[stringify_key("B", model)] = compute_metrics(b_rows, gold)
        mode_metrics[stringify_key("C", model)] = compute_metrics(c_rows, gold)

    examples = _build_examples(names, files, mode_results, models)
    conclusions = _conclusions(mode_metrics, models, gold)
    return {"modes": mode_metrics, "results": mode_results, "examples": examples,
            "conclusions": conclusions}


def stringify_key(mode: str, model: str) -> str:
    """Ключ режима для отчёта: ``A`` или ``B:model`` / ``C:model``."""
    return mode if mode == "A" else f"{mode}:{model}"


def _build_examples(names: list[str], files: dict, mode_results: dict, models: list[str]) -> list[dict]:
    """Собирает примеры по первому читателю: чтение LLM, кандидаты B и итоги B/C."""
    if not models:
        return []
    primary = models[0]
    b_rows = {row["file"]: row for row in mode_results.get(f"B:{primary}", [])}
    c_rows = {row["file"]: row for row in mode_results.get(f"C:{primary}", [])}
    examples: list[dict] = []
    for name in names:
        result = b_rows.get(name)
        if not result:
            continue
        example = example_rows(result, files.get(name, {}))
        example["b_top1"] = result.get("top1")
        example["c_top1"] = (c_rows.get(name) or {}).get("top1")
        examples.append(example)
    return examples


def _conclusions(mode_metrics: dict, models: list[str], gold: dict) -> list[str]:
    """Формулирует честные выводы из чисел, не подгоняя их под желаемое."""
    counts = gold.get("counts", {})
    notes = [f"Эталон: {counts.get('total', 0)} фото, вычислимых {counts.get('evaluable', 0)} "
             f"(однозначных {counts.get('matched', 0)}, неоднозначных {counts.get('ambiguous', 0)}), "
             f"вне каталога {counts.get('not_found', 0)}."]
    baseline = mode_metrics.get("A", {})
    base_acc = baseline.get("accuracy_at_1")
    notes.append(f"A (классика): accuracy@1 {base_acc}, "
                 f"@5 {baseline.get('accuracy_at_5')}, not_found {baseline.get('not_found_rate')}, "
                 f"p50/p95 {baseline.get('latency_ms', {}).get('p50')}/{baseline.get('latency_ms', {}).get('p95')} мс.")
    for model in models:
        b = mode_metrics.get(f"B:{model}", {})
        c = mode_metrics.get(f"C:{model}", {})
        empty = b.get("samples", 0) - b.get("observations_nonempty", 0)
        delta_b = round(b["accuracy_at_1"] - base_acc, 4) if (base_acc is not None
                                                              and b.get("accuracy_at_1") is not None) else None
        delta_c = round(c["accuracy_at_1"] - base_acc, 4) if (base_acc is not None
                                                              and c.get("accuracy_at_1") is not None) else None
        notes.append(f"B[{model}]: accuracy@1 {b.get('accuracy_at_1')}, @5 {b.get('accuracy_at_5')}, "
                     f"пустых наблюдений {empty}/{b.get('samples', 0)}, p50/p95 "
                     f"{b.get('latency_ms', {}).get('p50')}/{b.get('latency_ms', {}).get('p95')} мс"
                     + (f", Δ к A {delta_b:+}" if delta_b is not None else "") + ".")
        escalations = sum(1 for row in c.get("rows", []) if row.get("path") != "classic")
        notes.append(f"C[{model}]: accuracy@1 {c.get('accuracy_at_1')}, @5 {c.get('accuracy_at_5')}, "
                     f"эскалаций {escalations}/{c.get('samples', 0)}, p50/p95 "
                     f"{c.get('latency_ms', {}).get('p50')}/{c.get('latency_ms', {}).get('p95')} мс"
                     + (f", Δ к A {delta_c:+}" if delta_c is not None else "") + ".")
        if empty == b.get("samples", 0) and b.get("samples", 0):
            notes.append(f"{model} не уложилась в онлайн-бюджет наблюдателя ни на одном фото: "
                         "модель пригодна только для офлайн-разбора (задел C-073).")
    return notes


def render_console(payload: dict) -> None:
    """Печатает сводку A/B/C в консоль."""
    print("===== СТЕНД A/B/C (21 фото) =====")
    for key in ["A"] + [f"{m}:{model}" for model in payload.get("models", []) for m in ("B", "C")]:
        metrics = payload["modes"].get(key)
        if not metrics:
            continue
        latency = metrics["latency_ms"]
        print(f"  {key:<22} acc@1 {metrics['accuracy_at_1']} · acc@5 {metrics['accuracy_at_5']} · "
              f"strict@1 {metrics['strict_at_1']} · not_found {metrics['not_found_rate']} · "
              f"p50 {latency['p50']} / p95 {latency['p95']} мс")
    print("  выводы:")
    for note in payload.get("conclusions", []):
        print(f"    - {note}")


def main(argv: list[str] | None = None) -> int:
    """CLI стенда: прогон A/B/C, запись JSON и Markdown отчётов."""
    parser = argparse.ArgumentParser(description="Стенд A/B/C на 21 полевом фото")
    parser.add_argument("--dir", default=str(DEFAULT_FIELD_DIR))
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS),
                        help="список моделей-читателей через запятую (задел под C-073)")
    parser.add_argument("--only", default="", help="ограничить набор фото именами через запятую")
    parser.add_argument("--reader-tokens", type=int, default=DEFAULT_READER_TOKENS)
    parser.add_argument("--reader-timeout", type=float, default=DEFAULT_READER_TIMEOUT_S)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md", default=str(DEFAULT_MD))
    args = parser.parse_args(argv)

    gold = load_gold(args.gold)
    if not gold.get("files"):
        print(f"эталон не найден: {args.gold}. Сначала: scripts/eval/field21_gold.py", file=sys.stderr)
        return 2
    catalog = load_catalog(args.catalog)
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    only = [name.strip() for name in args.only.split(",") if name.strip()]
    payload = run_stand(gold, catalog, Path(args.dir), models, only=only,
                        tokens=args.reader_tokens, timeout_s=args.reader_timeout)
    payload.update({"gold": gold, "models": models, "field_dir": args.dir})
    out, md = Path(args.out), Path(args.md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md.write_text(build_markdown(payload), encoding="utf-8")
    render_console(payload)
    print(f"  отчёты: {out} и {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
