#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Честный замер кириллического OCR против текущего: proxy-эталоны с истинным текстом
#          (год, сладость, CER/WER, похожесть) и полевые фото (читаемость, время)
# SCOPE: ENGINES, PROXY_SIZE, FIELD_LIMIT, ensure_cyrillic_env, build_proxy_cases, load_field_labels,
#        text_of, year_hit, sweetness_hit, best_similarity, cer_wer, bench_proxy, bench_field,
#        build_report, main
# DEPENDS: argparse, importlib.util, json, statistics, sys, time, pathlib, numpy, rapidfuzz,
#          packages.common.catalog, packages.common.config, packages.common.logging,
#          packages.common.ocr, packages.common.text_norm
# LINKS: M-OCR, M-EVAL
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# ENGINES - сравниваемые по умолчанию движки OCR
# PROXY_SIZE - сколько синтетических proxy-эталонов брать по умолчанию
# FIELD_LIMIT - сколько полевых фото брать по умолчанию
# ensure_cyrillic_env - включает кириллический движок из vendored-весов и возвращает статус
# build_proxy_cases - синтетические proxy-эталоны с известным текстом (истина для CER/WER)
# load_field_labels - полевые фото с эталонными годом и сладостью
# text_of - весь распознанный текст одной строкой
# year_hit - найден ли правильный год
# sweetness_hit - найдена ли правильная сладость
# best_similarity - лучшая похожесть строк OCR на эталонную строку
# cer_wer - CER и WER распознанного текста против истины (с приведением к одной письменности)
# bench_proxy - прогон движка на proxy-эталонах: поля, CER/WER, похожесть, время
# bench_field - прогон движка на полевых фото: читаемость, поля, похожесть, время
# build_report - отчёт с двумя таблицами и прямым выводом
# main - точка входа
# END_MODULE_MAP
"""Замер кириллического OCR против текущего.

Два независимых набора:

- **proxy-эталоны** — синтетические этикетки, текст которых известен точно, поэтому
  считаются не только год и сладость, но и CER/WER распознанного текста. Это
  единственный набор, где CER/WER вообще измерим.
- **полевые фото** — реальные снимки с эталонными годом и сладостью из разметки
  (`reports/real100_labels.json`). Эталонного текста у них нет, поэтому здесь
  измеряются доля непустого текста, попадание полей и время.

Кириллический движок подключается флагом `OCR_ENGINE=cyrillic` (или `fusion`) и по
умолчанию не используется; при отсутствии весов он молча уступает текущему RapidOCR.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from packages.common.catalog import load_catalog  # noqa: E402
from packages.common.config import settings  # noqa: E402
from packages.common.logging import log_marker  # noqa: E402
from packages.common.ocr import run_ocr  # noqa: E402
from packages.common.text_norm import cyr_to_lat, fold, similarity  # noqa: E402

ENGINES = ("rapidocr", "cyrillic")
PROXY_SIZE = 40
FIELD_LIMIT = 40

__all__ = ["ENGINES", "FIELD_LIMIT", "PROXY_SIZE", "bench_field", "bench_proxy", "best_similarity",
           "build_proxy_cases", "build_report", "cer_wer", "ensure_cyrillic_env", "load_field_labels",
           "main", "sweetness_hit", "text_of", "year_hit"]


def _load_proxy_module():
    """Загружает генератор синтетических этикеток из соседнего бенчмарка (без пакета scripts)."""
    target = _ROOT / "scripts" / "eval" / "ocr_benchmark.py"
    spec = importlib.util.spec_from_file_location("_c066_ocr_benchmark", target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"не удалось загрузить генератор синтетики: {target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_cyrillic_env() -> str:
    """Включает кириллический движок из vendored-весов, если они на месте.

    Без этого замер `cyrillic` молча давал нули: движок считался недоступным, и
    нулевые метрики выглядели как «плохое качество» вместо «не измерено».
    """
    model = _ROOT / "data" / "models" / "ocr" / "cyrillic_PP-OCRv5_rec.onnx"
    keys = _ROOT / "data" / "models" / "ocr" / "cyrillic_dict.txt"
    if not (model.is_file() and keys.is_file()):
        return "НЕДОСТУПЕН: нет весов или словаря в data/models/ocr"
    object.__setattr__(settings, "ocr_rec_model", str(model))
    object.__setattr__(settings, "ocr_rec_keys", str(keys))
    return f"доступен: {model.name} + {keys.name}"


def build_proxy_cases(count: int, seed: int = 42) -> list[dict]:
    """Синтетические proxy-эталоны с известным текстом этикетки."""
    module = _load_proxy_module()
    cases: list[dict] = []
    for case in module.synthetic_cases(count, seed):
        truth = case["truth"]
        truth_text = " ".join([str(truth["winery"]), str(truth["name"]), str(truth["grape"]),
                               f"{truth['year']} {truth['sweetness']}"])
        cases.append({"id": case["id"], "image": case["image"], "truth": truth,
                      "truth_text": truth_text, "dirty": case.get("dirty", False)})
    return cases


def load_field_labels(labels_path: str | Path, limit: int) -> list[dict]:
    """Полевые фото со строгим эталоном (совпавшие и винодельня, и деталь)."""
    target = Path(labels_path)
    if not target.is_file():
        return []
    rows: list[dict] = []
    for row in json.loads(target.read_text(encoding="utf-8")).get("rows", []):
        strict = [candidate for candidate in (row.get("slug_candidates") or [])
                  if candidate.get("winery_score", 0) >= 0.86 and candidate.get("detail_score", 0) >= 0.86]
        if not strict:
            continue
        rows.append({"file": row["file"], "slug": str(strict[0]["slug"]), "year": row.get("year"),
                     "sweetness": row.get("sweetness"), "winery": row.get("winery"),
                     "name": row.get("name")})
    return rows[:limit]


def text_of(result) -> str:
    """Весь распознанный текст одной строкой."""
    return " ".join(str(line) for line in (getattr(result, "lines", []) or []))


def year_hit(expected: object, result) -> bool:
    """Найден ли правильный год: отдельным полем или в тексте."""
    wanted = str(expected or "").strip()
    if not wanted:
        return False
    if str(getattr(result, "vintage", "") or "").strip() == wanted:
        return True
    return wanted in text_of(result)


def sweetness_hit(expected: object, result) -> bool:
    """Найдена ли правильная сладость: отдельным полем или словом в тексте."""
    wanted = fold(str(expected or ""))
    if not wanted:
        return False
    if fold(str(getattr(result, "sweetness", "") or "")) == wanted:
        return True
    return wanted in fold(text_of(result))


def best_similarity(lines: list[str], reference: object) -> float:
    """Лучшая похожесть строк OCR на эталонную строку каталога."""
    target = str(reference or "").strip()
    if not target or not lines:
        return 0.0
    return round(max(similarity(str(line), target) for line in lines), 4)


def _script_unified(text: str) -> str:
    """Приводит текст к одной письменности, чтобы CER/WER мерил чтение, а не алфавит."""
    return cyr_to_lat(fold(text))


def _sorted_words(text: str) -> str:
    """Слова в алфавитном порядке: порядок строк у движков разный, он не про чтение."""
    return " ".join(sorted(_script_unified(text).split()))


def cer_wer(prediction: str, reference: str) -> tuple[float, float]:
    """CER и WER распознанного текста против истины.

    Порядок слов снимается сортировкой: детекторы RapidOCR и кириллического движка
    возвращают строки в разном порядке, и без этого CER/WER мерил бы порядок, а не
    качество чтения. Словарь приводится к одной письменности (кириллица→латиница),
    иначе смена алфавита выглядела бы как ошибка распознавания.
    """
    ref = _sorted_words(reference)
    pred = _sorted_words(prediction)
    if not ref:
        return 0.0, 0.0
    cer = Levenshtein.distance(pred, ref) / max(len(ref), 1)
    ref_words = ref.split()
    pred_words = pred.split()
    wer = Levenshtein.distance(pred_words, ref_words) / max(len(ref_words), 1)
    return round(cer, 4), round(wer, 4)


def _latency_block(latencies: list[float]) -> dict:
    """p50/p95 времени без обращения к каталогу."""
    if not latencies:
        return {"latency_p50_ms": 0.0, "latency_p95_ms": 0.0}
    ordered = sorted(latencies)
    return {"latency_p50_ms": round(statistics.median(ordered), 1),
            "latency_p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 1)}


def bench_proxy(engine: str, cases: list[dict]) -> dict:
    """Прогон движка на proxy-эталонах: поля, CER/WER, похожесть и время."""
    year_ok = sweet_ok = 0
    cer_values: list[float] = []
    wer_values: list[float] = []
    winery_values: list[float] = []
    name_values: list[float] = []
    latencies: list[float] = []
    samples: list[dict] = []
    for case in cases:
        started = time.perf_counter()
        result = run_ocr(case["image"], engine=engine)
        latencies.append((time.perf_counter() - started) * 1000.0)
        text = text_of(result)
        truth = case["truth"]
        year_ok += 1 if year_hit(truth["year"], result) else 0
        sweet_ok += 1 if sweetness_hit(truth["sweetness"], result) else 0
        cer, wer = cer_wer(text, case["truth_text"])
        cer_values.append(cer)
        wer_values.append(wer)
        winery_values.append(best_similarity(result.lines, truth["winery"]))
        name_values.append(best_similarity(result.lines, truth["name"]))
        if len(samples) < 5:
            samples.append({"id": case["id"], "truth": case["truth_text"][:120],
                            "text": text[:120], "cer": cer})
    total = len(cases) or 1
    block = {"cases": len(cases),
             "year_accuracy": round(year_ok / total, 4),
             "sweetness_accuracy": round(sweet_ok / total, 4),
             "cer_mean": round(statistics.mean(cer_values), 4) if cer_values else 0.0,
             "wer_mean": round(statistics.mean(wer_values), 4) if wer_values else 0.0,
             "winery_similarity_mean": round(statistics.mean(winery_values), 4) if winery_values else 0.0,
             "name_similarity_mean": round(statistics.mean(name_values), 4) if name_values else 0.0,
             "samples": samples}
    block.update(_latency_block(latencies))
    return block


def bench_field(engine: str, references: list[dict], catalog: dict, photos: Path) -> dict:
    """Прогон движка на полевых фото: читаемость, поля, похожесть на каталог и время."""
    year_ok = sweet_ok = non_empty = 0
    similarity_values: list[float] = []
    latencies: list[float] = []
    samples: list[dict] = []
    photos_count = 0
    for reference in references:
        path = photos / reference["file"]
        if not path.is_file():
            continue
        from PIL import Image  # noqa: PLC0415

        image = np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1].copy()
        started = time.perf_counter()
        result = run_ocr(image, engine=engine)
        latencies.append((time.perf_counter() - started) * 1000.0)
        photos_count += 1
        text = text_of(result)
        if text.strip():
            non_empty += 1
        year_ok += 1 if year_hit(reference["year"], result) else 0
        sweet_ok += 1 if sweetness_hit(reference["sweetness"], result) else 0
        entry = catalog.get(reference.get("slug") or "") or {}
        target = f"{entry.get('winery') or ''} {entry.get('name') or ''}".strip()
        similarity_values.append(best_similarity(result.lines, target))
        if len(samples) < 5:
            samples.append({"file": reference["file"], "truth_year": reference["year"],
                            "truth_sweetness": reference["sweetness"], "text": text[:120]})
    total = photos_count or 1
    block = {"photos": photos_count,
             "year_accuracy": round(year_ok / total, 4),
             "sweetness_accuracy": round(sweet_ok / total, 4),
             "non_empty_share": round(non_empty / total, 4),
             "catalog_similarity_mean": round(statistics.mean(similarity_values), 4) if similarity_values else 0.0,
             "samples": samples}
    block.update(_latency_block(latencies))
    return block


def _delta(cyrillic: dict, baseline: dict, key: str) -> float:
    """Разница метрики кириллического движка к текущему."""
    return round(float(cyrillic.get(key, 0.0)) - float(baseline.get(key, 0.0)), 4)


def build_report(results: dict[str, dict], status: str, flag_default: str,
                 out_json: str, out_md: str) -> dict:
    """Собирает таблицы, вывод и пишет отчёт json+md."""
    baseline = results.get("rapidocr")
    cyrillic = results.get("cyrillic")
    verdict = "недостаточно данных для вывода о кириллическом движке"
    deltas: dict[str, float] = {}
    if baseline and cyrillic:
        for key in ("year_accuracy", "sweetness_accuracy"):
            deltas[f"proxy_{key}"] = _delta(cyrillic["proxy"], baseline["proxy"], key)
            deltas[f"field_{key}"] = _delta(cyrillic["field"], baseline["field"], key)
        deltas["proxy_cer_mean"] = _delta(cyrillic["proxy"], baseline["proxy"], "cer_mean")
        proxy_gain = deltas["proxy_year_accuracy"] + deltas["proxy_sweetness_accuracy"]
        # Вывод соединяет оба набора: proxy показывает чтение слов, поле — реальные фото.
        # Рост только на proxy при падении года на поле выигрышем не объявляется.
        if proxy_gain > 0 and deltas["proxy_cer_mean"] < 0:
            lead = (f"читает слова и сладость лучше (proxy сладость "
                    f"{deltas['proxy_sweetness_accuracy']:+}, CER {deltas['proxy_cer_mean']:+})")
        elif proxy_gain > 0:
            lead = (f"на proxy сладость {deltas['proxy_sweetness_accuracy']:+}, "
                    f"год {deltas['proxy_year_accuracy']:+}, но CER {deltas['proxy_cer_mean']:+}")
        else:
            lead = (f"на proxy прироста нет: год {deltas['proxy_year_accuracy']:+}, "
                    f"сладость {deltas['proxy_sweetness_accuracy']:+}, CER {deltas['proxy_cer_mean']:+}")
        if deltas["field_year_accuracy"] < 0:
            verdict = (f"кириллический движок {lead}, НО на реальных фото год хуже "
                       f"({deltas['field_year_accuracy']:+}, сладость {deltas['field_sweetness_accuracy']:+}) — "
                       "однозначного выигрыша нет, движок остаётся выключенным по умолчанию")
        elif proxy_gain > 0:
            verdict = (f"кириллический движок {lead} и не ухудшает год на реальных фото — "
                       "полезен для чтения слов, включается флагом")
        else:
            verdict = (f"кириллический движок {lead} и не даёт прироста на реальных фото — "
                       "остаётся выключенным по умолчанию")
    report = {"engines": results, "cyrillic_status": status, "flag_default": flag_default,
              "deltas": deltas, "verdict": verdict,
              "caveat": ("CER/WER считается только на proxy-эталонах: у полевых фото нет эталонного текста; "
                         "порядок слов снят сортировкой (детекторы возвращают строки по-разному), "
                         "письменность приведена к одной (кириллица→латиница), иначе смена алфавита "
                         "выглядела бы как ошибка чтения; CER/WER штрафует и склейку слов одним токеном")}
    Path(out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Замер кириллического OCR (C-066)", "",
             f"- Кириллический движок: **{status}**",
             f"- Движок по умолчанию: `{flag_default or 'rapidocr'}` (кириллица выключена, включается `OCR_ENGINE=cyrillic`)",
             f"- Движки сравнения: {', '.join(results)}", "",
             "## Proxy-эталоны (истинный текст известен)", "",
             "| Движок | Кейсов | Год | Сладость | CER | WER | Похожесть винодельни | Похожесть названия | p50 / p95 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for engine, data in results.items():
        block = data["proxy"]
        lines.append(f"| {engine} | {block['cases']} | {block['year_accuracy']} | {block['sweetness_accuracy']} | "
                     f"{block['cer_mean']} | {block['wer_mean']} | {block['winery_similarity_mean']} | "
                     f"{block['name_similarity_mean']} | {block['latency_p50_ms']} / {block['latency_p95_ms']} мс |")
    lines += ["", "## Полевые фото (эталонного текста нет)", "",
              "| Движок | Фото | Год | Сладость | Непустой текст | Похожесть на каталог | p50 / p95 |",
              "|---|---|---|---|---|---|---|"]
    for engine, data in results.items():
        block = data["field"]
        lines.append(f"| {engine} | {block['photos']} | {block['year_accuracy']} | {block['sweetness_accuracy']} | "
                     f"{block['non_empty_share']} | {block['catalog_similarity_mean']} | "
                     f"{block['latency_p50_ms']} / {block['latency_p95_ms']} мс |")
    lines += ["", "## Вывод", "", f"**{verdict}**", "",
              "## Оговорки", "", f"- {report['caveat']}",
              "- Кириллический движок включается флагом `OCR_ENGINE=cyrillic` (или `fusion`); по умолчанию — `rapidocr`.",
              "- При недоступных весах кириллический движок уступает текущему RapidOCR без ошибок."]
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    """Прогоняет движки на proxy и полевых наборах и пишет отчёт."""
    parser = argparse.ArgumentParser(description="Замер кириллического OCR")
    parser.add_argument("--labels", default=str(_ROOT / "reports" / "real100_labels.json"))
    parser.add_argument("--photos", default=str(_ROOT / "data" / "samples" / "real100"))
    parser.add_argument("--proxy", type=int, default=PROXY_SIZE)
    parser.add_argument("--limit", type=int, default=FIELD_LIMIT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--engines", default=",".join(ENGINES))
    parser.add_argument("--out", default=str(_ROOT / "reports" / "ocr_cyrillic_bench.json"))
    parser.add_argument("--md", default=str(_ROOT / "reports" / "ocr_cyrillic_bench.md"))
    args = parser.parse_args(argv)

    status = ensure_cyrillic_env()
    print(f"кириллический движок: {status}")

    proxy_cases = build_proxy_cases(args.proxy, args.seed)
    references = load_field_labels(args.labels, args.limit)
    if not proxy_cases and not references:
        print("нет данных для замера")
        return 1
    catalog = load_catalog(_ROOT / settings.catalog_jsonl)

    engines = [name.strip() for name in args.engines.split(",") if name.strip()]
    results: dict[str, dict] = {}
    for engine in engines:
        results[engine] = {
            "proxy": bench_proxy(engine, proxy_cases),
            "field": bench_field(engine, references, catalog, Path(args.photos)),
        }
        proxy = results[engine]["proxy"]
        field = results[engine]["field"]
        print(f"  {engine:<10} proxy год {proxy['year_accuracy']} слад {proxy['sweetness_accuracy']} "
              f"CER {proxy['cer_mean']} · поле год {field['year_accuracy']} "
              f"слад {field['sweetness_accuracy']} текст {field['non_empty_share']} · "
              f"p50 {field['latency_p50_ms']} мс")

    report = build_report(results, status, (settings.ocr_engine or "rapidocr"), args.out, args.md)
    print(f"вывод: {report['verdict']}")
    log_marker("M-EVAL", "ocr_cyrillic_bench", "DONE", cyrillic=report["cyrillic_status"][:40],
               verdict=report["verdict"][:80])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
