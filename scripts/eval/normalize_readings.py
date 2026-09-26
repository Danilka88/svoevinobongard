#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Офлайн-проход нормализации чтений под каталог (C-077): кэш для замера без конкуренции моделей
# SCOPE: DEFAULT_INPUT, DEFAULT_OUTPUT, DEFAULT_CATALOG, DEFAULT_MODEL, normalize_records, main
# DEPENDS: argparse, json, sys, pathlib, packages.common.catalog, packages.common.llm.normalizer,
#          packages.common.observer
# LINKS: M-EVAL, M-COMMON
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# BASE - корень репозитория для путей по умолчанию
# DEFAULT_INPUT - отчёт LLM-пути с сохранёнными чтениями (reports/llm_match.json)
# DEFAULT_OUTPUT - кэш нормализации (reports/normalize_cache.json)
# DEFAULT_CATALOG - catalog.jsonl источник словаря виноделен/регионов/enum
# DEFAULT_MODEL - модель нормализатора
# normalize_records - нормализует наблюдения записей и возвращает кэш по ключу чтения
# main - CLI: читает отчёт, нормализует каждое чтение, пишет кэш
# END_MODULE_MAP
"""Отдельный проход нормализации чтений под каталог.

Нормализатор (маленькая LLM) прогоняется один раз на сохранённых чтениях, без тяжёлого читателя в
VRAM: это убирает искажение результата от переключения моделей и повторяет прод-поведение с кэшем.
Кэш по ключу сырого чтения затем используется замером и онлайн-путём.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from packages.common.catalog import load_catalog
from packages.common.llm.normalizer import normalize_reading
from packages.common.observer import parse_observation

BASE = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = BASE / "reports" / "llm_match.json"
DEFAULT_OUTPUT = BASE / "reports" / "normalize_cache.json"
DEFAULT_CATALOG = BASE / "data" / "catalog" / "catalog.jsonl"
DEFAULT_MODEL = "gemma4:e4b-mlx"

__all__ = ["DEFAULT_CATALOG", "DEFAULT_INPUT", "DEFAULT_MODEL", "DEFAULT_OUTPUT", "main",
           "normalize_records"]


def normalize_records(records: list[dict], catalog: dict, *, model: str, timeout_s: float = 60.0,
                      num_gpu: int | None = 0) -> dict:
    """Нормализует наблюдение каждого чтения и возвращает кэш «сырое чтение → стандартизация».

    Ключ кэша — строка сырого наблюдения из отчёта, поэтому замер может найти запись без вызова LLM.
    """
    cache: dict[str, dict] = {}
    for index, record in enumerate(records, start=1):
        raw = str(record.get("observation") or "").strip()
        if not raw:
            continue
        observation = parse_observation(raw)
        result = normalize_reading(observation, catalog, model=model, timeout_s=timeout_s,
                                   num_gpu=num_gpu)
        cache[raw] = result
        print(f"  [{index}/{len(records)}] {record.get('file', '')}: "
              f"winery={result.get('winery') or '—'} brand={result.get('brand') or '—'} "
              f"({result.get('source')})", flush=True)
    return cache


def main(argv: list[str] | None = None) -> int:
    """CLI: читает отчёт замеров, нормализует чтения, пишет кэш."""
    parser = argparse.ArgumentParser(description="Нормализация чтений под каталог (C-077)")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)

    source = Path(args.input)
    if not source.is_file():
        print(f"нет отчёта с чтениями: {source}", file=sys.stderr)
        return 2
    payload = json.loads(source.read_text(encoding="utf-8"))
    records = payload.get("results") or []
    if not records:
        print(f"в {source} нет результатов (ключ results)", file=sys.stderr)
        return 2
    catalog = load_catalog(args.catalog)
    cache = normalize_records(records, catalog, model=args.model, timeout_s=args.timeout)

    existing: dict = {}
    output = Path(args.output)
    if output.is_file():
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
        except ValueError:
            existing = {}
    merged = {**existing, **cache}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    llm_count = sum(1 for item in cache.values() if item.get("source") == "llm")
    print(f"  нормализовано: {len(cache)} ({llm_count} через LLM), кэш: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
