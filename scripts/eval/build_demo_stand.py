#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Генерация демо-данных стенда «Примеры работы» из отчётов (21 фото) одним прогоном
# SCOPE: DEFAULT_LLM_MATCH, DEFAULT_GOLD, DEFAULT_FIELD_DIR, DEFAULT_OUT, DEFAULT_PUBLIC_DIR, LABELS,
#        scenario_of, copy_photos, build_samples, main
# DEPENDS: argparse, json, shutil, sys, pathlib
# LINKS: M-EVAL, M-WEB
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# BASE - корень репозитория
# DEFAULT_LLM_MATCH - отчёт LLM-пути (чтение/нормализация/кандидаты/итог)
# DEFAULT_GOLD - эталон field21_gold (статус и ожидаемый slug)
# DEFAULT_FIELD_DIR - каталог 21 проверенного полевого фото
# DEFAULT_OUT - JSON стенда
# DEFAULT_PUBLIC_DIR - каталог копий фото для веба (public/demo/img)
# LABELS - человекочитаемые названия полей наблюдения/нормализации
# scenario_of - сценарий примера по статусу эталона
# copy_photos - копирует фото стенда в public и возвращает карту имён
# build_samples - собирает список примеров стенда из отчёта и эталона
# main - CLI: копирует фото и пишет samples.json
# END_MODULE_MAP
"""Генератор демо-стенда «Примеры работы».

Собирает СТАТИЧНЫЙ JSON для отдельной вкладки: судьи видят, как система работала на 21 проверенном
полевом фото — что прочитала модель, как нормализовала под каталог, какие кандидаты рассмотрела и чем
всё закончилось. Данные берутся из уже собранных отчётов (reports/llm_match.json, field21_gold.json),
поэтому система НЕ запускается: стенд открывается мгновенно и не требует пересчёта.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
DEFAULT_LLM_MATCH = BASE / "reports" / "llm_match.json"
DEFAULT_GOLD = BASE / "reports" / "field21_gold.json"
DEFAULT_FIELD_DIR = BASE / "data" / "samples" / "real_field"
DEFAULT_OUT = BASE / "apps" / "web" / "public" / "demo" / "samples.json"
DEFAULT_PUBLIC_DIR = BASE / "apps" / "web" / "public" / "demo" / "img"

LABELS = {"winery": "Винодельня", "brand": "Бренд", "wine_name": "Название", "vintage": "Год",
          "grape": "Сорт", "sweetness": "Сладость", "color": "Цвет", "wine_type": "Тип",
          "region": "Регион"}

__all__ = ["DEFAULT_FIELD_DIR", "DEFAULT_GOLD", "DEFAULT_LLM_MATCH", "DEFAULT_OUT", "DEFAULT_PUBLIC_DIR",
           "LABELS", "build_samples", "copy_photos", "main", "scenario_of"]


def scenario_of(status: str) -> str:
    """Сценарий примера по статусу эталона: однозначное / серия / вне каталога."""
    return {"matched": "matched", "ambiguous": "ambiguous", "not_found": "not_found"}.get(
        str(status or ""), "not_found")


def _ascii(name: str) -> str:
    """Безопасное имя файла для веба: не-ASCII → простой индексный суффикс."""
    return name


def copy_photos(field_dir: Path, public_dir: Path, files: list[str]) -> dict[str, str]:
    """Копирует фото стенда в public и возвращает карту «исходное имя → публичный путь».

    Имена не транслитерируем (кириллица в URL допустима), но каждый файл копируется под своим именем
    с префиксом ``img/``. Возвращает относительные пути для веба.
    """
    public_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    for index, name in enumerate(files, start=1):
        source = field_dir / name
        if not source.is_file():
            continue
        # Индексный префикс гарантирует уникальность и корректность URL для любых имён (в т.ч. кириллицы).
        suffix = source.suffix.lower() or ".jpg"
        target_name = f"{index:02d}{suffix}"
        shutil.copyfile(source, public_dir / target_name)
        mapping[name] = f"img/{target_name}"
    return mapping


def _clean_fields(source: dict) -> list[dict]:
    """Поля наблюдения/нормализации в вид «подпись → значение» без пустых."""
    rows: list[dict] = []
    for key, label in LABELS.items():
        value = (source or {}).get(key)
        if value is None or str(value).strip() == "":
            continue
        rows.append({"label": label, "value": str(value)})
    return rows


def build_samples(llm_match: dict, gold: dict, photo_map: dict[str, str]) -> list[dict]:
    """Собирает примеры стенда: фото, чтение, нормализация, кандидаты, итог, путь, latency."""
    gold_files = gold.get("files") or {}
    samples: list[dict] = []
    for result in llm_match.get("results") or []:
        name = str(result.get("file") or "")
        record = gold_files.get(name) or {}
        normalized = result.get("normalized") or {}
        expected = record.get("expected_slug") or ((record.get("candidates") or [{}])[0].get("slug")
                                                   if record.get("candidates") else None)
        samples.append({
            "file": name,
            "photo": photo_map.get(name, ""),
            "scenario": scenario_of(record.get("status")),
            "status": record.get("status"),
            "human_label": record.get("human_label", ""),
            "expected_slug": expected,
            "reading": str(result.get("observation") or ""),
            "reading_fields": _clean_fields(result.get("fields") or {}),
            "normalized": _clean_fields(normalized),
            "normalized_source": normalized.get("source", ""),
            "candidates": list(result.get("visual_candidates") or [])[:5],
            "text_candidates": list(result.get("text_candidates") or [])[:5],
            "top1": result.get("top1"),
            "not_found": bool(result.get("not_found")),
            "path": result.get("path", ""),
            "latency_ms": result.get("latency_ms", 0),
            "reader_ms": result.get("reader_ms", 0),
        })
    return samples


def main(argv: list[str] | None = None) -> int:
    """CLI: копирует фото стенда и пишет samples.json."""
    parser = argparse.ArgumentParser(description="Генерация демо-стенда (C-092)")
    parser.add_argument("--llm-match", default=str(DEFAULT_LLM_MATCH))
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--field-dir", default=str(DEFAULT_FIELD_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--public-dir", default=str(DEFAULT_PUBLIC_DIR))
    args = parser.parse_args(argv)

    match_path, gold_path = Path(args.llm_match), Path(args.gold)
    if not match_path.is_file() or not gold_path.is_file():
        print("нужны reports/llm_match.json и reports/field21_gold.json", file=sys.stderr)
        return 2
    llm_match = json.loads(match_path.read_text(encoding="utf-8"))
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    files = [str(r.get("file") or "") for r in (llm_match.get("results") or [])]
    photo_map = copy_photos(Path(args.field_dir), Path(args.public_dir), files)
    samples = build_samples(llm_match, gold, photo_map)

    from collections import Counter
    counts = Counter(sample["scenario"] for sample in samples)
    payload = {"generated": True, "total": len(samples), "counts": dict(counts),
               "metrics": llm_match.get("metrics", {}), "samples": samples}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  примеров: {len(samples)} ({dict(counts)}); фото скопировано: {len(photo_map)}")
    print(f"  стенд: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
