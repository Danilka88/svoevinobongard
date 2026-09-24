#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Офлайн-обогащение целевых near-dup slug моделью gemma4:26b с чекпоинтами и возобновлением
# SCOPE: parse_args, target_slugs, load_done, enrich_one, run, merge, main
# DEPENDS: numpy, PIL, packages.common.catalog, packages.common.config, packages.common.images,
#          packages.common.logging, packages.common.ocr, packages.common.preprocess,
#          services.matcher.app.textfusion, services.vlm.app.extract
# LINKS: M-VLM, M-TEXT, M-CATALOG, M-OCR
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# ENRICH_KEYS - поля, которые дозаполняются в объединённом обогащении
# DEFAULT_UPLOADS - каталог эталонных изображений по умолчанию
# parse_args - аргументы CLI (run или merge)
# target_slugs - целевые slug near-dup групп
# load_done - какие slug уже обработаны в чекпоинте
# enrich_one - один вызов 26b по эталонному изображению с нормализацией полей
# run - последовательный прогон с чекпоинтом, возобновлением и time-box
# merge - объединение OCR-обогащения C-013 с VLM-полями (приоритет у 26b)
# main - точка входа
# END_MODULE_MAP
"""Офлайн-обогащение метаданных через gemma4:26b.

Прогон длинный (228 slug x ~22 с ≈ 1.4 ч), поэтому результат пишется
инкрементально в JSONL-чекпоинт и прогон можно продолжить флагом ``--resume``.
Исходный ``catalog.jsonl`` не изменяется — это проверяется сравнением байт.

При объединении с OCR-обогащением C-013 приоритет у VLM для ``year``/``sweetness``
(26b даёт vintage acc 0.75 против 0.333 у OCR), а источник каждого поля
сохраняется в ``sources`` для трассируемости.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from packages.common.catalog import load_catalog  # noqa: E402
from packages.common.config import settings  # noqa: E402
from packages.common.images import flatten_alpha  # noqa: E402
from packages.common.logging import log_marker  # noqa: E402
from packages.common.ocr import parse_fields  # noqa: E402
from packages.common.preprocess import PreprocessConfig, build_vlm_inputs  # noqa: E402
from services.matcher.app.textfusion import near_dup_key  # noqa: E402
from services.vlm.app.extract import extract_fields  # noqa: E402

ENRICH_KEYS = ("year", "sweetness", "grape")
DEFAULT_UPLOADS = (
    _ROOT.parent / "материалы для конкурса" / "Датасет"
    / "prod-svoe-vino-strapi" / "prod-svoe-vino" / "strapi" / "uploads"
)

__all__ = ["ENRICH_KEYS", "target_slugs", "load_done", "enrich_one", "run", "merge", "main"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы CLI."""
    parser = argparse.ArgumentParser(description="Офлайн-обогащение через gemma4:26b")
    parser.add_argument("command", choices=["run", "merge"])
    parser.add_argument("--model", default="gemma4:26b-mlx")
    parser.add_argument("--checkpoint", default="data/catalog/catalog_vlm_enriched.jsonl")
    parser.add_argument("--ocr-enriched", default="data/catalog/catalog_enriched.jsonl")
    parser.add_argument("--merged-out", default="data/catalog/catalog_enriched_v2.jsonl")
    parser.add_argument("--uploads", default=str(DEFAULT_UPLOADS))
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--time-budget-min", type=float, default=110.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report", default="")
    return parser.parse_args(argv)


def target_slugs(catalog: dict) -> list[str]:
    """Целевые slug — члены near-dup групп (как в C-013/C-014)."""
    groups: dict[str, list[str]] = {}
    for slug, record in catalog.items():
        groups.setdefault(near_dup_key(record), []).append(slug)
    return sorted({slug for slugs in groups.values() if len(slugs) > 1 for slug in slugs})


def load_done(checkpoint: Path) -> dict[str, dict]:
    """Читает уже обработанные записи чекпоинта (для возобновления)."""
    done: dict[str, dict] = {}
    if not checkpoint.is_file():
        return done
    with checkpoint.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("slug"):
                done[str(record["slug"])] = record
    return done


def enrich_one(image: np.ndarray, model: str, timeout: float) -> tuple[dict, float, str]:
    """Один вызов 26b: возвращает (поля, latency_ms, сырой ответ)."""
    images, _ = build_vlm_inputs(image, PreprocessConfig(pas_tiles=settings.pas_tiles))
    started = time.perf_counter()
    result = extract_fields(
        images, None, model=model, max_tokens=512, timeout_s=timeout,
        num_ctx=settings.vlm_num_ctx, reasoning_effort=settings.vlm_reasoning_effort,
    )
    latency = (time.perf_counter() - started) * 1000.0
    raw = f"winery={result.winery!r} name={result.wine_name!r} vintage={result.vintage!r} " \
          f"grape={result.grape!r} sweetness={result.sweetness!r}"
    if not result.ok:
        return {}, latency, f"FAILED {result.reason if hasattr(result, 'reason') else ''} {raw}"[:300]
    parsed = parse_fields([result.vintage or "", result.sweetness or "", result.wine_name or ""])
    fields: dict = {}
    if parsed["vintage"]:
        fields["year"] = int(parsed["vintage"])
    if parsed["sweetness"]:
        fields["sweetness"] = parsed["sweetness"]
    grape = " ".join(str(result.grape or "").split())
    if grape and len(grape) > 2:
        fields["grape"] = grape
    return fields, latency, raw


def run(args: argparse.Namespace) -> int:
    """Последовательный прогон с чекпоинтом, возобновлением и time-box."""
    catalog_path = _ROOT / settings.catalog_jsonl
    before = catalog_path.read_bytes()
    catalog = load_catalog(catalog_path)
    uploads = Path(args.uploads)
    checkpoint = _ROOT / args.checkpoint
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    targets = [slug for slug in target_slugs(catalog) if (uploads / (catalog[slug].get("image") or "")).is_file()]
    done = load_done(checkpoint) if args.resume else {}
    if not args.resume and done:
        log_marker("M-VLM", "enrich26b", "IGNORE_CHECKPOINT", slugs=len(done))
    pending = [slug for slug in targets if slug not in done]
    if args.limit:
        pending = pending[: args.limit]
    budget_s = args.time_budget_min * 60.0
    started = time.perf_counter()
    log_marker("M-VLM", "enrich26b", "START", targets=len(targets), pending=len(pending),
               resumed=len(done), model=args.model, budget_min=args.time_budget_min)

    processed = 0
    with checkpoint.open("a", encoding="utf-8") as handle:
        for slug in pending:
            elapsed = time.perf_counter() - started
            if elapsed > budget_s:
                print(f"TIME-BOX {args.time_budget_min} мин исчерпан: обработано {processed}, осталось {len(pending) - processed}")
                log_marker("M-VLM", "enrich26b", "TIMEOUT", processed=processed,
                           left=len(pending) - processed)
                break
            path = uploads / (catalog[slug].get("image") or "")
            try:
                with Image.open(path) as raw_image:
                    rgba = np.asarray(raw_image.convert("RGBA"))
                image = flatten_alpha(rgba[:, :, [2, 1, 0, 3]])
                fields, latency, raw = enrich_one(image, args.model, args.timeout)
                ok = bool(fields)
            except Exception as error:  # noqa: BLE001 - один сбойный slug не должен валить прогон
                fields, latency, raw, ok = {}, 0.0, f"EXC {type(error).__name__}: {error}"[:300], False
            record = {
                "slug": slug, "ok": ok, "fields": fields, "latency_ms": round(latency, 1),
                "model": args.model, "source": "vlm-26b", "raw": raw,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            processed += 1
            if processed % 5 == 0 or processed == 1:
                rate = elapsed / max(processed, 1)
                left = len(pending) - processed
                print(f"[{processed}/{len(pending)}] {slug[:34]:<34} fields={fields} "
                      f"{latency/1000:.1f}с ETA {left * rate / 60:.0f} мин", flush=True)
            log_marker("M-VLM", "enrich26b", "SLUG", slug=slug, ok=ok, fields=len(fields),
                       latency_ms=round(latency, 1))

    if catalog_path.read_bytes() != before:
        print("ERROR: catalog.jsonl изменён — это запрещено", file=sys.stderr)
        return 3
    total_done = len(load_done(checkpoint))
    print(f"\nчекпоинт: {total_done}/{len(targets)} slug · файл {checkpoint.relative_to(_ROOT)}")
    print("catalog.jsonl не изменён: True")
    log_marker("M-VLM", "enrich26b", "DONE", done=total_done, targets=len(targets))
    return 0


def merge(args: argparse.Namespace) -> int:
    """Объединяет OCR-обогащение C-013 с VLM-полями: для year/sweetness приоритет у 26b."""
    ocr_path = _ROOT / args.ocr_enriched
    vlm_path = _ROOT / args.checkpoint
    out_path = _ROOT / args.merged_out
    ocr_rows = [json.loads(line) for line in ocr_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    vlm_rows = load_done(vlm_path)

    merged: list[dict] = []
    stats = {"vlm_year": 0, "ocr_year": 0, "vlm_sweetness": 0, "ocr_sweetness": 0,
             "vlm_grape": 0, "overrides": 0, "slugs": 0}
    for row in ocr_rows:
        slug = str(row.get("slug") or "")
        ocr_fields = dict(row.get("enriched") or {})
        vlm_fields = dict((vlm_rows.get(slug) or {}).get("fields") or {})
        sources: dict[str, str] = {}
        fields: dict = {}
        for key in ENRICH_KEYS:
            ocr_value, vlm_value = ocr_fields.get(key), vlm_fields.get(key)
            if vlm_value and ocr_value and vlm_value != ocr_value:
                stats["overrides"] += 1
            if vlm_value:
                fields[key] = vlm_value
                sources[key] = "vlm-26b"
            elif ocr_value:
                fields[key] = ocr_value
                sources[key] = "ocr-c013"
            if key == "year":
                stats["vlm_year"] += bool(vlm_fields.get(key))
                stats["ocr_year"] += bool(ocr_value) and not vlm_fields.get(key)
            if key == "sweetness":
                stats["vlm_sweetness"] += bool(vlm_fields.get(key))
                stats["ocr_sweetness"] += bool(ocr_value) and not vlm_fields.get(key)
            if key == "grape":
                stats["vlm_grape"] += bool(vlm_fields.get(key))
        stats["slugs"] += bool(fields)
        record = dict(row)
        record["enriched"] = fields
        record["sources"] = sources
        merged.append(record)

    with out_path.open("w", encoding="utf-8") as handle:
        for record in merged:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = {
        "phase": "merge", "slugs": len(merged), "slug_with_fields": stats["slugs"],
        "by_source": {"year": {"vlm": stats["vlm_year"], "ocr": stats["ocr_year"]},
                      "sweetness": {"vlm": stats["vlm_sweetness"], "ocr": stats["ocr_sweetness"]},
                      "grape": {"vlm": stats["vlm_grape"]}},
        "conflicts_vlm_won": stats["overrides"], "out": str(out_path.relative_to(_ROOT)),
    }
    if args.report:
        target = _ROOT / args.report
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log_marker("M-TEXT", "merge26b", "DONE", slugs=len(merged), with_fields=stats["slugs"],
               overrides=stats["overrides"])
    print("=== ОБЪЕДИНЕНИЕ ОБОГАЩЕНИЙ ===")
    print(f"slug всего: {len(merged)} · с заполненными полями: {stats['slugs']}")
    print(f"year: 26b {stats['vlm_year']} · OCR {stats['ocr_year']}")
    print(f"sweetness: 26b {stats['vlm_sweetness']} · OCR {stats['ocr_sweetness']}")
    print(f"grape: 26b {stats['vlm_grape']} · конфликтов в пользу 26b: {stats['overrides']}")
    print(f"out: {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Запускает прогон или объединение."""
    args = parse_args(argv)
    return run(args) if args.command == "run" else merge(args)


if __name__ == "__main__":
    raise SystemExit(main())
