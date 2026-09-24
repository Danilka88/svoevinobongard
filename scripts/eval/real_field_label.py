#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Псевдо-разметка реальных полевых фото локальными моделями (OCR fusion + gemma4:26b)
# SCOPE: parse_args, load_manifest, label_one, main
# DEPENDS: numpy, PIL, packages.common.images, packages.common.logging, packages.common.ocr,
#          packages.common.preprocess, services.vlm.app.extract
# LINKS: M-EVAL, M-VLM, M-DATA
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# DEFAULT_DIR - каталог реальных фото по умолчанию
# DEFAULT_OUT - файл кэша разметки
# parse_args - аргументы CLI (каталог, кэш, модель, лимит, time-box)
# load_manifest - читает manifest.tsv с текстом этикетки из имени файла
# load_done - уже размеченные файлы из кэша (возобновление)
# label_one - OCR fusion и 26b по одному фото
# main - последовательная разметка с чекпоинтом и time-box
# END_MODULE_MAP
"""Псевдо-разметка реальных фото.

Реальные фото используются только для теста, поэтому разметка делается локально и
кэшируется: OCR-движок fusion читает латиницу с цифрами и кириллицу, а gemma4:26b
даёт структурные поля. Результат — JSONL по одному фото на строку, поэтому прогон
переживает прерывание и продолжается флагом --resume.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from packages.common.config import settings  # noqa: E402
from packages.common.images import flatten_alpha  # noqa: E402
from packages.common.logging import log_marker  # noqa: E402
from packages.common.ocr import parse_fields, run_ocr  # noqa: E402
from packages.common.preprocess import PreprocessConfig, build_vlm_inputs  # noqa: E402
from services.vlm.app.extract import extract_fields  # noqa: E402

DEFAULT_DIR = "data/samples/real_field"
DEFAULT_OUT = "reports/real_field_labels.jsonl"

__all__ = ["DEFAULT_DIR", "DEFAULT_OUT", "label_one", "load_done", "load_manifest", "main", "parse_args"]


def _display(path: Path) -> str:
    """Путь для печати: относительный, если внутри проекта, иначе как есть."""
    try:
        return str(path.resolve().relative_to(_ROOT))
    except ValueError:
        return str(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы CLI."""
    parser = argparse.ArgumentParser(description="Псевдо-разметка реальных фото")
    parser.add_argument("--dir", default=str(_ROOT / DEFAULT_DIR))
    parser.add_argument("--out", default=str(_ROOT / DEFAULT_OUT))
    parser.add_argument("--model", default="gemma4:26b-mlx")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--time-budget-min", type=float, default=40.0)
    parser.add_argument("--no-vlm", action="store_true", help="только OCR, без 26b")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def load_manifest(directory: Path) -> list[tuple[Path, str]]:
    """Читает manifest.tsv: файл и текст этикетки из имени (слабая GT)."""
    manifest = directory / "manifest.tsv"
    rows: list[tuple[Path, str]] = []
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            name, _, label = line.partition("\t")
            path = directory / name
            if path.is_file():
                rows.append((path, label.strip()))
    if not rows:  # фолбэк: имя файла как текст этикетки
        for path in sorted(directory.glob("*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                rows.append((path, path.stem.replace("_", " ").strip()))
    return rows


def load_done(out_path: Path) -> dict[str, dict]:
    """Возвращает уже размеченные записи кэша по имени файла."""
    done: dict[str, dict] = {}
    if not out_path.is_file():
        return done
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("file"):
            done[str(record["file"])] = record
    return done


def _read_image(path: Path) -> np.ndarray | None:
    """Читает фото как BGR-кадр целиком."""
    try:
        with Image.open(path) as raw:
            rgba = np.asarray(raw.convert("RGBA"))
    except Exception:  # noqa: BLE001
        return None
    return flatten_alpha(rgba[:, :, [2, 1, 0, 3]])


def label_one(path: Path, text: str, model: str, timeout: float, use_vlm: bool) -> dict:
    """Размечает одно фото: OCR fusion и (опционально) 26b."""
    image = _read_image(path)
    if image is None:
        return {"file": path.name, "label_text": text, "ok": False, "reason": "unreadable-image"}
    started = time.perf_counter()
    ocr = run_ocr(image, engine="fusion")
    ocr_ms = (time.perf_counter() - started) * 1000.0
    record: dict = {
        "file": path.name,
        "label_text": text,
        "ok": True,
        "ocr_engine": ocr.engine,
        "ocr_lines": list(ocr.lines),
        "ocr_text": " | ".join(ocr.lines[:6]),
        "ocr_vintage": ocr.vintage,
        "ocr_sweetness": ocr.sweetness,
        "ocr_latency_ms": round(ocr_ms, 1),
        "vlm": {},
        "vlm_latency_ms": 0.0,
    }
    if not use_vlm:
        return record
    try:
        images, _ = build_vlm_inputs(image, PreprocessConfig(pas_tiles=settings.pas_tiles))
        started = time.perf_counter()
        fields = extract_fields(images, None, model=model, max_tokens=512, timeout_s=timeout,
                                num_ctx=settings.vlm_num_ctx,
                                reasoning_effort=settings.vlm_reasoning_effort)
        record["vlm_latency_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        if fields.ok:
            parsed = parse_fields([fields.vintage or "", fields.sweetness or ""])
            record["vlm"] = {
                "winery": fields.winery or "", "name": fields.wine_name or "",
                "year": int(parsed["vintage"]) if parsed["vintage"] else None,
                "sweetness": parsed["sweetness"] or "", "grape": fields.grape or "",
                "region": fields.region or "",
                "raw_vintage": fields.vintage or "", "raw_sweetness": fields.sweetness or "",
            }
        else:
            record["vlm"] = {"error": getattr(fields, "reason", "vlm-failed")}
    except Exception as error:  # noqa: BLE001 - сбой модели не должен валить разметку
        record["vlm"] = {"error": f"{type(error).__name__}: {error}"[:200]}
    return record


def main(argv: list[str] | None = None) -> int:
    """Размечает фото последовательно и пишет кэш после каждой записи."""
    args = parse_args(argv)
    directory = Path(args.dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(directory)
    done = load_done(out_path) if args.resume else {}
    pending = [(path, text) for path, text in rows if path.name not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"фото: {len(rows)} · уже размечено: {len(done)} · к обработке: {len(pending)}")
    budget_s = args.time_budget_min * 60.0
    started = time.perf_counter()
    processed = 0
    with out_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for path, text in pending:
            if time.perf_counter() - started > budget_s:
                print(f"TIME-BOX {args.time_budget_min} мин: обработано {processed}, осталось {len(pending) - processed}")
                break
            record = label_one(path, text, args.model, args.timeout, not args.no_vlm)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            processed += 1
            vlm = record.get("vlm") or {}
            print(f"[{processed}/{len(pending)}] {path.name[:34]:<36} ocr={len(record.get('ocr_lines') or [])} строк "
                  f"vlm={'ok' if vlm and not vlm.get('error') else '—'} {record['vlm_latency_ms']/1000:.1f}с", flush=True)
            log_marker("M-EVAL", "real_field_label", "FILE", file=path.name, ocr_lines=len(record.get("ocr_lines") or []),
                       vlm=bool(vlm and not vlm.get("error")))
    total = len(load_done(out_path))
    print(f"\nкэш: {total} из {len(rows)} фото · файл {_display(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
