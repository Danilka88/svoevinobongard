#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Разметка 100 реальных фото организатора моделью 26b и OCR с чекпоинтом, сопоставление с каталогом
# SCOPE: parse_args, load_done, label_one, run, match_catalog, write_reports, main
# DEPENDS: csv, json, numpy, PIL, packages.common.catalog, packages.common.config, packages.common.images,
#          packages.common.logging, packages.common.ocr, packages.common.preprocess, packages.common.text_norm,
#          services.normalize.app.tta, services.vlm.app.extract
# LINKS: M-EVAL, M-VLM, M-DATA, M-TEXT
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# DEFAULT_DIR - каталог скопированных фото
# DEFAULT_LABELS - файл чекпоинта разметки
# STRONG_THRESHOLD - порог для признания совпадения с каталогом
# WEAK_THRESHOLD - минимальный балл для частичного совпадения
# parse_args - аргументы CLI (run или report)
# load_done - уже размеченные файлы из чекпоинта
# label_one - разметка одного фото моделью и OCR
# run - последовательный прогон с чекпоинтом, возобновлением и time-box
# catalog_features - однократная свёртка полей каталога для сравнения
# match_catalog - сопоставление полей с каталогом: strong, weak или none
# write_reports - JSON, CSV и список спорных кейсов
# main - точка входа
# END_MODULE_MAP
"""Разметка реальных фото организатора.

Прогон длинный (100 фото по 20-30 секунд при офлайн-инференсе 26b), поэтому каждый результат
пишется отдельной строкой JSONL с немедленным flush: прерывание не теряет прогресс, повторный
запуск продолжает с места остановки. По истечении time-box прогон останавливается сам и сообщает,
сколько успел обработать.

Имена файлов организатора не несут меток (score_дата_время), поэтому истина не предполагается:
модель читает поля, а сопоставление с каталогом делается отдельным шагом через text_norm
(транслит лат↔кир и алиасы), и каждому фото присваивается признак strong/weak/none.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
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
from packages.common.ocr import parse_fields, run_ocr  # noqa: E402
from packages.common.preprocess import PreprocessConfig  # noqa: E402
from packages.common.text_norm import fold, similarity  # noqa: E402
from services.normalize.app.tta import build_vlm_inputs  # noqa: E402
from services.vlm.app.extract import extract_fields  # noqa: E402

DEFAULT_DIR = "data/samples/real100"
DEFAULT_LABELS = "reports/real100_labels.jsonl"
STRONG_THRESHOLD = 0.86
WEAK_THRESHOLD = 0.62

__all__ = ["DEFAULT_DIR", "DEFAULT_LABELS", "STRONG_THRESHOLD", "WEAK_THRESHOLD",
           "catalog_features", "label_one", "load_done", "main", "match_catalog", "parse_args",
           "run", "write_reports"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы CLI."""
    parser = argparse.ArgumentParser(description="Разметка реальных фото организатора")
    parser.add_argument("mode", nargs="?", default="run", choices=["run", "report"])
    parser.add_argument("--dir", default=str(_ROOT / DEFAULT_DIR))
    parser.add_argument("--labels", default=str(_ROOT / DEFAULT_LABELS))
    parser.add_argument("--out", default=str(_ROOT / "reports" / "real100_labels.json"))
    parser.add_argument("--model", default="gemma4:26b-mlx")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--time-budget-min", type=float, default=75.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--review", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def load_done(labels_path: Path) -> dict[str, dict]:
    """Читает уже размеченные записи чекпоинта по имени файла."""
    done: dict[str, dict] = {}
    if not labels_path.is_file():
        return done
    for line in labels_path.read_text(encoding="utf-8").splitlines():
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


def label_one(path: Path, orig_name: str, model: str, timeout: float) -> dict:
    """Размечает одно фото: 26b по полному кадру и OCR fusion."""
    image = _read_image(path)
    if image is None:
        return {"file": path.name, "orig_name": orig_name, "ok": False, "reason": "unreadable"}
    record: dict = {"file": path.name, "orig_name": orig_name, "ok": True,
                    "vlm": {}, "vlm_latency_ms": 0.0, "ocr_engine": "", "ocr_lines": [],
                    "ocr_text": "", "ocr_vintage": "", "ocr_sweetness": ""}
    try:
        ocr = run_ocr(image, engine="fusion")
        record.update({"ocr_engine": ocr.engine, "ocr_lines": list(ocr.lines),
                       "ocr_text": " | ".join(ocr.lines[:8]),
                       "ocr_vintage": ocr.vintage, "ocr_sweetness": ocr.sweetness})
    except Exception as error:  # noqa: BLE001
        record["ocr_error"] = type(error).__name__
    try:
        images, _ = build_vlm_inputs(image, PreprocessConfig(pas_tiles=settings.pas_tiles))
        started = time.perf_counter()
        fields = extract_fields(images, None, model=model, max_tokens=512, timeout_s=timeout,
                                num_ctx=settings.vlm_num_ctx,
                                reasoning_effort=settings.vlm_reasoning_effort)
        record["vlm_latency_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        if fields.ok:
            parsed = parse_fields([fields.vintage or "", fields.sweetness or ""])
            region = fields.region or ""
            record["vlm"] = {
                "winery": fields.winery or "", "wine_name": fields.wine_name or "",
                "vintage": int(parsed["vintage"]) if parsed["vintage"] else None,
                "grape": fields.grape or "", "sweetness": parsed["sweetness"] or "",
                "region": region, "raw_vintage": fields.vintage or "",
                "raw_sweetness": fields.sweetness or "",
            }
        else:
            record["vlm"] = {"error": getattr(fields, "reason", "vlm-failed")}
    except Exception as error:  # noqa: BLE001
        record["vlm"] = {"error": f"{type(error).__name__}: {error}"[:200]}
    return record


def run(args: argparse.Namespace) -> int:
    """Последовательный прогон с чекпоинтом, возобновлением и time-box."""
    directory = Path(args.dir)
    labels_path = Path(args.labels)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = directory / "manifest.tsv"
    entries: list[tuple[Path, str]] = []
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                name, _, orig = line.partition("\t")
                if (directory / name).is_file():
                    entries.append((directory / name, orig))
    done = load_done(labels_path) if args.resume else {}
    pending = [item for item in entries if item[0].name not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"фото: {len(entries)} · размечено: {len(done)} · к обработке: {len(pending)}")
    budget_s = args.time_budget_min * 60.0
    started = time.perf_counter()
    processed = 0
    with labels_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for path, orig in pending:
            if time.perf_counter() - started > budget_s:
                print(f"TIME-BOX {args.time_budget_min} мин: обработано {processed}, осталось {len(pending) - processed}")
                log_marker("M-EVAL", "label_real100", "TIMEOUT", processed=processed,
                           left=len(pending) - processed)
                break
            record = label_one(path, orig, args.model, args.timeout)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            processed += 1
            fields = record.get("vlm") or {}
            print(f"[{processed}/{len(pending)}] {path.name[:28]:<30} "
                  f"winery={str(fields.get('winery') or '')[:16]:<18} "
                  f"name={str(fields.get('wine_name') or '')[:16]:<18} {record['vlm_latency_ms']/1000:.1f}с", flush=True)
    total = len(load_done(labels_path))
    print(f"\nчекпоинт: {total} из {len(entries)} · файл {labels_path.relative_to(_ROOT)}")
    log_marker("M-EVAL", "label_real100", "DONE", done=total, entries=len(entries))
    return 0


def catalog_features(catalog: dict) -> list[tuple[str, str, str, str, str]]:
    """Готовит свёрнутые поля каталога один раз: сравнение считается по ним."""
    return [(slug, fold(record.get("winery") or ""), fold(record.get("name") or ""),
             fold(record.get("grape") or ""), str(record.get("year") or ""))
            for slug, record in catalog.items()]


def _quick(text: str, other: str) -> float:
    """Дешёвая оценка схожести для отбора кандидатов перед точным сравнением."""
    if not text or not other:
        return 0.0
    return difflib.SequenceMatcher(None, text, other).quick_ratio()


def match_catalog(fields: dict, catalog: dict, features: list | None = None,
                  prefilter: int = 30) -> dict:
    """Сопоставляет поля этикетки с каталогом: strong, weak или none.

    Правило: strong — совпали винодельня И (название ИЛИ сорт) и лучший slug уникален;
    weak — совпала только часть; none — совпадений нет.

    Точное сравнение дорогое, поэтому сначала идёт дешёвый отбор (свёрнутые строки и quick_ratio),
    и только лучшие кандидаты проверяются через similarity.
    """
    winery = str(fields.get("winery") or "")
    name = str(fields.get("wine_name") or "")
    grape = str(fields.get("grape") or "")
    year = str(fields.get("vintage") or "")
    if not any((winery, name, grape)):
        return {"in_catalog": "none", "slug_candidates": [], "note": "нет полей для сопоставления"}
    query = (fold(winery), fold(name), fold(grape))
    if features is None:
        features = catalog_features(catalog)
    cheap: list[tuple[float, str]] = []
    for slug, fwinery, fname, fgrape, fyear in features:
        detail = max(_quick(query[1], fname), _quick(query[2], fgrape))
        cheap.append((0.55 * _quick(query[0], fwinery) + 0.45 * detail, slug))
    cheap.sort(key=lambda item: (-item[0], item[1]))
    scored: list[tuple[float, float, float, str]] = []
    for _, slug in cheap[:prefilter]:
        record = catalog[slug]
        winery_score = similarity(winery, record.get("winery")) if winery else 0.0
        name_score = similarity(name, record.get("name")) if name else 0.0
        grape_score = similarity(grape, record.get("grape")) if grape else 0.0
        detail = max(name_score, grape_score)
        if year and str(record.get("year") or "") == year:
            detail = min(1.0, detail + 0.05)
        combined = 0.55 * winery_score + 0.45 * detail
        if combined > 0.0:
            scored.append((combined, winery_score, detail, slug))
    scored.sort(key=lambda item: (-item[0], item[3]))
    candidates = [{"slug": slug, "score": round(combined, 3), "winery_score": round(winery_score, 3),
                   "detail_score": round(detail, 3)} for combined, winery_score, detail, slug in scored[:3]]
    if not candidates:
        return {"in_catalog": "none", "slug_candidates": [], "note": "совпадений нет"}
    best = scored[0]
    strong = best[1] >= STRONG_THRESHOLD and best[2] >= STRONG_THRESHOLD
    second_close = len(scored) > 1 and scored[1][0] >= best[0] - 0.05 and scored[1][2] >= STRONG_THRESHOLD
    if strong and not second_close:
        level = "strong"
    elif best[0] >= WEAK_THRESHOLD:
        level = "weak"
    else:
        level = "none"
    note = "кандидат однозначен" if level == "strong" else (
        "несколько близких кандидатов" if second_close else "частичное совпадение")
    return {"in_catalog": level, "slug_candidates": candidates, "note": note}


def write_reports(args: argparse.Namespace) -> int:
    """Собирает JSON, CSV и список спорных кейсов."""
    labels = load_done(Path(args.labels))
    catalog = load_catalog(_ROOT / settings.catalog_jsonl)
    features = catalog_features(catalog)
    rows: list[dict] = []
    for record in labels.values():
        fields = record.get("vlm") or {}
        matched = match_catalog(fields, catalog, features)
        top = (matched["slug_candidates"] or [{}])[0]
        rows.append({
            "file": record.get("file", ""),
            "orig_name": record.get("orig_name", ""),
            "winery": str(fields.get("winery") or ""),
            "name": str(fields.get("wine_name") or ""),
            "year": fields.get("vintage"),
            "grape": str(fields.get("grape") or ""),
            "sweetness": str(fields.get("sweetness") or ""),
            "region": str(fields.get("region") or ""),
            "in_catalog": matched["in_catalog"],
            "top_slug": top.get("slug", ""),
            "top_score": top.get("score", 0.0),
            "slug_candidates": matched["slug_candidates"],
            "note": matched["note"],
            "ocr_head": " | ".join((record.get("ocr_lines") or [])[:3])[:120],
            "ocr_vintage": record.get("ocr_vintage", ""),
            "vlm_latency_ms": record.get("vlm_latency_ms", 0.0),
            "vlm_ok": bool(fields) and not fields.get("error"),
        })
    rows.sort(key=lambda row: ({"strong": 0, "weak": 1, "none": 2}[row["in_catalog"]], -float(row["top_score"] or 0)))
    summary = {level: sum(1 for row in rows if row["in_catalog"] == level) for level in ("strong", "weak", "none")}
    with_text = sum(1 for row in rows if row["winery"] or row["name"])
    report = {
        "photos": len(rows), "vlm_ok": sum(1 for row in rows if row["vlm_ok"]),
        "with_any_field": with_text, "summary": summary,
        "avg_vlm_latency_ms": round(sum(float(row["vlm_latency_ms"] or 0) for row in rows) / len(rows), 1) if rows else 0.0,
        "thresholds": {"strong": STRONG_THRESHOLD, "weak": WEAK_THRESHOLD},
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "winery", "name", "year", "grape",
                                                    "sweetness", "region", "in_catalog", "top_slug",
                                                    "top_score", "ocr_head"],
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    # спорные кейсы: слабые и отсутствующие совпадения с уверенной моделью, либо расхождение OCR и модели по году
    contentious = [
        row for row in rows
        if (row["in_catalog"] in {"weak", "none"} and row["vlm_ok"] and (row["winery"] or row["name"]))
        or (row["year"] and row["ocr_vintage"] and str(row["year"]) != str(row["ocr_vintage"]))
    ][: args.review]
    review_path = _ROOT / "reports" / "real100_review.csv"
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "winery", "name", "year", "grape", "in_catalog",
                                                    "top_slug", "top_score", "ocr_vintage", "ocr_head", "note"],
                                extrasaction="ignore")
        writer.writeheader()
        for row in contentious:
            writer.writerow(row)
    print("=== РАЗМЕТКА 100 РЕАЛЬНЫХ ФОТО ===")
    print(f"  фото: {len(rows)} · модель дала поля: {report['vlm_ok']} · с текстом: {with_text}")
    print(f"  средняя задержка 26b: {report['avg_vlm_latency_ms']} мс")
    print(f"  в каталоге: strong {summary['strong']} · weak {summary['weak']} · none {summary['none']}")
    print(f"  файлы: {out.relative_to(_ROOT)} · {csv_path.relative_to(_ROOT)} · {review_path.relative_to(_ROOT)} ({len(contentious)} спорных)")
    log_marker("M-EVAL", "label_real100_report", "DONE", **summary)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Запускает разметку или сборку отчётов."""
    args = parse_args(argv)
    return run(args) if args.mode == "run" else write_reports(args)


if __name__ == "__main__":
    raise SystemExit(main())
