#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Мультиридерная разметка реальных фото с консенсусом судьи, датасетом и аугментацией
# SCOPE: parse_args, available_models, read_vlm, read_ocr, build_readings, judge, run, dataset,
#        report, main
# DEPENDS: csv, json, subprocess, time, numpy, PIL, packages.common.annotation, packages.common.catalog,
#          packages.common.images, packages.common.llm.structured, packages.common.logging,
#          packages.common.ocr, packages.common.preprocess, packages.common.text_norm,
#          scripts.eval.augment, scripts.eval.label_real_photos, services.normalize.app.tta,
#          services.vlm.app.extract
# LINKS: M-EVAL, M-DATA
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# PRIMARY_MODEL - сильная VLM-модель ридера R1
# SMALL_MODEL - слабая VLM-модель ридера R3
# FALLBACK_SMALL_MODEL - запасная слабая модель, если основной нет в реестре
# QWEN_MODEL - четвёртый ридер: компактная мультимодальная модель
# DEFAULT_LABELS - чекпоинт прогона
# DEFAULT_CONSENSUS - сводка консенсуса
# DEFAULT_DATASET - принятые метки
# DEFAULT_ESCALATIONS - очередь ручной проверки
# DEFAULT_AUGMENTED - синтетические варианты
# parse_args - аргументы CLI
# available_models - список моделей в локальном реестре
# read_vlm - показания VLM-ридера
# read_ocr - показания OCR-ридера
# build_readings - показания всех ридеров для одного фото
# judge - вердикт судьи по показаниям и каталогу
# run - прогон с чекпоинтом, возобновлением и тайм-боксом
# dataset - сборка датасета консенсуса, эскалаций и аугментаций
# report - отчёт по результатам прогона
# main - точка входа
# END_MODULE_MAP
"""Мультиридерная разметка: три ридера, судья по каталогу и датасет с защитой от утечки.

R1 — сильная VLM (gemma4:26b-mlx), R2 — OCR-fusion, R3 — слабая VLM (gemma4:e4b-mlx, при
отсутствии — gemma4:12b-mlx). Судья сверяет показания с каталогом и принимает метку только при
согласии и подтверждении; расхождения уходят в ручную очередь с историей спора.

Прогон длинный (десятки секунд на фото), поэтому результат пишется построчно с возобновлением и
ограничением времени: прерывание не теряет прогресс.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from packages.common.annotation import adjudicate, assign_split, judge_payload, split_policy  # noqa: E402
from packages.common.catalog import load_catalog  # noqa: E402
from packages.common.config import settings  # noqa: E402
from packages.common.images import read_bgr  # noqa: E402
from packages.common.llm.structured import parse_structured  # noqa: E402
from packages.common.logging import log_marker  # noqa: E402
from packages.common.ocr import run_ocr  # noqa: E402
from packages.common.preprocess import PreprocessConfig  # noqa: E402
from scripts.eval.label_real_photos import catalog_features, match_catalog  # noqa: E402
from services.normalize.app.tta import build_vlm_inputs  # noqa: E402
from services.vlm.app.extract import extract_fields  # noqa: E402

PRIMARY_MODEL = "gemma4:26b-mlx"
SMALL_MODEL = "gemma4:e4b-mlx"
FALLBACK_SMALL_MODEL = "gemma4:12b-mlx"
QWEN_MODEL = "qwen3-vl:2b"
DEFAULT_LABELS = "reports/real100_multilabel.jsonl"
DEFAULT_CONSENSUS = "reports/real100_consensus.json"
DEFAULT_DATASET = "data/annotation/consensus.jsonl"
DEFAULT_ESCALATIONS = "data/annotation/escalations.jsonl"
DEFAULT_AUGMENTED = "data/annotation/augmented.jsonl"

__all__ = ["PRIMARY_MODEL", "SMALL_MODEL", "FALLBACK_SMALL_MODEL", "QWEN_MODEL", "DEFAULT_LABELS",
           "available_models", "build_readings", "dataset", "judge", "main", "parse_args",
           "read_ocr", "read_vlm", "report", "run"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы CLI."""
    parser = argparse.ArgumentParser(description="Мультиридерная разметка реальных фото")
    parser.add_argument("mode", nargs="?", default="run", choices=["run", "dataset", "report"])
    parser.add_argument("--dir", default=str(_ROOT / "data" / "samples" / "real100"))
    parser.add_argument("--labels", default=str(_ROOT / DEFAULT_LABELS))
    parser.add_argument("--consensus", default=str(_ROOT / DEFAULT_CONSENSUS))
    parser.add_argument("--dataset", default=str(_ROOT / DEFAULT_DATASET))
    parser.add_argument("--escalations", default=str(_ROOT / DEFAULT_ESCALATIONS))
    parser.add_argument("--augmented", default=str(_ROOT / DEFAULT_AUGMENTED))
    parser.add_argument("--report", default=str(_ROOT / "reports" / "multilabel_report.md"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--time-budget-min", type=float, default=60.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--variants", type=int, default=3, help="сколько аугментаций на принятое фото")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def available_models() -> set[str]:
    """Читает список моделей из локального реестра."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=30)  # noqa: S603, S607
        return {line.split()[0] for line in result.stdout.splitlines()[1:] if line.strip()}
    except Exception:  # noqa: BLE001 - без реестра работаем по факту ответа модели
        return set()




def _read_rgba(path: Path) -> np.ndarray | None:
    """Читает фото как RGBA-кадр: аугментация работает с четырьмя каналами."""
    try:
        with Image.open(path) as raw:
            return np.asarray(raw.convert("RGBA"))
    except Exception:  # noqa: BLE001
        return None


def read_vlm(image: np.ndarray, model: str, timeout: float) -> dict:
    """Показания VLM-ридера: поля этикетки через структурный разбор."""
    try:
        images, _ = build_vlm_inputs(image, PreprocessConfig(pas_tiles=settings.pas_tiles))
        fields = extract_fields(images, None, model=model, max_tokens=512, timeout_s=timeout,
                                num_ctx=settings.vlm_num_ctx,
                                reasoning_effort=settings.vlm_reasoning_effort)
    except Exception as error:  # noqa: BLE001 - отказ ридера не должен ронять прогон
        return {"ok": False, "model": model, "error": type(error).__name__}
    if not fields.ok:
        return {"ok": False, "model": model, "error": getattr(fields, "reason", "vlm failed")}
    return {"ok": True, "model": model, "winery": fields.winery or "",
            "wine_name": fields.wine_name or "", "vintage": fields.vintage or "",
            "grape": fields.grape or "", "sweetness": fields.sweetness or "",
            "region": fields.region or ""}


def read_ocr(image: np.ndarray, engine: str = "fusion") -> dict:
    """Показания OCR-ридера: текст, год и сладость."""
    try:
        result = run_ocr(image, engine=engine)
    except Exception as error:  # noqa: BLE001
        return {"ok": False, "engine": engine, "error": type(error).__name__}
    return {"ok": bool(result.ok), "engine": result.engine, "lines": list(result.lines or []),
            "text": " | ".join((result.lines or [])[:8]), "vintage": result.vintage or "",
            "sweetness": result.sweetness or ""}


def build_readings(image: np.ndarray, models: dict[str, str], timeout: float) -> dict:
    """Собирает показания всех трёх ридеров для одного фото."""
    return {"vlm_primary": read_vlm(image, models["primary"], timeout),
            "ocr_fusion": read_ocr(image),
            "vlm_qwen": read_vlm(image, models["qwen"], timeout),
            "vlm_small": read_vlm(image, models["small"], timeout)}


def judge(readings: dict, catalog: dict, features: list, *, use_paddle: bool = False,
          image: np.ndarray | None = None) -> dict:
    """Выносит вердикт: сверяет показания ридеров с каталогом.

    Для расхождений допускается дополнительный сигнал — PaddleOCR-VL, только на спорных фото.
    """
    primary = readings.get("vlm_primary") or {}
    payload = {key: primary.get(key) for key in ("winery", "wine_name", "vintage", "grape", "sweetness")}
    matched = match_catalog(payload, catalog, features)
    candidates = [{"record": catalog.get(item["slug"], {}), **item}
                  for item in (matched.get("slug_candidates") or [])]
    decision = adjudicate({reader: data for reader, data in readings.items() if data},
                          candidates)
    decision["in_catalog"] = matched.get("in_catalog", "none")
    decision["catalog_candidates"] = [item["slug"] for item in candidates]
    if use_paddle and image is not None:
        decision["paddle_signal"] = _paddle_signal(image)
    judged = parse_structured(json.dumps(judge_payload(decision), ensure_ascii=False), "judge")
    decision["judge_valid"] = judged["valid"]
    decision["judge_errors"] = judged["errors"]
    return decision


def _paddle_signal(image: np.ndarray) -> dict:
    """Дополнительный сигнал для спорных фото: PaddleOCR-VL как ещё один ридер."""
    try:
        from packages.common.ocr_vl import run_ocr_vl  # noqa: PLC0415 - тяжёлый импорт только для спорных фото
        result = run_ocr_vl(image)
        return {"ok": True, "text": " | ".join(list(getattr(result, "lines", []) or [])[:6])}
    except Exception as error:  # noqa: BLE001
        return {"ok": False, "error": type(error).__name__}


def _call_augment(module, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Вызывает аугментацию по фактической сигнатуре модуля.

    Сигнатура модуля аугментации может меняться, поэтому параметры подбираются по именам:
    генератор, вид фона и размер полотна берутся из самого модуля, а не угадываются.
    """
    import inspect  # noqa: PLC0415 - нужен только при сборке датасета

    function = getattr(module, "field_augment", None) or getattr(module, "random_background")
    parameters = inspect.signature(function).parameters
    kwargs: dict = {}
    filled_image = False
    for index, name in enumerate(parameters):
        lowered = name.lower()
        if "rng" in lowered or "random" in lowered or "seed" in lowered:
            kwargs[name] = rng
        elif any(hint in lowered for hint in ("image", "frame", "source", "reference", "rgba", "crop")):
            kwargs[name] = image
            filled_image = True
        elif "kind" in lowered or "background" in lowered:
            kwargs[name] = module.BACKGROUND_KINDS[0]
    if not filled_image:  # имя параметра неизвестно: первый аргумент всегда исходный кадр
        kwargs[list(parameters)[0]] = image
    return function(**kwargs)


def _load_done(path: Path) -> dict[str, dict]:
    """Читает уже обработанные записи чекпоинта."""
    done: dict[str, dict] = {}
    if not path.is_file():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("file"):
                done[str(record["file"])] = record
    return done


def run(args: argparse.Namespace) -> int:
    """Прогон разметки с чекпоинтом, возобновлением и тайм-боксом."""
    directory = Path(args.dir)
    labels_path = Path(args.labels)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    photos = sorted((path for path in directory.iterdir()
                     if path.suffix.lower() in {".webp", ".jpg", ".jpeg", ".png"}), key=lambda p: p.name)
    registry = available_models()
    models = {"primary": PRIMARY_MODEL,
              "qwen": QWEN_MODEL,
              "small": SMALL_MODEL if SMALL_MODEL in registry else FALLBACK_SMALL_MODEL}
    print(f"  модели: R1={models['primary']} · R2=OCR fusion · R3={models['qwen']} · R4={models['small']}"
          f"{'' if SMALL_MODEL in registry else ' (запасная вместо отсутствующей слабой)'}")
    done = _load_done(labels_path) if args.resume else {}
    pending = [path for path in photos if path.name not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"  фото: {len(photos)} · обработано: {len(done)} · к обработке: {len(pending)}")
    budget_s = args.time_budget_min * 60.0
    started = time.perf_counter()
    processed = 0
    with labels_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for path in pending:
            if time.perf_counter() - started > budget_s:
                print(f"TIME-BOX {args.time_budget_min} мин: обработано {processed}")
                break
            image = read_bgr(path)
            if image is None:
                continue
            photo_started = time.perf_counter()
            readings = build_readings(image, models, args.timeout)
            catalog = load_catalog(_ROOT / settings.catalog_jsonl)
            features = catalog_features(catalog)
            decision = judge(readings, catalog, features,
                             use_paddle=bool((readings.get("vlm_primary") or {}).get("ok") is False),
                             image=image)
            elapsed = round((time.perf_counter() - photo_started) * 1000.0, 1)
            record = {"file": path.name, "models": models, "readings": readings,
                      "decision": {key: value for key, value in decision.items() if key != "fields"},
                      "fields": decision.get("fields", {}), "elapsed_ms": elapsed,
                      "used_paddle": "paddle_signal" in decision}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            processed += 1
            print(f"[{processed}/{len(pending)}] {path.name[:26]:<28} "
                  f"{decision['verdict']:<8} согласие={decision['agreement']:<5} "
                  f"каталог={decision.get('in_catalog')} {elapsed/1000:.1f}с", flush=True)
    total = len(_load_done(labels_path))
    print(f"\nчекпоинт: {total} из {len(photos)} · файл {labels_path.relative_to(_ROOT)}")
    log_marker("M-EVAL", "multilabel_real100", "DONE", done=total, entries=len(photos))
    return 0


def dataset(args: argparse.Namespace) -> int:
    """Собирает датасет: принятые метки, эскалации и синтетические аугментации."""
    labels = _load_done(Path(args.labels))
    accepted: list[dict] = []
    escalated: list[dict] = []
    for name, record in labels.items():
        decision = record.get("decision") or {}
        models = record.get("models") or {}
        entry = {
            "file": name,
            "label": decision.get("label") or {},
            "confidence": decision.get("confidence", 0.0),
            "agreement": decision.get("agreement", 0.0),
            "in_catalog": decision.get("in_catalog", "none"),
            "catalog_candidates": decision.get("catalog_candidates") or [],
            "reasons": decision.get("reasons") or [],
            "provenance": {"readers": ["vlm_primary", "ocr_fusion", "vlm_qwen", "vlm_small"],
                           "models": {"vlm_primary": models.get("primary", ""),
                                      "vlm_qwen": models.get("qwen", ""),
                                      "vlm_small": models.get("small", "")},
                           "ocr_engine": (record.get("readings", {}).get("ocr_fusion") or {}).get("engine", ""),
                           "used_paddle": record.get("used_paddle", False)},
            "origin": "real_eval",
        }
        is_accepted = decision.get("verdict") == "accept"
        entry["split"] = assign_split("real_eval", escalated=not is_accepted)
        entry["policy"] = split_policy()
        (accepted if is_accepted else escalated).append(entry)
    dataset_path = Path(args.dataset)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w", encoding="utf-8") as handle:
        for entry in accepted:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with Path(args.escalations).open("w", encoding="utf-8") as handle:
        for entry in escalated:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  принято: {len(accepted)} · эскалировано: {len(escalated)}")
    # аугментации только для принятых фото: они наследуют разбиение источника
    augmented = _augment(dataset_path.parent, accepted, Path(args.dir), args.variants)
    with Path(args.augmented).open("w", encoding="utf-8") as handle:
        for entry in augmented:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  аугментаций: {len(augmented)} · файлы: {dataset_path.relative_to(_ROOT)}, "
          f"{Path(args.escalations).relative_to(_ROOT)}, {Path(args.augmented).relative_to(_ROOT)}")
    log_marker("M-EVAL", "multilabel_dataset", "DONE", accepted=len(accepted),
               escalated=len(escalated), augmented=len(augmented))
    return 0


def _augment(directory: Path, accepted: list[dict], photos_dir: Path, variants: int) -> list[dict]:
    """Строит синтетические варианты принятых фото через существующий модуль аугментации.

    Аугментация наследует разбиение источника: вариант оценочного фото остаётся оценочным, иначе
    копии оценочного набора просочились бы в обучение. Дубли исключаются по источнику и виду.
    """
    from scripts.eval import augment as augment_module  # noqa: PLC0415 - импорт ради единой точки аугментации
    rows: list[dict] = []
    for entry in accepted:
        source = photos_dir / entry["file"]
        image = _read_rgba(source)  # модуль аугментации принимает RGBA-кадр
        if image is None:
            continue
        for index in range(max(0, variants)):
            rng = np.random.default_rng(abs(hash((entry["file"], index))) % (2**32))
            try:
                variant = _call_augment(augment_module, image, rng)
            except Exception as error:  # noqa: BLE001 - сбой аугментации не должен ронять сборку
                print(f"  аугментация недоступна для {entry['file']}: {type(error).__name__}: {error}")
                break
            out_dir = directory / "images"
            out_dir.mkdir(parents=True, exist_ok=True)
            name = f"aug_{Path(entry['file']).stem}_{index}.jpg"
            array = np.asarray(variant)
            picture = Image.fromarray(array) if array.ndim == 3 and array.shape[2] == 4 else \
                Image.fromarray(array[:, :, ::-1] if array.ndim == 3 else array)
            picture.convert("RGB").save(out_dir / name, quality=88)
            rows.append({"file": str((out_dir / name).relative_to(_ROOT)),
                         "source": entry["file"], "kind": f"field_augment_{index}",
                         "label": entry["label"], "split": entry["split"],
                         "provenance": {**entry["provenance"], "synthetic": True},
                         "origin": "synthetic"})
    return rows


def report(args: argparse.Namespace) -> int:
    """Пишет отчёт и сводку консенсуса по результатам прогона."""
    labels = _load_done(Path(args.labels))
    rows = list(labels.values())
    accepted = [row for row in rows if (row.get("decision") or {}).get("verdict") == "accept"]
    escalated = [row for row in rows if (row.get("decision") or {}).get("verdict") == "escalate"]
    agreements = [float((row.get("decision") or {}).get("agreement") or 0.0) for row in rows]
    latencies = sorted(float(row.get("elapsed_ms") or 0.0) for row in rows)
    models = (rows[0].get("models") if rows else {}) or {}
    summary = {
        "photos": len(rows), "accepted": len(accepted), "escalated": len(escalated),
        "accept_rate": round(len(accepted) / len(rows), 4) if rows else 0.0,
        "agreement_avg": round(sum(agreements) / len(agreements), 4) if agreements else 0.0,
        "agreement_min": min(agreements) if agreements else 0.0,
        "agreement_max": max(agreements) if agreements else 0.0,
        "latency_p50_ms": latencies[len(latencies) // 2] if latencies else 0.0,
        "latency_total_s": round(sum(latencies) / 1000.0, 1),
        "models": models,
        "used_paddle": sum(1 for row in rows if row.get("used_paddle")),
        "policy": split_policy(),
        "accepted_labels": [{"file": row["file"], "label": (row.get("decision") or {}).get("label"),
                             "confidence": (row.get("decision") or {}).get("confidence")}
                            for row in accepted],
        "escalations": [{"file": row["file"], "reasons": (row.get("decision") or {}).get("reasons")}
                        for row in escalated],
    }
    Path(args.consensus).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Мультиридерная разметка реальных фото", "",
             f"Модели: R1 = {models.get('primary', '—')}, R3 = {models.get('small', '—')} "
             f"(R2 = OCR fusion)", "",
             f"- фото в прогоне: {summary['photos']}",
             f"- принято консенсусом: {summary['accepted']} ({summary['accept_rate']})",
             f"- эскалировано в ручную очередь: {summary['escalated']}",
             f"- согласие: среднее {summary['agreement_avg']}, от {summary['agreement_min']} до {summary['agreement_max']}",
             f"- время: p50 {summary['latency_p50_ms']} мс, всего {summary['latency_total_s']} с",
             f"- дополнительный сигнал PaddleOCR-VL: {summary['used_paddle']} фото", "",
             "## Политика разбиения", "", summary["policy"], "",
             "## Эскалации", ""]
    for item in summary["escalations"]:
        lines.append(f"- {item['file']}: {'; '.join(item['reasons'] or [])}")
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  принято {summary['accepted']} · эскалировано {summary['escalated']} · "
          f"согласие в среднем {summary['agreement_avg']} · p50 {summary['latency_p50_ms']} мс")
    print(f"  файлы: {Path(args.consensus).relative_to(_ROOT)} · {Path(args.report).relative_to(_ROOT)}")
    log_marker("M-EVAL", "multilabel_report", "DONE", **{k: summary[k] for k in
                                                         ("photos", "accepted", "escalated", "agreement_avg")})
    return 0


def main(argv: list[str] | None = None) -> int:
    """Запускает прогон, сборку датасета или отчёт."""
    args = parse_args(argv)
    if args.mode == "run":
        return run(args)
    return dataset(args) if args.mode == "dataset" else report(args)


if __name__ == "__main__":
    raise SystemExit(main())
