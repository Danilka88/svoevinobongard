#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Бенч моделей-наблюдателей: точность, задержка и доля корректного отказа на размеченном подмножестве
# SCOPE: MODELS, load_consensus, bench_model, build_report, main
# DEPENDS: json, statistics, time, numpy, PIL, packages.common.catalog, packages.common.config,
#          packages.common.logging, packages.common.observer, packages.common.preprocess,
#          services.normalize.app.tta, services.vlm.app.extract
# LINKS: M-EVAL, M-VLM
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# MODELS - модели-наблюдатели для сравнения
# load_consensus - читает консенсус-метки с правильными кандидатами
# bench_model - прогон одной модели по подмножеству
# build_report - пишет JSON и markdown отчёт бенча
# main - точка входа
# END_MODULE_MAP
"""Бенч моделей-наблюдателей.

Сравниваем модели по точности (попадание в правильное вино из консенсуса), задержке и доле
корректного отказа. Скорость вторична: наблюдатель вызывается только при недостаточной уверенности.
Честно фиксируются модели, которые не дали наблюдения вовсе.
"""
from __future__ import annotations

import argparse
import json
import statistics
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
from packages.common.logging import log_marker  # noqa: E402
from packages.common.observer import observation_fields, observer_candidates, parse_observation  # noqa: E402
from packages.common.preprocess import PreprocessConfig  # noqa: E402
from services.normalize.app.tta import build_vlm_inputs  # noqa: E402
from services.vlm.app.extract import observe_label  # noqa: E402

MODELS = ("qwen3-vl:2b", "gemma4:e4b-mlx", "gemma4:12b-mlx")

__all__ = ["MODELS", "bench_model", "build_report", "load_consensus", "main"]


def load_consensus(path: str | Path) -> list[dict]:
    """Читает консенсус-метки: правильный slug берётся из кандидатов каталога."""
    target = Path(path)
    if not target.is_file():
        return []
    rows: list[dict] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        candidates = entry.get("catalog_candidates") or []
        if candidates:
            rows.append({"file": entry["file"], "gold": str(candidates[0])})
    return rows


def bench_model(model: str, entries: list[dict], catalog: dict, photos: Path,
                timeout: float = 240.0) -> dict:
    """Прогон одной модели: наблюдение → поля → поиск по всему каталогу."""
    latencies: list[float] = []
    hits1 = hits5 = empty = 0
    rows: list[dict] = []
    for entry in entries:
        path = photos / entry["file"]
        if not path.is_file():
            continue
        image = np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1].copy()
        images, _ = build_vlm_inputs(image, PreprocessConfig(pas_tiles=settings.pas_tiles))
        started = time.perf_counter()
        result = observe_label(images, None, model=model, max_tokens=1024, timeout_s=timeout)
        elapsed = round(time.perf_counter() - started, 1)
        latencies.append(elapsed)
        text = str(result.get("raw_text") or "")
        observation = parse_observation(text)
        candidates = observer_candidates(observation, catalog, limit=5)
        slugs = [slug for slug, _ in candidates]
        if not text.strip():
            empty += 1
        if slugs and slugs[0] == entry["gold"]:
            hits1 += 1
        if entry["gold"] in slugs:
            hits5 += 1
        rows.append({"file": entry["file"], "gold": entry["gold"], "ok": bool(text.strip()),
                     "observation": text[:200], "fields": observation_fields(observation),
                     "top5": slugs[:5], "latency_s": elapsed})
        print(f"  [{model}] {entry['file'][:24]:<26} ok={str(bool(text.strip())):<5} "
              f"{(slugs[0] if slugs else '—')[:34]:<36} {elapsed:>5}с", flush=True)
    total = len(rows) or 1
    return {"model": model, "photos": len(rows), "identity_at_1": round(hits1 / total, 4),
            "identity_at_5": round(hits5 / total, 4), "empty_observations": empty,
            "empty_share": round(empty / total, 4),
            "latency_p50_s": round(statistics.median(latencies), 1) if latencies else 0.0,
            "latency_p95_s": round(sorted(latencies)[min(len(latencies) - 1, int(0.95 * len(latencies)))], 1) if latencies else 0.0,
            "rows": rows}


def build_report(results: list[dict], out_json: str, out_md: str) -> dict:
    """Пишет отчёт бенча: таблица моделей и оговорки."""
    best = max(results, key=lambda item: (item["identity_at_1"], -item["latency_p50_s"])) if results else None
    payload = {"models": results, "best_by_accuracy": best["model"] if best else "",
               "caveat": ("подмножество мало, метки собраны моделями: числа относительные; "
                          "модели без наблюдения отмечены как пустые")}
    Path(out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Бенч моделей-наблюдателей (C-063)", "",
             "| Модель | Фото | identity@1 | identity@5 | Пустых наблюдений | Задержка p50 / p95 |",
             "|---|---|---|---|---|---|"]
    for item in results:
        lines.append(f"| {item['model']} | {item['photos']} | {item['identity_at_1']} | "
                     f"{item['identity_at_5']} | {item['empty_observations']} "
                     f"({item['empty_share']}) | {item['latency_p50_s']} / {item['latency_p95_s']} с |")
    lines += ["", f"Лучшая по точности: **{payload['best_by_accuracy']}**", "",
              "## Оговорки", "", f"- {payload['caveat']}",
              "- Пустое наблюдение означает, что модель не дала текста (thinking-модели упираются в бюджет токенов);",
              "  такие модели нельзя ставить в цепочку наблюдателя без дополнительной настройки.",
              "- Скорость вторична: наблюдатель подключается только при недостаточной уверенности поиска."]
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    """Запускает бенч по подмножеству консенсус-меток."""
    parser = argparse.ArgumentParser(description="Бенч моделей-наблюдателей")
    parser.add_argument("--consensus", default=str(_ROOT / "data" / "annotation" / "consensus.jsonl"))
    parser.add_argument("--photos", default=str(_ROOT / "data" / "samples" / "real100"))
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--out", default=str(_ROOT / "reports" / "observer_bench.json"))
    parser.add_argument("--md", default=str(_ROOT / "reports" / "observer_bench.md"))
    args = parser.parse_args(argv)
    entries = load_consensus(args.consensus)[: args.limit]
    if not entries:
        print("консенсус-меток нет: сначала прогоните разметку")
        return 1
    catalog = load_catalog(_ROOT / settings.catalog_jsonl)
    results = [bench_model(model.strip(), entries, catalog, Path(args.photos))
               for model in args.models.split(",") if model.strip()]
    payload = build_report(results, args.out, args.md)
    print("===== БЕНЧ НАБЛЮДАТЕЛЕЙ =====")
    for item in results:
        print(f"  {item['model']:<16} identity@1 {item['identity_at_1']} · @5 {item['identity_at_5']} · "
              f"пустых {item['empty_observations']}/{item['photos']} · p50 {item['latency_p50_s']}с")
    print(f"  лучшая по точности: {payload['best_by_accuracy']}")
    log_marker("M-EVAL", "observer_bench", "DONE", best=payload["best_by_accuracy"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
