#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Финальная сводка метрик и машино-проверяемая проверка quality/SLA-гейта
# SCOPE: load_report, collect_final, check_gate, main
# DEPENDS: json, argparse, pathlib, packages.common.config, packages.common.logging
# LINKS: M-EVAL, M-DEPLOY
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# DEFAULT_MIN_ACC1 - порог acc@1 для гейта (текущий baseline пайплайна)
# DEFAULT_MAX_P95 - предел p95 latency в миллисекундах
# load_report - безопасное чтение отчёта
# collect_final - собирает итоговые метрики из baseline, калибровки и конфигурации
# check_gate - проверяет acc@1 и p95, возвращая список нарушений
# main - запись reports/final.json и/или проверка гейта
# END_MODULE_MAP
"""Итоговые метрики проекта и проверка гейта.

Одна команда отвечает на два вопроса: «какое качество у пайплайна сейчас» и
«не просело ли оно ниже зафиксированного порога». Порог acc@1 — текущий baseline
full-frame пайплайна (0.670 на 200 сэмплах, seed 42); SLA p95 взят из ТЗ (3000 мс).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from packages.common.config import settings  # noqa: E402
from packages.common.logging import log_marker  # noqa: E402

DEFAULT_MIN_ACC1 = 0.67
DEFAULT_MAX_P95 = 3000.0

__all__ = ["DEFAULT_MAX_P95", "DEFAULT_MIN_ACC1", "check_gate", "collect_final", "load_report", "main"]


def load_report(path: str | Path) -> dict:
    """Читает JSON-отчёт, возвращая пустой словарь при отсутствии или ошибке."""
    target = Path(path)
    if not target.is_file():
        return {}
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log_marker("M-EVAL", "final_report", "CORRUPT", path=str(target))
        return {}


def collect_final(baseline: dict, calibration: dict) -> dict:
    """Собирает итоговые метрики: качество, latency, пороги и включённые режимы."""
    field = baseline.get("synthetic_field") or {}
    latency = field.get("latency_ms") or {}
    near = baseline.get("near_duplicate") or {}
    chosen = calibration.get("chosen") or {}
    return {
        "quality": {
            "acc_at_1": field.get("accuracy@1"),
            "acc_at_5": field.get("accuracy@5"),
            "group_acc_at_1": field.get("group_accuracy@1"),
            "near_duplicate_acc_at_1": near.get("accuracy@1"),
            "near_duplicate_group_acc_at_1": near.get("group_accuracy@1"),
            "near_duplicate_samples": near.get("samples"),
            "self_acc_at_1": field.get("self_accuracy@1"),
        },
        "latency_ms": {"p50": latency.get("p50"), "p95": latency.get("p95")},
        "evaluation": {
            "samples": baseline.get("samples"),
            "seed": baseline.get("seed"),
            "normalize": baseline.get("normalize_applied"),
            "geo_method": baseline.get("geo_method"),
            "encoder": baseline.get("encoder"),
            "domain": "эталоны каталога + синтетическая аугментация (полевых фото нет с C-011)",
        },
        "thresholds": {
            "score_tau": settings.score_tau,
            "margin_delta": settings.margin_delta,
            "geo_weight": settings.geo_weight,
            "confidence_temperature": settings.confidence_temperature,
        },
        "calibration": {
            "score_tau_separation_f1": chosen.get("score_tau_separation_f1"),
            "margin_delta_separation_f1": chosen.get("margin_delta_separation_f1"),
            "confidence_separation_f1": chosen.get("confidence_separation_f1"),
        },
        "defaults": {
            "text_fusion_mode": settings.text_fusion_mode,
            "ocr_engine": settings.ocr_engine or "rapidocr",
            "vlm_fallback_enabled": settings.vlm_fallback_enabled,
            "pas_tiles": settings.pas_tiles,
            "geo_weight": settings.geo_weight,
        },
        "sla_ms": DEFAULT_MAX_P95,
        "min_acc_at_1": DEFAULT_MIN_ACC1,
    }


def check_gate(final: dict, min_acc1: float, max_p95: float) -> list[str]:
    """Проверяет гейт: возвращает список нарушений (пустой список — гейт пройден)."""
    problems: list[str] = []
    acc1 = (final.get("quality") or {}).get("acc_at_1")
    p95 = (final.get("latency_ms") or {}).get("p95")
    if acc1 is None:
        problems.append("acc@1 отсутствует: прогоните make eval")
    elif acc1 < min_acc1:
        problems.append(f"acc@1 {acc1} ниже порога {min_acc1}")
    if p95 is None:
        problems.append("p95 отсутствует: прогоните make eval")
    elif p95 > max_p95:
        problems.append(f"p95 {p95} мс превышает SLA {max_p95} мс")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Собирает итоговый отчёт и/или проверяет гейт."""
    parser = argparse.ArgumentParser(description="Итоговые метрики и quality/SLA-гейт")
    parser.add_argument("--baseline", default="reports/baseline.json")
    parser.add_argument("--calibration", default="reports/calibration.json")
    parser.add_argument("--out", default="reports/final.json")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--min-acc1", type=float, default=DEFAULT_MIN_ACC1)
    parser.add_argument("--max-p95", type=float, default=DEFAULT_MAX_P95)
    args = parser.parse_args(argv)

    baseline = load_report(_ROOT / args.baseline)
    calibration = load_report(_ROOT / args.calibration)
    final = collect_final(baseline, calibration)

    target = _ROOT / args.out
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    quality = final["quality"]
    print("=== ФИНАЛЬНЫЕ МЕТРИКИ ===")
    print(f"  acc@1 {quality['acc_at_1']} · acc@5 {quality['acc_at_5']} · группа acc@1 {quality['group_acc_at_1']}")
    print(f"  near-dup acc@1 {quality['near_duplicate_acc_at_1']} (n={quality['near_duplicate_samples']})")
    print(f"  latency p50 {final['latency_ms']['p50']} мс · p95 {final['latency_ms']['p95']} мс (SLA {final['sla_ms']})")
    print(f"  пороги: tau {final['thresholds']['score_tau']} · delta {final['thresholds']['margin_delta']} "
          f"· geo {final['thresholds']['geo_weight']} · temp {final['thresholds']['confidence_temperature']}")
    print(f"  дефолты: fusion={final['defaults']['text_fusion_mode']} · ocr={final['defaults']['ocr_engine']} "
          f"· vlm={final['defaults']['vlm_fallback_enabled']}")
    print(f"  отчёт: {target.relative_to(_ROOT)}")

    if not args.check:
        return 0
    problems = check_gate(final, args.min_acc1, args.max_p95)
    if problems:
        print("=== ГЕЙТ НЕ ПРОЙДЕН ===")
        for problem in problems:
            print(f"  ✗ {problem}")
        log_marker("M-EVAL", "gate", "FAILED", problems=len(problems))
        return 1
    print(f"=== ГЕЙТ ПРОЙДЕН: acc@1 {quality['acc_at_1']} ≥ {args.min_acc1}, "
          f"p95 {final['latency_ms']['p95']} мс ≤ {args.max_p95} мс ===")
    log_marker("M-EVAL", "gate", "PASSED", acc1=quality["acc_at_1"], p95=final["latency_ms"]["p95"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
