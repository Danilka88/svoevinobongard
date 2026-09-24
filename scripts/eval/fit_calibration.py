#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Обучение калибровки уверенности по реальным фото и проверка на отложенном подмножестве
# SCOPE: load_labels, pairs_from_scans, threshold_table, build_report, main
# DEPENDS: json, pathlib, numpy, packages.common.calibration, packages.common.logging
# LINKS: M-EVAL, M-DATA
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# THRESHOLDS - пороги вероятности для таблицы эскалации
# load_labels - читает разметку C-041 с кандидатами каталога
# pairs_from_consensus - пары для отложенной проверки по консенсус-меткам
# pairs_from_scans - строит пары «скор, правильность» из прогона сканера и правильных ответов
# threshold_table - таблица «порог → доля эскалаций → точность на остатке»
# build_report - пишет отчёт обучения калибровки
# main - точка входа
# END_MODULE_MAP
"""Обучение калибровки уверенности.

Обучаем на одном подмножестве реальных фото, проверяем на другом: так видно переоценку при сдвиге
распределения. Сырая уверенность сравнивается с калиброванной по ECE и по таблице порогов — именно
эта таблица показывает, сколько запросов уйдёт на проверку и какой ценой.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from packages.common.calibration import (  # noqa: E402
    apply,
    expected_calibration_error,
    fit_conservative,
    reliability_table,
    save,
    should_escalate,
)
from packages.common.logging import log_marker  # noqa: E402

THRESHOLDS = (0.90, 0.95, 0.99)

__all__ = ["THRESHOLDS", "build_report", "load_labels", "main", "pairs_from_consensus",
           "pairs_from_scans", "threshold_table"]


def load_labels(path: str | Path) -> dict[str, dict]:
    """Читает разметку C-041: строки с кандидатами каталога и признаком попадания."""
    target = Path(path)
    if not target.is_file():
        return {}
    payload = json.loads(target.read_text(encoding="utf-8"))
    return {row["file"]: row for row in payload.get("rows", [])}


def pairs_from_scans(scan_path: str | Path, labels: dict[str, dict], *,
                     require_strict: bool = True) -> list[tuple[float, int]]:
    """Строит пары «сырая уверенность, правильно ли» из прогона сканера.

    Правильным считается попадание в строгий кандидат разметки: сравнение идёт по slug, а не по
    свободному тексту. Пары без правильного ответа отбрасываются — иначе калибровка училась бы на
    шуме.
    """
    target = Path(scan_path)
    if not target.is_file():
        return []
    pairs: list[tuple[float, int]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        label = labels.get(row.get("file")) or {}
        gt = [candidate["slug"] for candidate in (label.get("slug_candidates") or [])
              if candidate.get("winery_score", 0) >= 0.86 and candidate.get("detail_score", 0) >= 0.86]
        if not gt and require_strict:
            continue
        if not gt:
            continue
        confidence = row.get("confidence")
        if confidence is None:
            continue
        pairs.append((float(confidence), 1 if row.get("slug") in gt else 0))
    return pairs


def pairs_from_consensus(scan_path: str | Path, consensus_path: str | Path) -> list[tuple[float, int]]:
    """Пары для отложенной проверки: консенсус-метки принятых случаев."""
    consensus = {}
    target = Path(consensus_path)
    if target.is_file():
        for line in target.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                candidates = entry.get("catalog_candidates") or []
                if candidates:
                    consensus[entry["file"]] = str(candidates[0])
    scans = Path(scan_path)
    if not scans.is_file() or not consensus:
        return []
    pairs: list[tuple[float, int]] = []
    for line in scans.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        gt = consensus.get(row.get("file"))
        if not gt or row.get("confidence") is None:
            continue
        pairs.append((float(row["confidence"]), 1 if row.get("slug") == gt else 0))
    return pairs


def threshold_table(pairs: list[tuple[float, int]], params, *, title: str) -> list[dict]:
    """Таблица «порог → доля эскалаций → точность на остатке», отдельно по сырому и калиброванному сигналу."""
    rows: list[dict] = []
    for signal in ("raw", "calibrated"):
        for threshold in THRESHOLDS:
            kept = [pair for pair in pairs
                    if not should_escalate(apply(params, pair[0]) if signal == "calibrated" else pair[0],
                                           threshold)]
            escalated = len(pairs) - len(kept)
            accuracy = round(float(np.mean([pair[1] for pair in kept])), 4) if kept else 0.0
            rows.append({"distribution": title, "signal": signal, "threshold": threshold,
                         "escalation_share": round(escalated / len(pairs), 4) if pairs else 0.0,
                         "kept": len(kept), "accuracy_on_kept": accuracy})
    return rows


def build_report(params, diagnostics: dict, fit_pairs, holdout_pairs, out_json: str, out_md: str) -> dict:
    """Считает ECE и таблицы порогов, пишет JSON и markdown."""
    fit_calibrated = [apply(params, raw) for raw, _ in fit_pairs]
    fit_correct = [value for _, value in fit_pairs]
    holdout_calibrated = [apply(params, raw) for raw, _ in holdout_pairs]
    holdout_correct = [value for _, value in holdout_pairs]
    report = {
        "params": params.as_dict(),
        "diagnostics": diagnostics,
        "fit": {"pairs": len(fit_pairs),
                "ece_raw": expected_calibration_error([raw for raw, _ in fit_pairs], fit_correct),
                "ece_calibrated": expected_calibration_error(fit_calibrated, fit_correct),
                "mean_p_raw": round(float(np.mean([raw for raw, _ in fit_pairs])), 4) if fit_pairs else 0.0,
                "mean_p_calibrated": round(float(np.mean(fit_calibrated)), 4) if fit_calibrated else 0.0,
                "accuracy": round(float(np.mean(fit_correct)), 4) if fit_correct else 0.0},
        "holdout": {"pairs": len(holdout_pairs),
                    "ece_raw": expected_calibration_error([raw for raw, _ in holdout_pairs], holdout_correct),
                    "ece_calibrated": expected_calibration_error(holdout_calibrated, holdout_correct),
                    "accuracy": round(float(np.mean(holdout_correct)), 4) if holdout_correct else 0.0},
        "thresholds": threshold_table(fit_pairs, params, title="обучающая выборка")
                      + threshold_table(holdout_pairs, params, title="отложенная выборка"),
        "reliability_holdout": reliability_table(holdout_calibrated, holdout_correct, bins=5),
        "examples": [{"raw": round(raw, 4), "correct": correct, "p_correct": apply(params, raw)}
                     for raw, correct in (fit_pairs[:5] + holdout_pairs[:5])],
        "caveat": "метки собраны моделями (консенсус разметки), значения относительные",
    }
    Path(out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Калибровка уверенности (C-061)", "",
             f"Параметры: slope {params.slope:.2f} · pivot {params.pivot:.4f} · "
             f"base_rate {params.base_rate} · shrink {params.shrink} · потолок {params.ceiling}", "",
             f"Обучающая выборка: {len(fit_pairs)} пар · отложенная: {len(holdout_pairs)} пар", "",
             "| Выборка | ECE сырой | ECE калиброванной | Точность |", "|---|---|---|---|",
             f"| обучающая | {report['fit']['ece_raw']} | {report['fit']['ece_calibrated']} | {report['fit']['accuracy']} |",
             f"| отложенная | {report['holdout']['ece_raw']} | {report['holdout']['ece_calibrated']} | {report['holdout']['accuracy']} |",
             "", "## Пороги", "", "| Выборка | Сигнал | Порог | Доля эскалаций | Осталось | Точность на остатке |",
             "|---|---|---|---|---|---|"]
    for row in report["thresholds"]:
        lines.append(f"| {row['distribution']} | {row['signal']} | {row['threshold']} | "
                     f"{row['escalation_share']} | {row['kept']} | {row['accuracy_on_kept']} |")
    lines += ["", "## Примеры отображения", "", "| Сырой скор | Правильно | p(correct) |", "|---|---|---|"]
    for row in report["examples"]:
        lines.append(f"| {row['raw']} | {row['correct']} | {row['p_correct']} |")
    lines += ["", "## Оговорки", "",
              "- Метки собраны моделями (консенсус разметки), поэтому числа **относительные**.",
              f"- Пар мало (обучение {len(fit_pairs)}, проверка {len(holdout_pairs)}): устойчивость калибровки ограничена;",
              "  консервативная поправка включается автоматически при переоценке на отложенной выборке."]
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    """Обучает калибровку и пишет отчёт."""
    parser = argparse.ArgumentParser(description="Обучение калибровки уверенности")
    parser.add_argument("--fit-scans", default=str(_ROOT / "reports" / "real100_scan.jsonl"))
    parser.add_argument("--labels", default=str(_ROOT / "reports" / "real100_labels.json"))
    parser.add_argument("--holdout-scans", default=str(_ROOT / "reports" / "baseline_clean_scan.jsonl"))
    parser.add_argument("--consensus", default=str(_ROOT / "data" / "annotation" / "consensus.jsonl"))
    parser.add_argument("--out", default=str(_ROOT / "data" / "calibration" / "platt_v1.json"))
    parser.add_argument("--report", default=str(_ROOT / "reports" / "calibration_report.json"))
    parser.add_argument("--md", default=str(_ROOT / "reports" / "calibration_report.md"))
    args = parser.parse_args(argv)
    labels = load_labels(args.labels)
    fit_pairs = pairs_from_scans(args.fit_scans, labels)
    holdout_pairs = pairs_from_consensus(args.holdout_scans, args.consensus)
    print(f"  пар: обучение {len(fit_pairs)} · отложенная проверка {len(holdout_pairs)}")
    if not fit_pairs:
        print("пар для обучения нет")
        return 1
    params, diagnostics = fit_conservative(fit_pairs, holdout_pairs)
    save(params, args.out)
    report = build_report(params, diagnostics, fit_pairs, holdout_pairs, args.report, args.md)
    print(f"===== КАЛИБРОВКА =====")
    print(f"  slope {params.slope:.2f} · pivot {params.pivot:.4f} · shrink {params.shrink} · потолок {params.ceiling}")
    print(f"  ECE обучающая {report['fit']['ece_raw']} → {report['fit']['ece_calibrated']}")
    print(f"  ECE отложенная {report['holdout']['ece_raw']} → {report['holdout']['ece_calibrated']} (точность {report['holdout']['accuracy']})")
    for row in report["thresholds"]:
        if row["signal"] == "calibrated":
            print(f"  {row['distribution']} порог {row['threshold']}: эскалаций {row['escalation_share']} · точность на остатке {row['accuracy_on_kept']}")
    print(f"  примеры: " + ", ".join(f"{row['raw']}→{row['p_correct']}" for row in report["examples"][:4]))
    log_marker("M-EVAL", "fit_calibration", "DONE", fit=len(fit_pairs), holdout=len(holdout_pairs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
