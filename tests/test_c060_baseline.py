# START_MODULE_CONTRACT
# PURPOSE: Проверка честного baseline: near-dup, метрики подмножества, таблица эскалации и веса четырёх ридеров
# SCOPE: CATALOG, CONSENSUS, ROWS, test_reader_weights_sum, test_gt_of_uses_catalog_candidate,
#        test_find_siblings_detects_series, test_find_siblings_ignores_other_winery,
#        test_metrics_on_subset, test_escalation_table_shape, test_report_files_are_written
# DEPENDS: json, pathlib, pytest, packages.common.annotation, scripts.eval.baseline_clean
# LINKS: M-EVAL, M-DATA
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# CATALOG - синтетический каталог с серией-двойником
# CONSENSUS - синтетические консенсус-метки разметки
# ROWS - синтетические строки прогона пайплайна
# test_reader_weights_sum - веса четырёх ридеров нормированы
# test_gt_of_uses_catalog_candidate - правильный slug берётся из кандидатов каталога
# test_find_siblings_detects_series - серия-двойник находится по винодельне и названию
# test_find_siblings_ignores_other_winery - чужая винодельня не считается двойником
# test_metrics_on_subset - метрики считаются только на принятом подмножестве
# test_escalation_table_shape - таблица эскалации содержит долю и точность на остатке
# test_report_files_are_written - отчёт пишется в JSON и markdown
# END_MODULE_MAP
"""Проверка baseline на синтетике: числа считаются там, где нужно, и не зависят от моделей."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from packages.common.annotation import READER_WEIGHTS
from scripts.eval.baseline_clean import (
    build_report,
    compute_metrics,
    escalation_table,
    find_siblings,
    gt_of,
)

__all__ = [
    "CATALOG",
    "CONSENSUS",
    "ROWS",
    "test_reader_weights_sum",
    "test_gt_of_uses_catalog_candidate",
    "test_find_siblings_detects_series",
    "test_find_siblings_ignores_other_winery",
    "test_metrics_on_subset",
    "test_escalation_table_shape",
    "test_report_files_are_written",
]

CATALOG = {
    "wine-2021": {"name": "Небо России Глера", "winery": "Небо России", "year": 2021, "sweetness": "сухое"},
    "wine-2022": {"name": "Небо России Глера", "winery": "Небо России", "year": 2022, "sweetness": "сухое"},
    "other": {"name": "Небо России Глера", "winery": "Другая винодельня", "year": 2021, "sweetness": "сухое"},
    "unique": {"name": "Совершенно другое вино", "winery": "Иная", "year": 2020, "sweetness": "сладкое"},
}

CONSENSUS = [
    {"file": "a.webp", "decision": {"verdict": "accept", "in_catalog": "strong", "catalog_candidates": ["wine-2021"]}},
    {"file": "b.webp", "decision": {"verdict": "accept", "in_catalog": "weak", "catalog_candidates": ["unique"]}},
    {"file": "c.webp", "decision": {"verdict": "escalate", "in_catalog": "none", "catalog_candidates": []}},
]

ROWS = [
    {"file": "a.webp", "slug": "wine-2021", "top5": ["wine-2021", "wine-2022"], "confidence": 0.95,
     "margin": 0.02, "not_found": False, "latency_ms": 900},
    {"file": "b.webp", "slug": "wrong", "top5": ["other", "unique"], "confidence": 0.80,
     "margin": 0.01, "not_found": False, "latency_ms": 1100},
    {"file": "c.webp", "slug": "wine-2021", "top5": [], "confidence": 0.30, "margin": 0.0,
     "not_found": True, "latency_ms": 800},
]


def test_reader_weights_sum() -> None:
    """Четыре ридера: сильная модель весит больше остальных, сумма нормирована."""
    assert set(READER_WEIGHTS) == {"vlm_primary", "ocr_fusion", "vlm_qwen", "vlm_small"}
    assert pytest.approx(sum(READER_WEIGHTS.values()), abs=1e-6) == 1.0
    assert READER_WEIGHTS["vlm_primary"] > READER_WEIGHTS["vlm_qwen"] > READER_WEIGHTS["vlm_small"]


def test_gt_of_uses_catalog_candidate() -> None:
    assert gt_of(CONSENSUS[0]) == "wine-2021"
    assert gt_of(CONSENSUS[2]) == ""  # эскалированный случай не считается правильным ответом


def test_find_siblings_detects_series() -> None:
    siblings = find_siblings("wine-2021", CATALOG)
    assert "wine-2022" in siblings


def test_find_siblings_ignores_other_winery() -> None:
    siblings = find_siblings("wine-2021", {key: value for key, value in CATALOG.items() if key != "wine-2022"})
    assert siblings == []


def test_metrics_on_subset() -> None:
    metrics = compute_metrics(ROWS, CONSENSUS, CATALOG)
    assert metrics["subset"] == 2  # только принятые метки с slug
    assert metrics["identity_at_1"] == 0.5
    assert metrics["identity_at_5"] == 1.0
    assert metrics["strict_subset"] == 1 and metrics["strict_only_at_1"] == 1.0
    assert metrics["near_dup_subset"] == 1  # двойник есть только у первой серии
    assert metrics["ood_cases"] == 1 and metrics["ood_not_found_rate"] == 1.0
    assert metrics["latency_p50_ms"] > 0
    assert metrics["margin_p95"] > 0


def test_escalation_table_shape() -> None:
    metrics = compute_metrics(ROWS, CONSENSUS, CATALOG)
    table = escalation_table(metrics["rows"])
    assert len(table) == 4  # три порога уверенности и действующее правило поиска
    by_threshold = {row["threshold"]: row for row in table if row["signal"] == "confidence"}
    assert by_threshold[0.90]["escalation_share"] == 0.5
    assert by_threshold[0.90]["accuracy_on_kept"] == 1.0
    assert by_threshold[0.99]["escalation_share"] == 1.0
    current = [row for row in table if "score_tau" in row["signal"]][0]
    assert current["escalation_share"] == 0.0  # среди подмножества отказов нет


def test_report_files_are_written(tmp_path: Path) -> None:
    metrics = compute_metrics(ROWS, CONSENSUS, CATALOG)
    out_json = tmp_path / "baseline.json"
    out_md = tmp_path / "baseline.md"
    build_report(metrics, escalation_table(metrics["rows"]), CONSENSUS,
                 {"primary": "gemma4:26b-mlx", "qwen": "qwen3-vl:2b", "small": "gemma4:e4b-mlx"},
                 len(ROWS), 100, str(out_json), str(out_md))
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["metrics"]["identity_at_1"] == 0.5
    assert payload["models"]["qwen"] == "qwen3-vl:2b"
    text = out_md.read_text(encoding="utf-8")
    assert "identity@1" in text and "Инструментация эскалации" in text
    assert "относительные" in text  # оговорка о слабой опоре на истину
