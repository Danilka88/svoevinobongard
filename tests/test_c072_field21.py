# START_MODULE_CONTRACT
# PURPOSE: Тесты C-072: эталон 21 фото, инвариант утечки тестовых фото, чистые метрики A/B/C и каскад
# SCOPE: CATALOG, test_gold_report_counts_and_sources, test_gold_evaluable_is_matched_plus_ambiguous,
#        test_leak_invariant_field21, test_leak_scan_detects_planted_hash,
#        test_clean_flywheel_leak_removes_only_matching, test_acceptable_slugs_and_evaluable,
#        test_map_label_matched_ambiguous_not_found, test_compute_metrics_accuracy_and_refusal,
#        test_cascade_decision_confident_and_escalation, test_percentile_and_markdown
# DEPENDS: json, sys, pathlib, pytest, scripts.eval.field21_gold, scripts.eval.field21_abc
# LINKS: M-EVAL, M-DATA
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# CATALOG - синтетический каталог для проверки маппинга
# test_gold_report_counts_and_sources - эталон 21 фото: счёт статусов и ручные правки
# test_gold_evaluable_is_matched_plus_ambiguous - вычислимых = однозначные + неоднозначные
# test_leak_invariant_field21 - ни один хеш тестового фото не встречается в защищённых поверхностях и flywheel
# test_leak_scan_detects_planted_hash - сканер утечки находит подложенный хеш
# test_clean_flywheel_leak_removes_only_matching - очистка лога удаляет только совпавшие записи
# test_acceptable_slugs_and_evaluable - приемлемые slug и вычислимость по статусу эталона
# test_map_label_matched_ambiguous_not_found - маппинг различает однозначный, неоднозначный и ненайденный
# test_compute_metrics_accuracy_and_refusal - accuracy@1/@5, strict@1, not_found и корректный отказ
# test_cascade_decision_confident_and_escalation - каскад: уверенная классика, эскалация и пустой наблюдатель
# test_percentile_and_markdown - перцентиль и Markdown-отчёт содержат режимы и примеры
# END_MODULE_MAP
"""Тесты стенда C-072.

Проверяются машинные инварианты без обращения к сети и моделям: эталонный маппинг,
отсутствие утечки тестовых фото, чистые функции метрик A/B/C и решение каскада. Это
позволяет ловить регрессии стенда до дорогого реального прогона.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.eval.field21_abc import (acceptable_slugs, build_markdown, cascade_decision,  # noqa: E402
                                      compute_metrics, is_evaluable, percentile)
from scripts.eval.field21_gold import (DEFAULT_FIELD_DIR, clean_flywheel_leak,  # noqa: E402
                                       gold_status_counts, leak_scan, map_label)
from packages.common.text_rank import text_index  # noqa: E402

CATALOG = {
    "abrau-dyurso-shardone": {"winery": "Абрау-Дюрсо", "name": "Шардоне", "grape": "Шардоне",
                              "year": None, "sweetness": None},
    "abrau-dyurso-fizz-polusladkoe": {"winery": "Абрау-Дюрсо", "name": "Полусладкое",
                                      "grape": "", "year": None, "sweetness": "полусладкое"},
    "abrau-dyurso-udelnoe-polusladkoe": {"winery": "Абрау-Дюрсо", "name": "Полусладкое",
                                         "grape": "", "year": None, "sweetness": "полусладкое"},
    "other-risling": {"winery": "Другое", "name": "Рислинг", "grape": "Рислинг",
                      "year": None, "sweetness": None},
}

__all__ = [
    "CATALOG",
    "test_acceptable_slugs_and_evaluable",
    "test_cascade_decision_confident_and_escalation",
    "test_clean_flywheel_leak_removes_only_matching",
    "test_compute_metrics_accuracy_and_refusal",
    "test_gold_evaluable_is_matched_plus_ambiguous",
    "test_gold_report_counts_and_sources",
    "test_leak_invariant_field21",
    "test_leak_scan_detects_planted_hash",
    "test_map_label_matched_ambiguous_not_found",
    "test_percentile_and_markdown",
]


def _gold_report() -> dict:
    """Читает сгенерированный эталон; пустой словарь, если отчёт ещё не построен."""
    path = _ROOT / "reports" / "field21_gold.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def test_gold_report_counts_and_sources():
    """Эталон содержит 21 фото, статусы сходятся, ручные правки помечены source=manual."""
    gold = _gold_report()
    if not gold:
        pytest.skip("reports/field21_gold.json не построен")
    files = gold["files"]
    assert len(files) == 21
    counts = gold_status_counts(files)
    assert counts["matched"] + counts["ambiguous"] + counts["not_found"] == 21
    assert counts["total"] == 21
    manual = [name for name, record in files.items() if record.get("source") == "manual"]
    assert manual, "ожидались ручные правки эталона"
    for name in manual:
        assert files[name].get("note"), "ручная правка обязана иметь обоснование"


def test_gold_evaluable_is_matched_plus_ambiguous():
    """Вычислимый набор — это ровно однозначные плюс неоднозначные."""
    gold = _gold_report()
    if not gold:
        pytest.skip("reports/field21_gold.json не построен")
    counts = gold["counts"]
    assert counts["evaluable"] == counts["matched"] + counts["ambiguous"]


def test_leak_invariant_field21():
    """Ни один sha256 тестового фото не встречается в галерее/индексе/обучении и логе flywheel."""
    report = leak_scan(DEFAULT_FIELD_DIR)
    assert report["field_photos"] == 21
    assert report["protected_clean"], f"утечка в защищённые артефакты: {report['protected_hits'][:3]}"
    assert report["flywheel_clean"], f"утечка в лог flywheel: {report['flywheel_hits']} записей"


def test_leak_scan_detects_planted_hash(tmp_path):
    """Сканер утечки находит подложенный хеш в защищённой поверхности."""
    surface = tmp_path / "surface"
    surface.mkdir()
    planted = "a" * 64
    (surface / "index.json").write_text(f'{{"hash": "{planted}"}}', encoding="utf-8")
    report = leak_scan("/nonexistent-field-dir", roots=(surface,),
                       flywheel_path=tmp_path / "none.jsonl", hashes={"photo.jpg": planted})
    assert not report["protected_clean"]
    assert any(hit["kind"] == "hash" for hit in report["protected_hits"])


def test_clean_flywheel_leak_removes_only_matching(tmp_path):
    """Очистка лога удаляет записи с тестовыми хешами и сохраняет остальные."""
    log = tmp_path / "queries.jsonl"
    rows = [
        {"image_sha256": "keep1", "slug": "a"},
        {"image_sha256": "drop", "slug": "b"},
        {"image_sha256": "keep2", "slug": "c"},
    ]
    log.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    removed = clean_flywheel_leak(log, ["drop"])
    assert removed == 1
    kept = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [row["slug"] for row in kept] == ["a", "c"]


def test_acceptable_slugs_and_evaluable():
    """matched даёт один slug, ambiguous — набор кандидатов, not_found — пусто."""
    assert acceptable_slugs({"status": "matched", "expected_slug": "s1"}) == {"s1"}
    assert acceptable_slugs({"status": "ambiguous",
                             "candidates": [{"slug": "s2"}, {"slug": "s3"}]}) == {"s2", "s3"}
    assert acceptable_slugs({"status": "not_found"}) == set()
    assert is_evaluable({"status": "matched"}) and is_evaluable({"status": "ambiguous"})
    assert not is_evaluable({"status": "not_found"})


def test_map_label_matched_ambiguous_not_found():
    """Маппинг различает однозначное совпадение, серию-двойник и отсутствие в каталоге."""
    index = text_index(CATALOG)
    matched = map_label("x.jpg", "Абрау дюрсо шардоне", CATALOG, index)
    ambiguous = map_label("y.jpg", "Абрау дюрсо полусладкое", CATALOG, index)
    missing = map_label("z.jpg", "zzq invisible cuvee", CATALOG, index)
    assert matched["status"] == "matched"
    assert matched["expected_slug"] == "abrau-dyurso-shardone"
    assert ambiguous["status"] == "ambiguous"
    assert len(acceptable_slugs(ambiguous)) >= 2
    assert missing["status"] == "not_found"


def test_compute_metrics_accuracy_and_refusal():
    """Метрики считают accuracy на вычислимых и корректность отказа на отсутствующих."""
    gold = {"files": {
        "a.jpg": {"status": "matched", "expected_slug": "s1"},
        "b.jpg": {"status": "ambiguous", "candidates": [{"slug": "s2"}, {"slug": "s3"}]},
        "c.jpg": {"status": "not_found", "candidates": []},
    }}
    results = [
        {"file": "a.jpg", "top1": "s1", "top5": ["s1"], "not_found": False, "latency_ms": 10.0,
         "path": "classic", "scores": [0.9], "observation": "x", "fields": {"winery": "A"}},
        {"file": "b.jpg", "top1": "s3", "top5": ["s3"], "not_found": False, "latency_ms": 20.0,
         "path": "classic", "scores": [0.5], "observation": "", "fields": {}},
        {"file": "c.jpg", "top1": "wrong", "top5": ["wrong"], "not_found": False, "latency_ms": 30.0,
         "path": "classic", "scores": [0.4], "observation": "", "fields": {}},
    ]
    metrics = compute_metrics(results, gold)
    assert metrics["evaluable"] == 2
    assert metrics["accuracy_at_1"] == 1.0
    assert metrics["accuracy_at_5"] == 1.0
    assert metrics["matched_only"] == 1
    assert metrics["strict_at_1"] == 1.0
    assert metrics["not_found_rate"] == 0.0
    assert metrics["refusal_correct_rate"] == 0.0
    assert metrics["latency_ms"]["p50"] == 20.0


def test_cascade_decision_confident_and_escalation():
    """Каскад сохраняет уверенную классику, эскалирует неуверенную и терпит пустое наблюдение."""
    confident = {"top1": "s1", "top5": ["s1"], "scores": [0.9], "confident": True, "not_found": False}
    assert cascade_decision(confident, [("s9", 0.99)])["route"] == "classic"
    unsure = {"top1": "s1", "top5": ["s1"], "scores": [0.3], "confident": False, "not_found": True}
    escalated = cascade_decision(unsure, [("s2", 0.8)])
    assert escalated["escalated"] and escalated["route"] == "observer" and escalated["top1"] == "s2"
    empty = cascade_decision(unsure, [])
    assert empty["route"] == "classic:observer_empty" and empty["top1"] == "s1"
    merged_boost = cascade_decision(unsure, [("s1", 0.5)])
    assert merged_boost["route"] == "classic+observer" and merged_boost["top1"] == "s1"


def test_percentile_and_markdown():
    """Перцентиль считается по ближайшему рангу, Markdown содержит режимы и примеры."""
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert percentile([], 0.95) == 0.0
    payload = {
        "gold": {"counts": {"total": 21, "matched": 4, "ambiguous": 7, "not_found": 10, "evaluable": 11}},
        "models": ["m1"],
        "modes": {
            "A": {"accuracy_at_1": 0.45, "accuracy_at_5": 0.45, "strict_at_1": 1.0, "not_found_rate": 0.76,
                  "refusal_correct_rate": 1.0, "paths": {"classic": 21}, "latency_ms": {"p50": 200.0, "p95": 500.0}},
            "B:m1": {"accuracy_at_1": 0.27, "accuracy_at_5": 0.36, "strict_at_1": 0.25, "not_found_rate": 0.0,
                     "refusal_correct_rate": 0.0, "paths": {"observer": 20}, "latency_ms": {"p50": 3000.0, "p95": 4000.0}},
            "C:m1": {"accuracy_at_1": 0.45, "accuracy_at_5": 0.45, "strict_at_1": 1.0, "not_found_rate": 0.0,
                     "refusal_correct_rate": 0.0, "paths": {"classic+observer": 16}, "latency_ms": {"p50": 3000.0, "p95": 4000.0}},
        },
        "examples": [{"file": "a.jpg", "human_label": "А", "expected": "s1", "status": "matched",
                      "observation": "winery: A", "fields": {"winery": "A"}, "candidates": ["s1"],
                      "top1": "s1", "not_found": False, "b_top1": "s1", "c_top1": "s1"}],
        "conclusions": ["вывод"],
    }
    markdown = build_markdown(payload)
    assert "accuracy" in markdown and "m1" in markdown
    assert "classic" in markdown
    assert "a.jpg" in markdown
