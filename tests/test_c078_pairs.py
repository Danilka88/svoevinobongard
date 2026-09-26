# START_MODULE_CONTRACT
# PURPOSE: Тесты C-078 сборки подтверждённых пар: источники, разбиение fit/holdout и защита от утечки
# SCOPE: CATALOG, test_assign_split_deterministic, test_assign_split_share, test_build_pairs_adds_split,
#        test_coverage_report_counts, test_load_feedback_only_up, test_leak_no_overlap
# DEPENDS: scripts.train.build_confirmed_pairs
# LINKS: M-EVAL, M-ACTIVE-LEARNING
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# CATALOG - синтетический каталог
# test_assign_split_deterministic - разбиение воспроизводимо по ключу
# test_assign_split_share - holdout занимает примерно 1/N
# test_build_pairs_adds_split - каждой паре назначается split
# test_coverage_report_counts - отчёт считает подтверждённые пары и покрытие
# test_load_feedback_only_up - только голоса vote=up становятся парами
# test_leak_no_overlap - fit и holdout не пересекаются (нет утечки)
# END_MODULE_MAP
"""Тесты сборки подтверждённых пар C-078 (без сети)."""
from __future__ import annotations

import json

from scripts.train.build_confirmed_pairs import (assign_split, build_pairs, coverage_report,
                                                 load_feedback_pairs)

CATALOG = {
    "s1": {"winery": "Кубань-Вино", "name": "Шато Тамань"},
    "s2": {"winery": "Абрау-Дюрсо", "name": "Резерв"},
}

__all__ = ["CATALOG", "test_assign_split_deterministic", "test_assign_split_share",
           "test_build_pairs_adds_split", "test_coverage_report_counts", "test_leak_no_overlap",
           "test_load_feedback_only_up"]


def test_assign_split_deterministic() -> None:
    """Разбиение одного ключа всегда одинаково."""
    assert assign_split("photo.jpg") == assign_split("photo.jpg")


def test_assign_split_share() -> None:
    """Holdout занимает примерно 1/N ключей."""
    splits = [assign_split(f"photo-{i}.jpg", holdout_mod=5) for i in range(500)]
    share = splits.count("holdout") / len(splits)
    assert 0.1 < share < 0.3


def test_build_pairs_adds_split() -> None:
    """Каждой паре назначается split fit или holdout."""
    gold = [{"file": "a.jpg", "slug": "s1", "source": "gold_matched"}]
    pairs = build_pairs(gold, [], [])
    assert len(pairs) == 1 and pairs[0]["split"] in {"fit", "holdout"}


def test_coverage_report_counts() -> None:
    """Отчёт считает подтверждённые (slug в каталоге) пары и покрытие."""
    pairs = [
        {"file": "a.jpg", "slug": "s1", "source": "gold_matched", "split": "fit"},
        {"file": "b.jpg", "slug": "s2", "source": "feedback_up", "split": "holdout"},
        {"file": "c.jpg", "slug": None, "source": "real100_consensus", "split": "fit"},
    ]
    report = coverage_report(pairs, CATALOG)
    assert report["confirmed_pairs"] == 2
    assert report["unresolved_pairs"] == 1
    assert report["fit"] == 1 and report["holdout"] == 1
    assert report["unique_slugs"] == 2
    assert report["ready_for_training"] is False


def test_load_feedback_only_up(tmp_path) -> None:
    """Только голоса vote=up становятся парами; down игнорируется."""
    path = tmp_path / "feedback.jsonl"
    path.write_text(
        json.dumps({"vote": "up", "slug": "s1", "image_sha256": "aaa"}) + "\n" +
        json.dumps({"vote": "down", "slug": "s2", "image_sha256": "bbb"}) + "\n",
        encoding="utf-8")
    pairs = load_feedback_pairs(path)
    assert len(pairs) == 1 and pairs[0]["slug"] == "s1"


def test_leak_no_overlap() -> None:
    """Fit и holdout не пересекаются по ключу — утечки нет."""
    gold = [{"file": f"photo-{i}.jpg", "slug": "s1", "source": "gold_matched"} for i in range(100)]
    pairs = build_pairs(gold, [], [])
    fit = {p["file"] for p in pairs if p["split"] == "fit"}
    holdout = {p["file"] for p in pairs if p["split"] == "holdout"}
    assert not (fit & holdout)
