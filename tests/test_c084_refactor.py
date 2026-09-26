# START_MODULE_CONTRACT
# PURPOSE: Тесты C-084: эквивалентность общего group_key прежнему алгоритму и отсутствие дублей/мёртвого кода
# SCOPE: test_group_key_matches_previous_algorithm, test_group_key_exported, test_group_key_shared_across_scripts,
#        test_dead_resize_helpers_removed
# DEPENDS: re, pathlib, packages.common.near_dup
# LINKS: M-COMMON, M-EVAL
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# ROOT - корень репозитория для чтения исходников в структурных проверках
# test_group_key_matches_previous_algorithm - group_key совпадает с прежней реализацией на наборе записей
# test_group_key_exported - group_key объявлен в __all__ near_dup
# test_group_key_shared_across_scripts - четыре скрипта импортируют group_key из near_dup, локальных копий нет
# test_dead_resize_helpers_removed - в трёх eval-скриптах нет мёртвых _resize_bgr/_MAX_SIDE/импорта flatten_alpha
# END_MODULE_MAP
"""Тесты рефакторинга C-084.

Проверяют две вещи: (1) общий `group_key` поведенчески идентичен прежним четырём копиям `_group_key`;
(2) дублирование и мёртвый код действительно удалены (структурная проверка исходников).
"""
from __future__ import annotations

import re
from pathlib import Path

from packages.common.near_dup import group_key

__all__ = ["test_dead_resize_helpers_removed", "test_group_key_exported",
           "test_group_key_matches_previous_algorithm", "test_group_key_shared_across_scripts"]

ROOT = Path(__file__).resolve().parents[1]
_SAMPLES = [
    {"winery": "Fanagoria", "name": "Рислинг 2023"},
    {"winery": "  Шато Тамань  ", "name": "Брют Резерв"},
    {"winery": "", "name": "Вино 0.75 л"},
    {"winery": "W", "name": "A" * 60},
    {"winery": "Коктебель", "name": "Кокур сухое 2025"},
    {},
    {"name": "Только название 12"},
]


def _previous_algorithm(record: dict) -> str:
    """Точная копия прежней `_group_key` до C-084 (эталон эквивалентности)."""
    winery = (record.get("winery") or "").strip().lower()
    name = " ".join(re.sub(r"\d+", " ", (record.get("name") or "").lower()).split())[:48]
    return f"{winery}|{name}"


def test_group_key_matches_previous_algorithm() -> None:
    """Общий group_key даёт тот же результат, что прежние четыре копии."""
    for record in _SAMPLES:
        assert group_key(record) == _previous_algorithm(record)


def test_group_key_exported() -> None:
    """group_key объявлен как публичный символ near_dup."""
    import packages.common.near_dup as near_dup_module

    assert "group_key" in near_dup_module.__all__


def test_group_key_shared_across_scripts() -> None:
    """Скрипты используют общий group_key; локальных определений _group_key не осталось."""
    consumers = [
        "scripts/eval/vlm_fallback_eval.py",
        "scripts/eval/ocr_signal_eval.py",
        "scripts/eval/baseline_report.py",
        "scripts/kb/build_label_kb.py",
    ]
    for relative in consumers:
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "from packages.common.near_dup import group_key" in source, relative
        assert "def _group_key" not in source, relative


def test_dead_resize_helpers_removed() -> None:
    """В трёх eval-скриптах удалены мёртвые _resize_bgr/_MAX_SIDE и неиспользуемый flatten_alpha."""
    cleaned = [
        "scripts/eval/vlm_fallback_eval.py",
        "scripts/eval/ocr_signal_eval.py",
        "scripts/eval/calibrate.py",
    ]
    for relative in cleaned:
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "_resize_bgr" not in source, relative
        assert "_MAX_SIDE" not in source, relative
        assert "flatten_alpha" not in source, relative
