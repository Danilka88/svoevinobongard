# START_MODULE_CONTRACT
# PURPOSE: Тесты C-083 маркер-логирования: эмиссия через logging, формат и поля маркера
# SCOPE: test_marker_emitted_via_logging, test_marker_format_and_fields, test_marker_level_info
# DEPENDS: packages.common.logging
# LINKS: M-COMMON
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# test_marker_emitted_via_logging - маркер эмитится через logging (перехватывается assertLogs)
# test_marker_format_and_fields - формат [MODULE][function][BLOCK] и поля сохраняются
# test_marker_level_info - маркер идёт на уровне INFO
# END_MODULE_MAP
"""Тесты маркер-логирования C-083."""
from __future__ import annotations

import logging

from packages.common.logging import log_marker

__all__ = ["test_marker_emitted_via_logging", "test_marker_format_and_fields",
           "test_marker_level_info"]


def test_marker_emitted_via_logging() -> None:
    """Маркер эмитится через logging-логгер (перехватывается assertLogs)."""
    with __import__("unittest").TestCase().assertLogs("grace.markers", level="INFO") as captured:
        log_marker("M-CATALOG", "ETL", "START")
    assert any("[M-CATALOG][ETL][START]" in line for line in captured.output)


def test_marker_format_and_fields() -> None:
    """Формат [MODULE][function][BLOCK] и дополнительные поля сохраняются."""
    with __import__("unittest").TestCase().assertLogs("grace.markers", level="INFO") as captured:
        log_marker("M-CATALOG", "parse_sitemap", "DONE", entries=2037)
    line = captured.output[0]
    assert "[M-CATALOG][parse_sitemap][DONE]" in line and "entries=2037" in line


def test_marker_level_info() -> None:
    """Уровень маркера — INFO (логгер grace.markers)."""
    logger = logging.getLogger("grace.markers")
    assert logger.level == logging.INFO
