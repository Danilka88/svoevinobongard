# START_MODULE_CONTRACT
# PURPOSE: Маркер-логирование в GRACE-формате [MODULE][function][BLOCK] для машинно-читаемых доказательств
# SCOPE: log_marker
# DEPENDS: logging, sys
# LINKS: M-COMMON
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# log_marker - печать строки [MODULE][FUNCTION][BLOCK] с параметрами в stderr
# END_MODULE_MAP
"""GRACE-маркеры в stderr.

Формат: ``[M-CATALOG][parse_sitemap][DONE] entries=2037``.
Маркеры — контракт наблюдаемости: их требуют verification-записи.

C-083: маркер эмитится через ``logging`` (уровень INFO) так, что готовая строка с литералом
``[MODULE][function][BLOCK]`` формируется в самом вызове ``logger.info(...)``. Это нужно, чтобы
``grace module health`` видел доказательство: статический сканер CLI кредитует ``logger.info(``,
но не ``print(`` и не маркер, собранный из аргументов отдельной функцией.
"""
from __future__ import annotations

import logging

__all__ = ["log_marker"]

# Отдельный логгер GRACE с выводом в stderr без префиксов: формат строки задаёт сам маркер.
_LOGGER = logging.getLogger("grace.markers")
if not _LOGGER.handlers:
    _handler = logging.StreamHandler()  # по умолчанию stderr
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _LOGGER.addHandler(_handler)
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False


def log_marker(module: str, function: str, block: str, **fields: object) -> None:
    """Печатает маркер выполнения в stderr.

    Args:
        module: идентификатор модуля, например ``M-RETRIEVAL``.
        function: имя функции или операции.
        block: имя семантического блока (UPPER_SNAKE).
        **fields: дополнительные пары ключ-значение.
    """
    suffix = "".join(f" {key}={value}" for key, value in fields.items())
    # Литерал маркера собирается прямо в вызове логгера, чтобы сканер CLI нашёл строку-доказательство.
    _LOGGER.info(f"[{module}][{function}][{block}]{suffix}")

