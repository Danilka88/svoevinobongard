# START_MODULE_CONTRACT
# PURPOSE: Маркер-логирование в GRACE-формате [MODULE][function][BLOCK] для машинно-читаемых доказательств
# SCOPE: log_marker
# DEPENDS: none (stdlib)
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
"""
from __future__ import annotations

import sys

__all__ = ["log_marker"]


def log_marker(module: str, function: str, block: str, **fields: object) -> None:
    """Печатает маркер выполнения в stderr.

    Args:
        module: идентификатор модуля, например ``M-RETRIEVAL``.
        function: имя функции или операции.
        block: имя семантического блока (UPPER_SNAKE).
        **fields: дополнительные пары ключ-значение.
    """
    suffix = "".join(f" {key}={value}" for key, value in fields.items())
    print(f"[{module}][{function}][{block}]{suffix}", file=sys.stderr, flush=True)
