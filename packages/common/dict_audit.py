# START_MODULE_CONTRACT
# PURPOSE: Реестр жёстких словарей проекта: видно, где словарь устаревает, чтобы обновлять его по данным
# SCOPE: DICTIONARIES, dict_inventory
# DEPENDS: importlib, packages.common
# LINKS: M-COMMON
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# DICTIONARIES - реестр жёстких словарей: модуль, символ, назначение
# dict_inventory - сводка по словарям: размер и назначение (устойчива к отсутствующим модулям)
# END_MODULE_MAP
"""Реестр жёстких словарей (C-079).

Принцип проекта: жёсткий словарь — детерминированный ФОЛБЭК, а LLM-нормализация (C-077) — ОБОБЩЕНИЕ
поверх него. Отказываться от словарей нельзя (скорость, воспроизводимость, работа без LLM), но и
считать их полными тоже: бренды меняют написание, появляются новые сорта и производители. Реестр делает
устаревание словаря наблюдаемым: по нему видно размер и назначение каждого словаря, и его можно
сопоставлять с покрытием подтверждённых пар (C-078).
"""
from __future__ import annotations

import importlib

__all__ = ["DICTIONARIES", "dict_inventory"]

# (модуль, символ, назначение) — целевые жёсткие словари проекта.
DICTIONARIES: tuple[tuple[str, str, str], ...] = (
    ("packages.common.text_norm", "ALIASES", "синонимы сортов/брендов лат↔кир"),
    ("packages.common.text_norm", "LOOKALIKES", "визуальные двойники латиницы"),
    ("packages.common.text_norm", "GREEK_LOOKALIKES", "греческие двойники кириллицы"),
    ("packages.common.text_norm", "LATIN_LOOKALIKES_CYR", "латинские двойники кириллицы"),
    ("packages.common.observer", "KEY_ALIASES", "варианты ключей наблюдения"),
    ("packages.common.observer", "ENUM_SYNONYMS", "синонимы enum сладости/цвета/типа"),
    ("packages.common.observer", "PLACEHOLDER_VALUES", "заглушки модели («неразборчиво»)"),
    ("packages.common.llm.structured", "KEY_ALIASES", "варианты ключей структурированного вывода"),
    ("packages.common.llm.structured", "ENUM_SYNONYMS", "синонимы enum структурированного вывода"),
    ("packages.common.near_dup", "_SWEETNESS_ALIASES", "синонимы сладости для near-dup"),
    ("packages.common.near_dup", "_COLOR_ALIASES", "синонимы цвета для near-dup"),
    ("services.matcher.app.textfusion", "_COLOR_MAP", "карта цвета для текстового фьюжна"),
)


def _size(value: object) -> int:
    """Размер словаря/набора/списка; для прочего — 0."""
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return 0


def dict_inventory() -> dict:
    """Сводка по жёстким словарям: размер и назначение.

    Отсутствующий модуль или символ не роняют функцию — запись помечается ``missing``: реестр должен
    переживать рефакторинг и честно показывать, что словарь исчез, а не падать.
    """
    inventory: dict[str, dict] = {}
    for module_name, symbol, purpose in DICTIONARIES:
        key = f"{module_name}.{symbol}"
        try:
            module = importlib.import_module(module_name)
            value = getattr(module, symbol, None)
        except Exception:  # noqa: BLE001 - отсутствующий модуль не должен ронять аудит
            inventory[key] = {"size": 0, "purpose": purpose, "missing": True}
            continue
        inventory[key] = {"size": _size(value) if value is not None else 0,
                          "purpose": purpose, "missing": value is None}
    return inventory
