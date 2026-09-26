# START_MODULE_CONTRACT
# PURPOSE: Тесты C-079 реестра жёстких словарей: находит целевые словари, возвращает размеры, устойчив
# SCOPE: test_inventory_finds_target_dicts, test_inventory_sizes_positive, test_inventory_stable_to_missing,
#        test_dictionaries_registry_sane
# DEPENDS: packages.common.dict_audit
# LINKS: M-COMMON
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# test_inventory_finds_target_dicts - все целевые словари присутствуют в сводке
# test_inventory_sizes_positive - размеры найденных словарей положительны
# test_inventory_stable_to_missing - отсутствующий модуль не роняет сводку
# test_dictionaries_registry_sane - реестр содержит путь, символ и назначение для каждого словаря
# END_MODULE_MAP
"""Тесты реестра жёстких словарей C-079."""
from __future__ import annotations

from packages.common.dict_audit import DICTIONARIES, dict_inventory

__all__ = ["test_dictionaries_registry_sane", "test_inventory_finds_target_dicts",
           "test_inventory_sizes_positive", "test_inventory_stable_to_missing"]


def test_dictionaries_registry_sane() -> None:
    """Реестр описывает модуль, символ и назначение для каждого словаря."""
    assert DICTIONARIES
    for module_name, symbol, purpose in DICTIONARIES:
        assert module_name and symbol and purpose


def test_inventory_finds_target_dicts() -> None:
    """Сводка содержит все целевые словари проекта."""
    inventory = dict_inventory()
    for module_name, symbol, _purpose in DICTIONARIES:
        assert f"{module_name}.{symbol}" in inventory


def test_inventory_sizes_positive() -> None:
    """Размеры ключевых словарей положительны (они реально заполнены)."""
    inventory = dict_inventory()
    assert inventory["packages.common.text_norm.GREEK_LOOKALIKES"]["size"] > 0
    assert inventory["packages.common.observer.ENUM_SYNONYMS"]["size"] > 0
    assert inventory["packages.common.near_dup._SWEETNESS_ALIASES"]["size"] > 0


def test_inventory_stable_to_missing(monkeypatch) -> None:
    """Отсутствующий модуль помечается missing и не роняет сводку."""
    import packages.common.dict_audit as audit

    monkeypatch.setattr(audit, "DICTIONARIES",
                        (("nonexistent.module.xyz", "ALIASES", "тест"),))
    inventory = audit.dict_inventory()
    assert inventory["nonexistent.module.xyz.ALIASES"]["missing"] is True
