# START_MODULE_CONTRACT
# PURPOSE: Тесты C-085 реестра брендов: winery_key из каталога, разрешение омоглифов, устойчивость
# SCOPE: CATALOG, test_winery_key_of, test_build_index_from_catalog, test_resolve_exact_and_homoglyph,
#        test_resolve_unknown_is_empty, test_add_variant, test_index_size_and_sync
# DEPENDS: packages.common.brands
# LINKS: M-COMMON
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# CATALOG - синтетический каталог для проверки индекса брендов
# test_winery_key_of - ключ винодельни: транслит, только буквы/цифры/дефис
# test_build_index_from_catalog - индекс строится из каталога (ключи и варианты)
# test_resolve_exact_and_homoglyph - точная кириллица и омоглифы резолвятся в ключ
# test_resolve_unknown_is_empty - неизвестный бренд даёт пустой ключ
# test_add_variant - добавление варианта написания
# test_index_size_and_sync - размер индекса и пополнение при росте каталога
# END_MODULE_MAP
"""Тесты реестра брендов C-085."""
from __future__ import annotations

from packages.common.brands import (add_variant, build_winery_index, index_size, resolve_winery,
                                    winery_key_of)

CATALOG = {
    "abrau-dyurso-fizz": {"winery": "Абрау-Дюрсо", "name": "Физз"},
    "kuban-vino-shato-taman": {"winery": "Кубань-Вино", "name": "Шато Тамань"},
    "fanagoriya-beloe": {"winery": "Фанагория", "name": "Белое"},
}

__all__ = ["CATALOG", "test_add_variant", "test_build_index_from_catalog", "test_index_size_and_sync",
           "test_resolve_exact_and_homoglyph", "test_resolve_unknown_is_empty", "test_winery_key_of"]


def test_winery_key_of() -> None:
    """Ключ винодельни — транслит каталожного имени, только буквы/цифры/дефис."""
    assert winery_key_of("Абрау-Дюрсо") == "abraudyurso"
    assert winery_key_of("Кубань-Вино") == "kubanvino"
    assert winery_key_of("") == ""


def test_build_index_from_catalog() -> None:
    """Индекс строится из каталога: есть ключи брендов."""
    index = build_winery_index(CATALOG)
    assert index["wineries"]["Абрау-Дюрсо"] == "abraudyurso"
    assert "abraudyurso" in index["variants"]


def test_resolve_exact_and_homoglyph() -> None:
    """Точная кириллица резолвится, а омоглифы (лат. O в кириллице) сводятся к тому же ключу."""
    index = build_winery_index(CATALOG)
    assert resolve_winery("Абрау-Дюрсо", index) == "abraudyurso"
    assert resolve_winery("АБРАУ-ДЮРСO", index) == "abraudyurso"   # латинская O
    assert resolve_winery("Фанагория", index) == "fanagoriya"


def test_resolve_unknown_is_empty() -> None:
    """Неизвестный бренд даёт пустой ключ (не выдумываем)."""
    index = build_winery_index(CATALOG)
    assert resolve_winery("Несуществующий Бренд", index) == ""
    assert resolve_winery("", index) == ""


def test_add_variant() -> None:
    """Добавление варианта написания к ключу (обогащение)."""
    index = build_winery_index(CATALOG)
    assert add_variant(index, "abraudyurso", "Abrau-Durso") is True
    assert resolve_winery("Abrau-Durso", index) == "abraudyurso"
    assert add_variant(index, "abraudyurso", "Abrau-Durso") is False   # повтор не добавляется


def test_index_size_and_sync() -> None:
    """Размер индекса отражает число брендов; рост каталога пополняет индекс."""
    index = build_winery_index(CATALOG)
    assert index_size(index) == 3
    grown = dict(CATALOG)
    grown["new"] = {"winery": "Новый Бренд", "name": "Вино"}
    assert index_size(build_winery_index(grown)) == 4
