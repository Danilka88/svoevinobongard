# START_MODULE_CONTRACT
# PURPOSE: Верификация нормализации текста: транслит лат/кир, OCR-двойники, алиасы сортов и брендов
# SCOPE: test_fold_normalizes, test_delookalike_ocr_doubles, test_cyr_to_lat_slug, test_lat_to_cyr_name, test_alias_variants_match, test_similarity_latin_vs_cyrillic, test_tokens_keep_distinguishing_words, test_similarity_no_false_identity
# DEPENDS: packages.common.text_norm, pytest
# LINKS: M-COMMON, M-MATCHER, V-M-TEXT
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# test_fold_normalizes - базовая нормализация снимает регистр и пунктуацию
# test_delookalike_ocr_doubles - OCR-двойники латиницы заменяются на кириллицу
# test_cyr_to_lat_slug - кириллица переводится в латиницу для сравнения со slug
# test_lat_to_cyr_name - латиница переводится в кириллицу для сравнения с названием
# test_alias_variants_match - алиасы сводят варианты сорта, типа и бренда к одной основе
# test_similarity_latin_vs_cyrillic - латинская этикетка совпадает с кириллическим названием каталога
# test_tokens_keep_distinguishing_words - различающие слова и годы сохраняются токенами
# test_similarity_no_false_identity - разные вина не отождествляются норм
# END_MODULE_MAP
"""V-M-TEXT: нормализация текста для сопоставления."""
from __future__ import annotations

from packages.common.text_norm import (
    apply_aliases,
    cyr_to_lat,
    delookalike,
    fold,
    forms,
    lat_to_cyr,
    similarity,
    tokens,
)

__all__ = [
    "test_alias_variants_match",
    "test_cyr_to_lat_slug",
    "test_delookalike_ocr_doubles",
    "test_fold_normalizes",
    "test_lat_to_cyr_name",
    "test_similarity_latin_vs_cyrillic",
    "test_similarity_no_false_identity",
    "test_tokens_keep_distinguishing_words",
]


def test_fold_normalizes() -> None:
    """Нормализация снимает регистр, диакритику и пунктуацию."""
    assert fold("  Шато   Тамань,  Полусладкое! ") == "шато тамань полусладкое"
    assert fold("Rosé d'Anjou") == "rose danjou"


def test_delookalike_ocr_doubles() -> None:
    """Латинские двойники, которые выдаёт OCR на кириллице, заменяются обратно."""
    assert delookalike("cyxoe") == "сухое"
    assert delookalike("BИHO") == "вино"


def test_cyr_to_lat_slug() -> None:
    """Кириллица переводится в латиницу так, чтобы совпадать со slug каталога."""
    assert cyr_to_lat("Пино Нуар") == "pino nuar"
    assert cyr_to_lat("Рислинг") == "risling"


def test_lat_to_cyr_name() -> None:
    """Латиница переводится в кириллицу для сравнения с русским названием."""
    assert lat_to_cyr("risling") == "рислинг"
    assert lat_to_cyr("shardone") == "шардоне"


def test_alias_variants_match() -> None:
    """Алиасы сводят варианты сорта, типа сладости и бренда к одной основе."""
    assert apply_aliases("sauvignon blanc") == apply_aliases("совиньон блан")
    assert apply_aliases("semi-sweet") == apply_aliases("полусладкое")
    assert apply_aliases("chateau tamagne") == apply_aliases("шато тамань")
    assert apply_aliases("millesimato") == apply_aliases("миллезимато")


def test_similarity_latin_vs_cyrillic() -> None:
    """Латинская этикетка совпадает с кириллическим названием каталога."""
    assert similarity("Шато Тамань", "chateau tamagne") == 1.0
    assert similarity("Пино Нуар", "pino-nuar") == 1.0
    assert similarity("Фанагория", "fanagoriya") == 1.0
    assert similarity("Каберне Совиньон", "kaberne-sovinon") >= 0.95
    assert similarity("сухое", "dry") == 1.0
    assert similarity("Крым", "krym") == 1.0


def test_tokens_keep_distinguishing_words() -> None:
    """Различающие слова и годы остаются токенами и не схлопываются."""
    left = tokens("Рислинг Рейнский 2022")
    right = tokens("Рислинг 2022")
    assert "рейнский" in left and "рейнский" not in right
    assert "2022" in left and "2022" in right


def test_similarity_no_false_identity() -> None:
    """Разные вина не отождествляются: бренд без названия не даёт полного совпадения."""
    assert similarity("Рислинг", "Шардоне") < 0.6
    assert similarity("Шато Тамань Резерв", "Абрау Дюрсо Резерв") < 0.75
    assert forms("Рислинг Рейнский") != forms("Рислинг")
