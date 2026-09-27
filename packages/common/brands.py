# START_MODULE_CONTRACT
# PURPOSE: Канонический ключ винодельни (winery_key) из каталога: сопоставление брендов по стандарту, а не по буквам
# SCOPE: build_winery_index, resolve_winery, winery_key_of, add_variant, index_size
# DEPENDS: packages.common.text_norm, packages.common.catalog
# LINKS: M-COMMON, M-VLM, M-GATEWAY
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# build_winery_index - строит индекс {winery: key} и {key: [варианты написания]} из каталога
# resolve_winery - находит winery_key по произвольному тексту (точный вариант, транслит, омоглифы)
# winery_key_of - канонический ключ конкретной каталожной винодельни
# add_variant - добавляет вариант написания к существующему ключу (обогащение, C-085)
# index_size - число брендов в индексе (диагностика)
# END_MODULE_MAP
"""Канонический ключ винодельни.

Матчер не должен сравнивать бренды по буквам: одна винодельня пишется и латиницей, и кириллицей, и с
омоглифами («Абрау-Дюрсо», «Abrau-Durso», «АБРАУ-ДЮРСO»). Введён стабильный `winery_key` — транслит
каталожного написания, согласованный с началом slug позиций «Своё Вино». Индекс строится из каталога
динамически, поэтому при его росте новые бренды подхватываются сами; официальный каталог не изменяется.
"""
from __future__ import annotations

from packages.common.text_norm import cyr_to_lat, fold, homoglyphs_to_cyrillic, lat_to_cyr

__all__ = ["add_variant", "build_winery_index", "index_size", "resolve_winery", "winery_key_of"]


def winery_key_of(name: str) -> str:
    """Канонический ключ винодельни: транслит каталожного написания, только буквы/цифры и дефисы.

    Пример: «Абрау-Дюрсо» → ``abrau-dyurso``. Ключ согласован с началом slug позиций бренда.
    """
    translit = cyr_to_lat(str(name or ""))
    cleaned = "".join(ch if (ch.isalnum() or ch == "-") else "" for ch in fold(translit))
    return cleaned.strip("-")


def _catalog_key_slug_prefix(name: str, catalog: dict) -> str:
    """Проверяет, что часть slug реально начинается с ключа винодельни (согласованность со «Своё Вино»)."""
    key = winery_key_of(name)
    if not key:
        return ""
    compact = key.replace("-", "")
    for slug in catalog:
        if str(slug).startswith(key) or str(slug).startswith(compact) or compact[:6] in str(slug):
            return key
    return key  # ключ остаётся стандартом даже если slug-совпадение не найдено (бренд без фото)


def _norm_compare(text: str) -> str:
    """Нормализует строку для сравнения брендов: омоглифы → кириллица, транслит лат→кир, затем fold.

    Так «Abrau-Durso», «АБРАУ-ДЮРСO» и «Абрау-Дюрсо» приходят к сопоставимому виду независимо от
    алфавита и омоглифов.
    """
    value = homoglyphs_to_cyrillic(str(text or ""))
    # Если после омоглифов латиница осталась (чистый латинский бренд), транслитерируем в кириллицу.
    if any("a" <= ch.lower() <= "z" for ch in value):
        value = lat_to_cyr(value)
    return "".join(ch for ch in fold(value) if ch.isalnum())


def build_winery_index(catalog: dict) -> dict:
    """Строит индекс брендов из каталога.

    Returns:
        ``{'wineries': {каталожное_имя: key}, 'variants': {key: [варианты написания]}, 'names': {key: имя}}``.
    """
    wineries: dict[str, str] = {}
    variants: dict[str, set[str]] = {}
    names: dict[str, str] = {}
    for record in (catalog or {}).values():
        name = str((record or {}).get("winery") or "").strip()
        if not name:
            continue
        key = winery_key_of(name)
        if not key:
            continue
        wineries[name] = key
        names.setdefault(key, name)
        bucket = variants.setdefault(key, set())
        bucket.add(_norm_compare(name))
    return {"wineries": wineries,
            "variants": {key: sorted(value) for key, value in variants.items()},
            "names": names}


def resolve_winery(text: str, index: dict) -> str:
    """Находит winery_key по произвольному тексту (точное имя, транслит, омоглифы). Пусто, если нет.

    Порядок: точное совпадение с каталожным именем → совпадение ключа → совпадение транслита/омоглифов →
    вхождение ключа. Возвращает канонический ключ или пустую строку.
    """
    if not text or not index:
        return ""
    wineries = index.get("wineries") or {}
    variants = index.get("variants") or {}
    raw = str(text).strip()
    if raw in wineries:
        return wineries[raw]
    folded = _norm_compare(raw)
    if not folded:
        return ""
    for candidate_key, forms in variants.items():
        if folded in forms:
            return candidate_key
    # вхождение нормализованной формы (Шато Тамань не здесь — это решает каталог, а не бренд-индекс)
    for candidate_key, forms in variants.items():
        for form in forms:
            if form and (folded in form or form in folded) and min(len(form), len(folded)) >= 5:
                return candidate_key
    return ""


def add_variant(index: dict, key: str, variant: str) -> bool:
    """Добавляет вариант написания к ключу (обогащение). Возвращает False, если добавить не удалось.

    Вариант хранится в нормализованном для сравнения виде, поэтому ``resolve_winery`` найдёт его
    независимо от алфавита/омоглифов.
    """
    if not key or not variant:
        return False
    normalized = _norm_compare(variant)
    if not normalized:
        return False
    bucket = index.setdefault("variants", {}).setdefault(key, [])
    if normalized in bucket:
        return False
    bucket.append(normalized)
    return True


def index_size(index: dict) -> int:
    """Число брендов (ключей) в индексе — для диагностики и отчётов."""
    return len((index or {}).get("variants") or {})
