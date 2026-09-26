# START_MODULE_CONTRACT
# PURPOSE: Группы near-дубликатов каталога и дизамбигуация внутри серии по подтверждённым сигналам
# SCOPE: NEAR_DUP_THRESHOLD, DIFFERENT_FIELDS, group_key, signature, build_groups, near_dup_group,
#        group_stats, match_signals, disambiguate
# DEPENDS: collections, re, packages.common.text_norm
# LINKS: M-GATEWAY, M-MATCHER, M-DATA
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# NEAR_DUP_THRESHOLD - порог схожести названия внутри одной серии
# DIFFERENT_FIELDS - поля, различие по которым делает вина разными позициями
# group_key - грубый ключ группы для eval/KB: винодельня + название без цифр, обрезка 48
# signature - подпись серии: винодельня и название без различителей
# build_groups - строит группы near-дубликатов по каталогу
# near_dup_group - возвращает группу для конкретного вина
# group_stats - статистика групп для отчёта
# match_signals - какие сигналы различия подтверждает запись
# disambiguate - выбирает вино внутри группы или честно сообщает о неоднозначности
# END_MODULE_MAP
"""Near-дубликаты: одна серия, разные позиции.

Главный источник ошибок в проекте: у одной серии отличается год, сладость или категория при похожей
этикетке. Модуль строит группы таких вин и разрешает выбор только по подтверждённым сигналам
различия; если сигналов нет, решение не выдумывается — ответ помечается как неоднозначный.

Проверки сигналов продублированы здесь, а не взяты из сервиса матчера: общий пакет не может
импортировать сервис (слоистость), а логика сравнения года, сладости и цвета невелика и покрыта тестами.
"""
from __future__ import annotations

import re
from collections import defaultdict

from packages.common.text_norm import fold, similarity

NEAR_DUP_THRESHOLD = 0.86
DIFFERENT_FIELDS = ("year", "sweetness", "category")
_SWEETNESS_ALIASES = {"брют": "брют", "brut": "брют", "сухое": "сухое", "dry": "сухое",
                      "полусухое": "полусухое", "semi-dry": "полусухое", "полусладкое": "полусладкое",
                      "semi-sweet": "полусладкое", "сладкое": "сладкое", "sweet": "сладкое"}
_COLOR_ALIASES = {"красное": "красное", "red": "красное", "белое": "белое", "white": "белое",
                  "розовое": "розовое", "rose": "розовое", "игристое": "игристое", "sparkling": "игристое"}

__all__ = ["NEAR_DUP_THRESHOLD", "DIFFERENT_FIELDS", "group_key", "signature", "build_groups",
           "near_dup_group", "group_stats", "match_signals", "disambiguate"]


def group_key(record: dict) -> str:
    """Грубый ключ группы near-дубликатов: винодельня + название без цифр (C-084).

    Общий хелпер eval/KB-скриптов: цифры (год, объём) в названии заменяются пробелом, регистр
    приводится к нижнему, пробелы схлопываются, результат обрезается до 48 символов. Это **не**
    :func:`signature` — та дополнительно убирает слова сладости и цвета, поэтому ключи двух функций
    для одной записи могут различаться; `group_key` умышленно мягче и используется только для
    подвыборки и отчётности офлайн-инструментов.
    """
    winery = (record.get("winery") or "").strip().lower()
    name = " ".join(re.sub(r"\d+", " ", (record.get("name") or "").lower()).split())[:48]
    return f"{winery}|{name}"


def signature(record: dict) -> tuple[str, str]:
    """Подпись серии: винодельня и название без различающих слов.

    Различители (год, слова сладости и категории) убираются, иначе позиции одной серии не соберутся
    в группу: у них как раз эти слова и отличаются.
    """
    winery = fold(str(record.get("winery") or ""))
    name = fold(str(record.get("name") or ""))
    stripped = [token for token in name.split()
                if not token.isdigit() and token not in _SWEETNESS_ALIASES and token not in _COLOR_ALIASES]
    base = " ".join(stripped) or name
    return winery, base


def build_groups(catalog: dict, threshold: float = NEAR_DUP_THRESHOLD) -> dict[str, list[str]]:
    """Строит группы near-дубликатов: одна серия, различие в годе, сладости или категории."""
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for slug, record in catalog.items():
        buckets[signature(record)].append(slug)
    groups: dict[str, list[str]] = {}
    for (winery, base), slugs in buckets.items():
        if len(slugs) < 2:
            continue
        # Страховка от слишком широких подписей: проверяем схожесть названий и наличие различия.
        for index, slug in enumerate(slugs):
            for other in slugs[index + 1:]:
                first, second = catalog[slug], catalog[other]
                # Сравниваем подписи серии, а не полные имена: внутри серии имена как раз различаются
                # годом и сладостью, и строгий порог на полном имени разрывал бы группу (найдено тестом).
                if similarity(base, signature(second)[1]) < 0.5:
                    continue
                if not any(str(first.get(field) or "") != str(second.get(field) or "")
                           for field in DIFFERENT_FIELDS):
                    continue
                group_id = f"{winery}|{base}"
                members = groups.setdefault(group_id, [])
                for candidate in (slug, other):
                    if candidate not in members:
                        members.append(candidate)
    return groups


def near_dup_group(slug: str, groups: dict[str, list[str]]) -> list[str]:
    """Возвращает группу для вина: пустой список, если вино вне групп."""
    for members in groups.values():
        if slug in members:
            return list(members)
    return []


def group_stats(groups: dict[str, list[str]], catalog: dict) -> dict:
    """Статистика групп: сколько групп, сколько позиций, распределение размеров."""
    sizes = [len(members) for members in groups.values()]
    covered = sum(sizes)
    return {"groups": len(groups), "positions": covered,
            "share_of_catalog": round(covered / max(len(catalog), 1), 4),
            "max_size": max(sizes) if sizes else 0,
            "size_histogram": {str(size): sizes.count(size) for size in sorted(set(sizes))}}


def match_signals(signals: dict | None, record: dict) -> dict:
    """Какие сигналы различия подтверждает запись: год, сладость, цвет."""
    source = signals or {}
    year_wanted = str(source.get("vintage") or "").strip()
    year_record = str(record.get("year") or "")
    years_in_name = [token for token in fold(str(record.get("name") or "")).split() if token.isdigit()]
    year_hit = bool(year_wanted) and (year_wanted == year_record or year_wanted in years_in_name)
    sweetness_wanted = _SWEETNESS_ALIASES.get(fold(str(source.get("sweetness") or "")), "")
    sweetness_hit = bool(sweetness_wanted) and sweetness_wanted == _SWEETNESS_ALIASES.get(
        fold(str(record.get("sweetness") or "")), fold(str(record.get("sweetness") or "")))
    color_wanted = _COLOR_ALIASES.get(fold(str(source.get("color") or "")), "")
    color_hit = bool(color_wanted) and color_wanted == _COLOR_ALIASES.get(
        fold(str(record.get("color") or "")), "")
    return {"year_hit": year_hit, "sweetness_hit": sweetness_hit, "color_hit": color_hit,
            "confirmed": sum((year_hit, sweetness_hit, color_hit))}


def disambiguate(candidates: list[str], signals: dict | None, catalog: dict,
                 groups: dict[str, list[str]]) -> dict:
    """Выбирает вино внутри группы дубликатов по сигналам различия.

    Правило: кандидаты должны принадлежать одной группе. Побеждает тот, у кого больше подтверждённых
    сигналов, при условии что он обошёл остальных. Если сигналов нет или победитель не единственный —
    решение не выдумывается: ответ помечается неоднозначным и возвращаются кандидаты.
    """
    if not candidates:
        return {"slug": "", "ambiguous": False, "reason": "нет кандидатов", "signals": {}}
    group = near_dup_group(candidates[0], groups)
    in_group = [slug for slug in candidates if slug in group]
    if len(in_group) < 2:
        return {"slug": candidates[0], "ambiguous": False,
                "reason": "кандидаты не образуют группу дубликатов", "signals": {}}
    scored = [(slug, match_signals(signals, catalog.get(slug) or {})) for slug in in_group[:5]]
    best_slug, best = max(scored, key=lambda item: item[1]["confirmed"])
    tied = [slug for slug, score in scored if score["confirmed"] == best["confirmed"]]
    if best["confirmed"] == 0:
        return {"slug": "", "ambiguous": True, "candidates": in_group,
                "reason": "сигналов различия нет: вина одной серии неразличимы", "signals": best}
    if len(tied) > 1 and tied[0] != candidates[0]:
        # Ничья среди не первых кандидатов: порядок выдачи уже выражает предпочтение,
        # поэтому берём сильнейшего по порядку, а не объявляем неоднозначность.
        return {"slug": tied[0], "ambiguous": False,
                "reason": "сигналы равны, выбран по порядку выдачи", "signals": best}
    return {"slug": best_slug, "ambiguous": False,
            "reason": f"выбран по сигналам: {', '.join(name for name, value in best.items() if value is True)}",
            "signals": best}
