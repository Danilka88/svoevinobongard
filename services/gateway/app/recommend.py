# START_MODULE_CONTRACT
# PURPOSE: Рекомендации поверх каталога: аналоги из других виноделен и цифровой сомелье с детерминированным fallback
# SCOPE: find_similar, pairing_for, catalog_recommendations, build_sommelier_prompt, parse_sommelier_reply
# DEPENDS: re, packages.common.catalog, packages.common.logging, packages.common.llm.router
# LINKS: M-GATEWAY
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# find_similar - аналоги из других виноделен по метаданным (запасной путь)
# find_similar_official - официальные похожие вина портала через API
# official_dishes - официальные гастросочетания вина по сорту, региону и цвету с объяснением
# pairing_for - детерминированные гастрономические пары по цвету и сладости
# catalog_recommendations - подбор вин по ответам пользователя без LLM
# build_sommelier_prompt - промпт для текстовой модели сомелье
# parse_sommelier_reply - разбор ответа модели в рекомендации и пары
# END_MODULE_MAP
"""Retention-логика: аналоги и цифровой сомелье.

Аналоги считаются по метаданным каталога: совпадение сорта, региона и цвета
дают скор, винодельня искомого вина исключается — ТЗ прямо требует аналоги
«из других виноделен».

Сомелье пытается получить ответ у текстовой модели через LLM Router, но при
любом сбое (нет модели, таймаут, битый JSON) откатывается к детерминированному
подбору по каталогу: фича остаётся рабочей даже без LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from packages.common.catalog import CatalogStore
from packages.common.official_api import absolute_url
from packages.common.logging import log_marker

__all__ = ["find_similar", "find_similar_official", "official_dishes", "pairing_for", "catalog_recommendations",
           "build_sommelier_prompt", "parse_sommelier_reply"]

_MAX_ITEMS = 8


@dataclass
class Similarity:
    """Скор похожести с расшифровкой совпадений."""

    score: float = 0.0
    reasons: list[str] = field(default_factory=list)


def _norm(value: str | None) -> str:
    """Нижний регистр без лишних пробелов."""
    return " ".join((value or "").lower().split())


def find_similar(store: CatalogStore, slug: str, limit: int = 6) -> list[dict]:
    """Ищет аналоги среди других виноделен.

    Args:
        store: read-only каталог.
        slug: исходное вино.
        limit: максимум результатов.

    Returns:
        Список словарей slug/name/winery/region/category/grape/score/reason,
        отсортированный по убыванию похожести. Пустой, если вино не найдено.
    """
    base = store.get(slug)
    if not base:
        return []
    base_winery = _norm(base.get("winery"))
    base_grape = _norm(base.get("grape"))
    base_region = _norm(base.get("region"))
    base_color = _norm(base.get("category"))

    scored: list[tuple[float, list[str], dict]] = []
    for other_slug, record in store.slug_to_record().items():
        if other_slug == slug:
            continue
        record_winery = _norm(record.get("winery"))
        if base_winery and record_winery == base_winery:
            continue  # ТЗ: аналоги именно из ДРУГИХ виноделен
        score = 0.0
        reasons: list[str] = []
        if base_grape and base_grape == _norm(record.get("grape")):
            score += 0.5
            reasons.append("тот же сорт")
        if base_color and base_color == _norm(record.get("category")):
            score += 0.2
            reasons.append("тот же цвет")
        if base_region and base_region == _norm(record.get("region")):
            score += 0.15
            reasons.append("тот же регион")
        else:
            score += 0.1
            reasons.append("другой регион")
        if not reasons:
            continue
        scored.append((score, reasons, record))

    scored.sort(key=lambda item: (-item[0], _norm(item[2].get("name"))))
    log_marker("M-GATEWAY", "find_similar", "DONE", base=slug, found=len(scored))
    return [
        {
            "slug": record["slug"],
            "name": record.get("name", ""),
            "winery": record.get("winery", ""),
            "region": record.get("region", ""),
            "category": record.get("category", ""),
            "grape": record.get("grape", ""),
            "score": round(score, 4),
            "reason": ", ".join(reasons),
        }
        for score, reasons, record in scored[:limit]
    ]


def official_dishes(record: dict) -> list[str]:
    """Официальные гастросочетания вина из API (C-017/C-018).

    Это источник истины: сомелье не выдумывает пары, а берёт их из данных портала.
    """
    return [str(dish) for dish in (record.get("dishes") or []) if dish]


def find_similar_official(slug: str, limit: int = 6, *, refresh: bool = False) -> list[dict]:
    """Похожие вина из официального API.

    Returns:
        Список словарей в формате SimilarItem или пустой список, если API недоступен
        (тогда вызывающий использует запасную логику по метаданным).
    """
    from packages.common.official_api import fetch_similar  # noqa: PLC0415

    payload = fetch_similar(slug, size=limit, refresh=refresh)
    if not payload:
        return []
    items: list[dict] = []
    for entry in payload:
        image = entry.get("image") or {}
        items.append({
            "slug": str(entry.get("slug") or ""),
            "name": str(entry.get("title") or ""),
            "winery": str(entry.get("manufacturer") or ""),
            "region": str(entry.get("region") or ""),
            "category": str(entry.get("category") or ""),
            "grape": "",
            "score": 1.0,
            "reason": "официальные похожие вина портала",
            "image_url": absolute_url(image.get("url") if isinstance(image, dict) else ""),
            "source": "official",
        })
    log_marker("M-GATEWAY", "similar_official", "DONE", slug=slug, items=len(items))
    return [item for item in items if item["slug"]]


def pairing_for(record: dict, official: list[str] | None = None) -> list[str]:
    """Гастрономические пары: приоритет у официальных блюд портала, иначе по цвету и сладости."""
    dishes = official if official is not None else official_dishes(record)
    if dishes:
        return dishes
    haystack = " ".join(_norm(record.get(field)) for field in ("category", "sweetness", "name"))
    if "игрист" in haystack or "брют" in haystack:
        return ["Икра и морепродукты", "Мягкие сыры", "Устрицы"]
    if "белое" in haystack:
        if "сладкое" in haystack:
            return ["Голубые сыры", "Фуа-гра", "Десерты с ягодами"]
        return ["Рыба и морепродукты", "Птица", "Мягкие и полутвёрдые сыры"]
    if "розовое" in haystack:
        return ["Салаты и овощи гриль", "Молодые сыры", "Лёгкая закуска"]
    if "оранжевое" in haystack:
        return ["Пряные блюда", "Азиатская кухня", "Твёрдые сыры"]
    if "красное" in haystack:
        if "сладкое" in haystack:
            return ["Шоколадные десерты", "Голубые сыры", "Сухофрукты"]
        return ["Мясо на гриле", "Выдержанные сыры", "Тушёные блюда"]
    return ["Сырная тарелка", "Закуски"]


def catalog_recommendations(store: CatalogStore, slug: str, answers: dict[str, str],
                            limit: int = 5) -> list[dict]:
    """Подбирает вина по каталогу без LLM: фильтр по ответам + похожесть."""
    base = store.get(slug) or {}
    wanted_color = _norm(answers.get("style") or answers.get("wine_type"))
    budget = _norm(answers.get("budget"))
    dish = _norm(answers.get("dish"))

    candidates: list[dict] = []
    for other_slug, record in store.slug_to_record().items():
        if other_slug == slug:
            continue
        if wanted_color and wanted_color not in _norm(record.get("category")) \
                and wanted_color not in _norm(record.get("sweetness")):
            continue
        if dish:
            haystack = _norm(f"{record.get('category')} {record.get('sweetness')}")
            if dish in {"рыба", "морепродукты"} and "красное" in haystack:
                continue
            if dish in {"мясо", "стейк"} and "белое" in haystack:
                continue
        score = 0.0
        if _norm(record.get("grape")) and _norm(record.get("grape")) == _norm(base.get("grape")):
            score += 0.3
        if _norm(record.get("region")) == _norm(base.get("region")):
            score += 0.2
        if _norm(record.get("winery")) != _norm(base.get("winery")):
            score += 0.2
        if budget and "низ" in budget:
            score += 0.1
        candidates.append({"slug": other_slug, "record": record, "score": score})

    candidates.sort(key=lambda item: (-item["score"], _norm(item["record"].get("name"))))
    return [
        {
            "slug": item["record"]["slug"],
            "name": item["record"].get("name", ""),
            "winery": item["record"].get("winery", ""),
            "why": "похожее вино из каталога: совпадают сорт, регион или цвет",
        }
        for item in candidates[:limit]
    ]


def build_sommelier_prompt(record: dict, answers: dict[str, str], candidates: list[dict]) -> list[dict]:
    """Собирает промпт для текстовой модели сомелье."""
    options = "; ".join(f"{item['slug']} ({item['name']})" for item in candidates[:5])
    user = (
        f"Вино: {record.get('name', '')} — {record.get('winery', '')}, "
        f"{record.get('region', '')}, сорт {record.get('grape', '')}, "
        f"цвет {record.get('category', '')}, сладость {record.get('sweetness', '')}.\n"
        f"Предпочтения гостя: {answers or 'не указаны'}.\n"
        f"Кандидаты из каталога: {options or 'нет'}.\n"
        'Верни СТРОГО JSON: {"recommendations":[{"slug":"","why":""}],"pairing":["",""]}. '
        "Только JSON."
    )
    return [
        {"role": "system", "content": "Ты цифровой сомелье российских вин. Отвечай только JSON."},
        {"role": "user", "content": user},
    ]


def parse_sommelier_reply(text: str, allowed_slugs: set[str]) -> dict:
    """Разбирает ответ модели, оставляя только существующие slug.

    Returns:
        Словарь recommendations и pairing; пустой, если JSON не восстановлен.
    """
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return {}
    import json

    try:
        payload = json.loads(match.group(0))
    except ValueError:
        return {}
    if not isinstance(payload, dict):
        return {}
    recommendations = []
    for item in payload.get("recommendations") or []:
        if not isinstance(item, dict):
            continue
        slug = str(item.get("slug", "")).strip()
        if slug in allowed_slugs:
            recommendations.append({"slug": slug, "why": str(item.get("why", ""))[:200]})
    pairing = [str(item)[:80] for item in (payload.get("pairing") or []) if str(item).strip()]
    if not recommendations and not pairing:
        return {}
    return {"recommendations": recommendations, "pairing": pairing}
