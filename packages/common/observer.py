# START_MODULE_CONTRACT
# PURPOSE: Режим наблюдателя: описание этикетки моделью и поиск по всему каталогу с гибридом к визуальному топ-5
# SCOPE: OBSERVER_INSTRUCTION, OBSERVATION_KEYS, ENUM_SYNONYMS, escalation_trigger, parse_observation,
#        observation_fields, observer_candidates, merge_candidates
# DEPENDS: json, packages.common.text_norm, packages.common.text_rank
# LINKS: M-EVAL, M-GATEWAY, M-VLM
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# OBSERVER_INSTRUCTION - инструкция наблюдателя: описывать видимое и не искать товар
# OBSERVATION_KEYS - ключи наблюдения, которые переносятся в поля поиска
# KEY_ALIASES - варианты написания ключей наблюдения
# ENUM_SYNONYMS - синонимы значений перечислений на русском и английском
# escalation_trigger - нужен ли режим наблюдателя
# parse_observation - разбор наблюдения: строки «ключ: значение», JSON или свободный текст
# observation_fields - поля для поиска, нормализованные по синонимам и написанию
# observer_candidates - поиск по всему каталогу по наблюдению
# merge_candidates - гибрид наблюдателя и визуального топа с бустом за совпадение
# END_MODULE_MAP
"""Режим наблюдателя.

Дешёвый визуальный поиск не всегда уверен, а галерея эталонов не покрывает новые и перерисованные
этикетки. Тогда модель получает роль наблюдателя: описывает то, что видно, без попытки назвать
товар. Каталог остаётся источником истины — поиск идёт по всем записям, и модель не выбирает SKU.

Формат наблюдения намеренно нетребовательный: строки «ключ: значение» (частая форма у компактных
моделей), свободный текст или JSON. Разбор и нормализация выполняются на нашей стороне.
"""
from __future__ import annotations

import json

from packages.common.text_norm import fold, greek_to_cyrillic, similarity
from packages.common.text_rank import text_search

OBSERVER_INSTRUCTION = (
    "Опиши этикетку на фото. Пиши только то, что видно, и не придумывай текст. "
    "Не ищи товар в каталоге и не называй артикулы.\n"
    "Ответ строго в таком виде (по одной строке на поле, пустые поля пропускай):\n"
    "winery: <винодельня или бренд>\n"
    "wine_name: <название вина>\n"
    "vintage: <год числом>\n"
    "grape: <сорт>\n"
    "sweetness: <сухое|полусухое|полусладкое|сладкое|брют>\n"
    "color: <красное|белое|розовое|игристое>\n"
    "wine_type: <тихое|игристое|креплёное>\n"
    "raw_text: <остальной читаемый текст этикетки>\n"
    "Пример ответа:\n"
    "winery: Скалистый берег\n"
    "wine_name: Шёпот цветов\n"
    "vintage: 2024\n"
    "grape: Рислинг\n"
    "sweetness: брют\n"
    "raw_text: SHOPOT CVETOV | 0.75 L\n"
    "Если текст не читается, напиши raw_text: неразборчиво. Не пересказывай эту инструкцию."
)

OBSERVATION_KEYS = ("winery", "wine_name", "vintage", "grape", "sweetness", "color", "wine_type",
                    "region", "volume", "brand_text")
# Инструкция наблюдателя просит ключи на английском, поэтому канонические имена — тоже псевдонимы.
KEY_ALIASES = {
    "winery": "winery",
    "wine_name": "wine_name",
    "name": "wine_name",
    "vintage": "vintage",
    "year": "vintage",
    "grape": "grape",
    "sweetness": "sweetness",
    "sugar": "sweetness",
    "color": "color",
    "colour": "color",
    "wine_type": "wine_type",
    "type": "wine_type",
    "region": "region",
    "volume": "volume",
    "raw_text": "raw_text",
    "винодельня": "winery", "производитель": "winery", "бренд": "winery", "brand": "winery",
    "название": "wine_name", "вино": "wine_name", "name": "wine_name", "wine": "wine_name",
    "год": "vintage", "урожай": "vintage", "year": "vintage", "миллезим": "vintage",
    "сорт": "grape", "виноград": "grape", "grapes": "grape", "variety": "grape",
    "сладость": "sweetness", "сахар": "sweetness", "sugar": "sweetness", "sweet": "sweetness",
    "цвет": "color", "colour": "color", "тип": "wine_type", "kind": "wine_type", "style": "wine_type",
    "регион": "region", "appellation": "region", "объём": "volume", "объем": "volume", "volume": "volume",
    "raw_text": "raw_text", "текст": "raw_text", "текст этикетки": "raw_text",
}
ENUM_SYNONYMS = {
    "sweetness": {"брют": "брют", "brut": "брют", "dry": "сухое", "сухой": "сухое", "dry white": "сухое",
                  "semi-dry": "полусухое", "полусухой": "полусухое", "sweet": "сладкое", "сладкий": "сладкое",
                  "semi-sweet": "полусладкое", "полусладкий": "полусладкое", "extra brut": "экстрабрют",
                  "demi-sec": "полубрют", "sec": "брют"},
    "color": {"red": "красное", "white": "белое", "rose": "розовое", "rosé": "розовое", "pink": "розовое",
              "sparkling": "игристое", "orange": "оранжевое", "красный": "красное", "белый": "белое",
              "розовый": "розовое"},
    "wine_type": {"still": "тихое", "sparkling": "игристое", "fortified": "креплёное",
                  "dessert": "десертное", "крепленое": "креплёное"},
}

__all__ = ["OBSERVER_INSTRUCTION", "OBSERVATION_KEYS", "KEY_ALIASES", "ENUM_SYNONYMS",
           "escalation_trigger", "parse_observation", "observation_fields", "observer_candidates",
           "merge_candidates"]


def escalation_trigger(p_correct: float | None, confident: bool, threshold: float = 0.95) -> bool:
    """Нужен ли наблюдатель: по калиброванной вероятности, иначе по уверенности поиска.

    Если калибровка доступна (p_correct не пусто), решает она — это вероятностная шкала. Иначе
    используется прежний признак уверенности поиска, поэтому поведение без калибровки предсказуемо.
    """
    if p_correct is not None:
        return float(p_correct) < float(threshold)
    return not confident


def _clean(value: str) -> str:
    return str(value or "").strip().strip('"').strip("'").strip("«»").strip()


def parse_observation(text: str | None) -> dict:
    """Разбирает наблюдение модели в структуру.

    Поддерживаются три формы: строки «ключ: значение» (основная), JSON-объект (если модель его
    вернула), свободный текст (уходит целиком в raw_text). Лишние строки не теряются: всё, что не
    распознано как ключ, попадает в raw_text — так наблюдение остаётся полным.
    """
    raw = str(text or "").strip()
    if not raw:
        return {"raw_text": ""}
    payload: dict = {}
    candidate = raw
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        candidate = candidate.split("\n", 1)[-1] if "\n" in candidate else candidate
    if candidate.strip().startswith("{"):
        try:
            loaded = json.loads(candidate)
            if isinstance(loaded, dict):
                payload = {str(key): value for key, value in loaded.items()}
        except (ValueError, TypeError):
            payload = {}
    if not payload:
        leftovers: list[str] = []
        for line in raw.replace("“", '"').replace("”", '"').splitlines():
            stripped = line.strip(" -*•\t")
            if not stripped:
                continue
            separator = ":" if ":" in stripped else ("=" if "=" in stripped else None)
            if separator:
                key, _, value = stripped.partition(separator)
                name = KEY_ALIASES.get(fold(key).replace(" ", "_"), "")
                if not name:
                    name = KEY_ALIASES.get(fold(key), "")
                if name:
                    # Известный ключ забираем даже с пустым значением: это скелет шаблона, который
                    # модель напечатала целиком, и он не должен засорять свободный текст.
                    value_text = _clean(value)
                    if name == "raw_text":
                        if value_text:
                            payload["raw_text"] = value_text
                    elif value_text:
                        payload[name] = value_text
                    continue
            leftovers.append(stripped)
        if leftovers and "raw_text" not in payload:
            payload["raw_text"] = " | ".join(leftovers)[:600]
    result: dict = {}
    for key, value in payload.items():
        # Свободный текст наблюдения переносим явно: он не должен теряться из-за нормализации ключа.
        if str(key).strip().lower() in {"raw_text", "текст", "text"}:
            raw_value = _clean(value if not isinstance(value, (list, dict)) else
                               json.dumps(value, ensure_ascii=False))
            if raw_value:
                result["raw_text"] = raw_value[:600]
            continue
        name = KEY_ALIASES.get(fold(str(key)), str(key))
        if name not in OBSERVATION_KEYS:
            continue
        text_value = _clean(value if not isinstance(value, (list, dict)) else json.dumps(value, ensure_ascii=False))
        if text_value:
            result[name] = text_value[:200]
    return result


def observation_fields(observation: dict, *, normalize: bool = True) -> dict:
    """Поля для поиска: значения нормализуются по синонимам, транслиту и регистру.

    Свободный текст этикетки сохраняется отдельно: он используется как дополнительный сигнал, но не
    подменяет структурные поля.
    """
    fields: dict = {}
    for key, value in (observation or {}).items():
        if key in ("raw_text", "brand_text"):
            continue
        # Омоглифы приводим к кириллице до сравнения с каталогом: модель читает кириллицу
        # греческими буквами, и без этого поиск промахивается.
        text = greek_to_cyrillic(_clean(str(value))) if normalize else _clean(str(value))
        if not text:
            continue
        if key in ENUM_SYNONYMS:
            folded = fold(text)
            mapped = ENUM_SYNONYMS[key].get(folded) or ENUM_SYNONYMS[key].get(text.lower())
            if mapped is None:
                best, best_score = "", 0.0
                for candidate in set(ENUM_SYNONYMS[key].values()):
                    score = similarity(text, candidate)
                    if score > best_score:
                        best, best_score = candidate, score
                mapped = best if best_score >= 0.72 else text
            text = mapped
        fields[key] = text
    if not fields:
        # Модель часто кладёт всё в свободный текст (пустые поля шаблона). Тогда ищем по тексту:
        # это тот же сигнал, просто без разметки по полям.
        raw = _clean(str((observation or {}).get("raw_text") or ""))
        if raw:
            fields["wine_name"] = greek_to_cyrillic(raw) if normalize else raw
    return fields


def observer_candidates(observation: dict, catalog: dict, *, limit: int = 10,
                        normalize: bool = True) -> list[tuple[str, float]]:
    """Ищет по всему каталогу по наблюдению (текстовый поиск, без ограничения визуального топа)."""
    fields = observation_fields(observation, normalize=normalize)
    if not fields:
        return []
    hits = text_search(fields, catalog, limit=limit)
    return [(slug, float(score)) for slug, score in hits]


def merge_candidates(observer_hits: list[tuple[str, float]], image_topk: list[tuple[str, float]],
                     *, boost: float = 0.15, observer_weight: float = 1.0) -> list[tuple[str, float]]:
    """Гибрид наблюдения и визуального топа.

    Совпадение с визуальным топом даёт буст: два независимых сигнала на один slug убедительнее.
    Кандидат наблюдателя, которого нет в визуальном топе, может победить — именно так находятся
    новые и перерисованные этикетки, выпавшие из галереи эталонов.
    """
    combined: dict[str, float] = {}
    for slug, score in image_topk:
        combined[slug] = max(combined.get(slug, 0.0), float(score))
    for slug, score in observer_hits:
        value = float(score) * observer_weight
        if slug in combined:
            value = min(1.0, value + boost)
        combined[slug] = max(combined.get(slug, 0.0), value)
    return sorted(combined.items(), key=lambda item: (-item[1], item[0]))
