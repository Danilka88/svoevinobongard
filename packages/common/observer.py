# START_MODULE_CONTRACT
# PURPOSE: Режим наблюдателя: описание этикетки моделью и поиск по всему каталогу с гибридом к визуальному топ-5
# SCOPE: OBSERVER_INSTRUCTION, OBSERVER_STRUCTURED_INSTRUCTION, OBSERVATION_KEYS, ENUM_SYNONYMS,
#        PLACEHOLDER_VALUES, escalation_trigger, parse_observation, parse_observation_structured,
#        is_informative, observation_fields, observer_candidates, merge_candidates
# DEPENDS: json, packages.common.text_norm, packages.common.text_rank, packages.common.llm.structured
# LINKS: M-EVAL, M-GATEWAY, M-VLM
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# OBSERVER_INSTRUCTION - инструкция наблюдателя: описывать видимое и не искать товар
# OBSERVER_STRUCTURED_INSTRUCTION - строгая JSON-схема наблюдения для моделей, способных её удержать
# OBSERVATION_KEYS - ключи наблюдения, которые переносятся в поля поиска
# KEY_ALIASES - варианты написания ключей наблюдения
# ENUM_SYNONYMS - синонимы значений перечислений на русском и английском
# PLACEHOLDER_VALUES - заглушки модели («неразборчиво», «не указано»), не являющиеся наблюдением
# escalation_trigger - нужен ли режим наблюдателя
# parse_observation - разбор наблюдения: строки «ключ: значение», JSON или свободный текст
# parse_observation_structured - разбор по JSON-схеме с откатом на терпимый «ключ: значение»
# is_informative - есть ли в наблюдении осмысленное значение (а не только заглушки)
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

from packages.common.llm.structured import parse_structured, repair_json
from packages.common.text_norm import fold, homoglyphs_to_cyrillic, similarity
from packages.common.text_rank import text_search

OBSERVER_INSTRUCTION = (
    "Опиши этикетку на фото. Пиши только то, что видно, и не придумывай текст. "
    "Шрифт бывает графичным, рукописным и стилизованным: читай буквы как они нарисованы, "
    "не додумывай знакомые слова, не подставляй известные бренды и не выдумывай год. "
    "Винодельня (бренд) и название вина — разные строки, не смешивай их. "
    "Не ищи товар в каталоге и не называй артикулы.\n"
    "Ответ строго в таком виде (по одной строке на поле, пустые поля пропускай):\n"
    "winery: <винодельня или бренд>\n"
    "wine_name: <название вина>\n"
    "vintage: <год числом>\n"
    "grape: <сорт>\n"
    "sweetness: <сухое|полусухое|полусладкое|сладкое|брют>\n"
    "color: <красное|белое|розовое|игристое>\n"
    "wine_type: <тихое|игристое|креплёное>\n"
    "volume: <объём, например 0.75 L>\n"
    "raw_text: <остальной читаемый текст этикетки>\n"
    "Пример ответа:\n"
    "winery: Скалистый берег\n"
    "wine_name: Шёпот цветов\n"
    "vintage: 2024\n"
    "grape: Рислинг\n"
    "sweetness: брют\n"
    "raw_text: SHOPOT CVETOV | 0.75 L\n"
    "Если текст не читается, оставь поля пустыми и напиши raw_text: неразборчиво. "
    "Не пересказывай эту инструкцию."
)

# C-073: строгая схема для моделей, способных удержать JSON. Онлайн-путь по умолчанию остаётся
# терпимым («ключ: значение»), а этот вариант используется там, где модель отвечает структурно.
OBSERVER_STRUCTURED_INSTRUCTION = (
    "Опиши этикетку на фото. Пиши только то, что видно, и не придумывай текст. "
    "Винодельня (бренд) и название вина — разные поля. Ответ строго одним JSON-объектом:\n"
    '{"winery": "", "wine_name": "", "vintage": "", "grape": "", "sweetness": "", '
    '"color": "", "wine_type": "", "volume": "", "raw_text": ""}\n'
    "Пустые и нечитаемые поля оставляй пустыми строками, без пояснений вокруг JSON."
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

# C-073: заглушки, которыми модель честно сообщает «не прочитал». Это не наблюдение: если их
# принять за значения, поиск по каталогу вернёт случайное вино по слову «неразборчиво».
PLACEHOLDER_VALUES = frozenset({
    "неразборчиво", "неразборчив", "неразборчивый", "не читается", "нечитаемо", "не прочитано",
    "не указано", "не указан", "не указана", "не найдено", "не найдено на этикетке", "нет",
    "нет данных", "нет информации", "отсутствует", "неизвестно", "неизв", "не видно",
    "unknown", "none", "null", "n/a", "na", "n a", "not found", "n а", "н/д", "нд", "-", "--", "—", "–",
    "没有", "无法辨认",
})
_PLACEHOLDER_FOLDED = frozenset(fold(value) for value in PLACEHOLDER_VALUES) | {""}

__all__ = ["OBSERVER_INSTRUCTION", "OBSERVER_STRUCTURED_INSTRUCTION", "OBSERVATION_KEYS", "KEY_ALIASES",
           "ENUM_SYNONYMS", "PLACEHOLDER_VALUES", "escalation_trigger", "parse_observation",
           "parse_observation_structured", "is_informative", "observation_fields", "observer_candidates",
           "merge_candidates"]


def _is_placeholder(value: object) -> bool:
    """Является ли значение заглушкой модели, а не прочитанным текстом."""
    text = str(value or "").strip()
    if not text:
        return True
    folded = fold(text)
    if folded:
        # «(нет)», «[неразборчиво]» и подобные скобочные формы сводятся к тому же множеству.
        stripped = folded.strip("()[]{}<>«»\"'").strip()
        return stripped in _PLACEHOLDER_FOLDED
    # fold выбрасывает не-латиницу/кириллицу (например, греческие буквы): если в исходной строке
    # есть буквы, это прочитанный текст, а не заглушка.
    return not any(char.isalpha() for char in text)


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
        loaded = None
        try:
            loaded = json.loads(candidate)
        except (ValueError, TypeError):
            # C-073: «грязный» JSON (фенсы, хвостовые запятые, обрыв) восстанавливается общим
            # ремонтом из structured.py, а не отбрасывается целиком.
            loaded = repair_json(candidate)
        if isinstance(loaded, dict):
            payload = {str(key): value for key, value in loaded.items()}
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
                    if value_text:
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
            # C-073: «raw_text: неразборчиво» — это честный отказ, а не текст для поиска.
            if raw_value and not _is_placeholder(raw_value):
                result["raw_text"] = raw_value[:600]
            continue
        name = KEY_ALIASES.get(fold(str(key)), str(key))
        if name not in OBSERVATION_KEYS:
            continue
        text_value = _clean(value if not isinstance(value, (list, dict)) else json.dumps(value, ensure_ascii=False))
        # C-073: заглушки («не указано», «(нет)», «unknown») не становятся полями поиска.
        if text_value and not _is_placeholder(text_value):
            result[name] = text_value[:200]
    return result


def parse_observation_structured(text: str | None) -> dict:
    """Разбирает наблюдение по JSON-схеме, откатываясь на терпимый «ключ: значение».

    Схема ``observer`` из ``structured.py`` гарантирует, что модель отдала ровно ожидаемые поля
    (ремонт фенсов, «умных» кавычек, обрывов). Если ответ не является JSON-объектом по схеме —
    используется обычный :func:`parse_observation`, поэтому поведение на прежних форматах не меняется.
    """
    if not str(text or "").strip():
        return {"raw_text": ""}
    parsed = parse_structured(str(text), "observer")
    if parsed.get("valid") and parsed.get("data"):
        return parse_observation(json.dumps(parsed["data"], ensure_ascii=False))
    return parse_observation(text)


def is_informative(observation: dict | None) -> bool:
    """Есть ли в наблюдении осмысленный текст, а не только заглушки.

    Инвариант против случайной выдачи: если модель вернула только «неразборчиво»/«не указано»
    или один год без названия, наблюдение не должно порождать кандидатов каталога. Считаются
    только текстовые поля (винодельня, название, сорт, регион, свободный текст) с двумя и более
    буквами: год, сладость и цвет сами по себе вино не идентифицируют.
    """
    data = observation or {}
    for key in ("winery", "wine_name", "grape", "region", "raw_text", "brand_text"):
        value = data.get(key)
        if isinstance(value, (list, dict)):
            value = json.dumps(value, ensure_ascii=False)
        text = str(value or "").strip()
        if _is_placeholder(text):
            continue
        if sum(1 for char in text if char.isalpha()) >= 2:
            return True
    return False


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
        # греческими буквами, а внутри смешанных токенов — ещё и латинскими двойниками;
        # чистые латинские бренды при этом не портятся (C-073).
        text = homoglyphs_to_cyrillic(_clean(str(value))) if normalize else _clean(str(value))
        if not text or _is_placeholder(text):
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
        if raw and not _is_placeholder(raw):
            fields["wine_name"] = homoglyphs_to_cyrillic(raw) if normalize else raw
    elif "wine_name" not in fields:
        # C-073: модель разметила только часть полей, но свободный текст этикетки информативен.
        # Раньше он терялся (использовался лишь при полностью пустых полях), и поиск шёл по одному
        # году. Подключаем свободный текст как название, когда названия среди полей нет.
        raw = _clean(str((observation or {}).get("raw_text") or ""))
        if raw and not _is_placeholder(raw):
            fields["wine_name"] = homoglyphs_to_cyrillic(raw) if normalize else raw
    return fields


def observer_candidates(observation: dict, catalog: dict, *, limit: int = 10,
                        normalize: bool = True) -> list[tuple[str, float]]:
    """Ищет по всему каталогу по наблюдению (текстовый поиск, без ограничения визуального топа).

    Инвариант C-073: неинформативное наблюдение (одни заглушки или пустота) кандидатов не даёт —
    иначе на честный отказ модели выдавалось бы случайное вино.
    """
    if not is_informative(observation):
        return []
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
