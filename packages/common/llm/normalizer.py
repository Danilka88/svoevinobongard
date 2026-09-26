# START_MODULE_CONTRACT
# PURPOSE: LLM-нормализация чтения этикетки под каталог: приведение винодельни и названия к каталожной форме
# SCOPE: NORMALIZER_WINERY_INSTRUCTION, NORMALIZER_FIELDS_INSTRUCTION, build_catalog_vocab,
#        build_winery_prompt, build_fields_prompt, parse_winery_answer, parse_fields_answer,
#        normalize_reading, reset_normalizer_cache
# DEPENDS: json, packages.common.catalog, packages.common.llm.ollama_native, packages.common.observer,
#          packages.common.text_norm
# LINKS: M-COMMON, M-VLM, M-GATEWAY
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# NORMALIZER_WINERY_INSTRUCTION - инструкция первого запроса: назвать каталожную винодельню
# NORMALIZER_FIELDS_INSTRUCTION - инструкция второго запроса: название и параметры
# build_catalog_vocab - динамический словарь каталога: винодельни, регионы, перечисления
# build_winery_prompt - промпт запроса винодельни: список каталога + чтение
# build_fields_prompt - промпт запроса названия и параметров
# parse_winery_answer - разбор ответа винодельни (винодельня + бренд) из обычного текста
# parse_fields_answer - разбор ответа названия/параметров из обычного текста
# normalize_reading - два запроса к маленькой LLM + фолбэк на транслит; кэш в памяти
# reset_normalizer_cache - очистка кэша (тесты и диагностика)
# END_MODULE_MAP
"""LLM-нормализация чтения под каталог (C-077).

Маленькая локальная модель (gemma4:e4b) приводит прочитанную этикетку к каталожной форме: винодельня,
бренд, название, сладость, цвет, тип, регион. В промпт подставляются актуальные значения каталога
(винодельни, регионы, перечисления), поэтому новые бренды и переименования подхватываются без словаря.
Два узких текстовых запроса надёжнее одного большого и, в отличие от JSON, посильны маленькой модели.

Нормализатор не принимает решений о товаре: он лишь приводит текст к каталожной форме. Slug выбирает
матчер из реального пула кандидатов. При недоступной LLM используется детерминированный фолбэк
(транслит/омоглифы из text_norm), поэтому путь не обрывается.
"""
from __future__ import annotations

from packages.common.llm.ollama_native import ollama_native_chat
from packages.common.observer import ENUM_SYNONYMS
from packages.common.text_norm import fold, homoglyphs_to_cyrillic

__all__ = ["NORMALIZER_FIELDS_INSTRUCTION", "NORMALIZER_WINERY_INSTRUCTION", "build_catalog_vocab",
           "build_fields_prompt", "build_winery_prompt", "normalize_reading", "parse_fields_answer",
           "parse_winery_answer", "reset_normalizer_cache"]

DEFAULT_NORMALIZER_MODEL = "gemma4:e4b-mlx"

NORMALIZER_WINERY_INSTRUCTION = (
    "Ты приводишь производителя вина к каталожному названию. Ниже — список производителей каталога. "
    "Назови РОВНО ОДНОГО производителя из списка, которому принадлежит этикетка, даже если на этикетке "
    "он написан латиницей, по-французски или иначе (Château Tamagne → Кубань-Вино; Abrau-Durso → "
    "Абрау-Дюрсо). Бренд с этикетки — это не производитель: «Шато Тамань»/«Château Tamagne» относится к "
    "производителю «Кубань-Вино». Если точного совпадения нет, назови самое близкое из списка.\n"
    "Ответ строго двумя строками:\n"
    "winery: <производитель из списка или пусто>\n"
    "brand: <бренд/линейка с этикетки, например Шато Тамань>\n"
    "Никаких пояснений, только эти две строки."
)

NORMALIZER_FIELDS_INSTRUCTION = (
    "Ты приводишь название вина и параметры этикетки к каталожной форме. Используй только эти значения:\n"
    "сладость: {sweetness}\n"
    "цвет: {colors}\n"
    "тип: {types}\n"
    "регион: {regions}\n"
    "Ответ строго строками (пустые пропускай), без пояснений:\n"
    "wine_name: <название вина по-русски, если возможно>\n"
    "sweetness: <одно значение из списка сладости>\n"
    "color: <одно значение из списка цвета>\n"
    "wine_type: <одно значение из списка типа>\n"
    "region: <одно значение из списка регионов>"
)

_CACHE: dict[str, dict] = {}
_MAX_CACHE = 512


def reset_normalizer_cache() -> None:
    """Очищает кэш нормализации (тесты и диагностика)."""
    _CACHE.clear()


def build_catalog_vocab(catalog: dict) -> dict:
    """Строит динамический словарь каталога для промптов.

    Returns:
        {'wineries': [...], 'regions': [...], 'sweetness': [...], 'colors': [...], 'types': [...]}
        со отсортированными уникальными значениями. При росте каталога список пополняется сам.
    """
    wineries, regions, sweetness, colors, types = set(), set(), set(), set(), set()
    for record in (catalog or {}).values():
        data = record or {}
        for key, bucket in (("winery", wineries), ("region", regions), ("sweetness", sweetness),
                            ("color", colors), ("type", types)):
            value = str(data.get(key) or "").strip()
            if value:
                bucket.add(value)
    return {"wineries": sorted(wineries), "regions": sorted(regions),
            "sweetness": sorted(sweetness), "colors": sorted(colors), "types": sorted(types)}


def _clean(value: object) -> str:
    """Однострочное текстовое значение."""
    return " ".join(str(value or "").split()).strip()


def build_winery_prompt(observation: dict, vocab: dict) -> str:
    """Промпт запроса винодельни: список каталога + чтение этикетки."""
    wineries = "\n".join(f"- {name}" for name in vocab.get("wineries", []))
    observed = "; ".join(f"{key}: {_clean(observation.get(key))}"
                         for key in ("winery", "wine_name", "raw_text") if _clean(observation.get(key)))
    return (f"{NORMALIZER_WINERY_INSTRUCTION}\n\n"
            f"Список виноделен каталога:\n{wineries}\n\n"
            f"Этикетка прочитана так:\n{observed or '— (пусто)'}\n")


def build_fields_prompt(observation: dict, vocab: dict, brand: str = "") -> str:
    """Промпт запроса названия и параметров."""
    instruction = NORMALIZER_FIELDS_INSTRUCTION.format(
        sweetness=", ".join(vocab.get("sweetness", [])) or "сухое, полусухое, полусладкое, сладкое, брют",
        colors=", ".join(vocab.get("colors", [])) or "красное, белое, розовое, оранжевое",
        types=", ".join(vocab.get("types", [])) or "тихое, игристое, креплёное",
        regions=", ".join(vocab.get("regions", [])) or "—",
    )
    observed = "; ".join(f"{key}: {_clean(observation.get(key))}"
                         for key in ("winery", "wine_name", "raw_text") if _clean(observation.get(key)))
    brand_line = f"Бренд: {brand}\n" if brand else ""
    return (f"{instruction}\n\n{brand_line}"
            f"Этикетка прочитана так:\n{observed or '— (пусто)'}\n")


def _lines(text: str) -> dict[str, str]:
    """Разбирает обычный текстовый ответ «ключ: значение» (терпимо к мусору)."""
    result: dict[str, str] = {}
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip().lstrip("-*•").strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        name = fold(key).replace(" ", "_")
        cleaned = value.strip().strip("「」\"'`«»").strip()
        if cleaned:
            result[name] = cleaned
    return result


def parse_winery_answer(text: str | None, vocab: dict) -> dict:
    """Разбор ответа первого запроса: каталожная винодельня и бренд.

    Если названное не совпало с каталогом дословно, ищем ближайшее вхождение; иначе винодельня пустая,
    но бренд сохраняется — поиск всё равно попробует его.
    """
    lines = _lines(str(text or ""))
    winery = lines.get("winery", "") or lines.get("винодельня", "") or lines.get("производитель", "")
    brand = lines.get("brand", "") or lines.get("бренд", "")
    folded = fold(winery)
    if folded:
        exact = next((name for name in vocab.get("wineries", []) if fold(name) == folded), "")
        if exact:
            winery = exact
        else:
            partial = next((name for name in vocab.get("wineries", [])
                            if folded and (folded in fold(name) or fold(name) in folded)), "")
            winery = partial or winery
    return {"winery": _clean(winery), "brand": _clean(brand)}


def _enum(name: str, value: str) -> str:
    """Приводит значение перечисления по синонимам; пустое остаётся пустым."""
    folded = fold(value)
    if not folded:
        return ""
    synonyms = ENUM_SYNONYMS.get(name, {})
    if folded in synonyms:
        return synonyms[folded]
    for key, target in synonyms.items():
        if fold(key) == folded:
            return target
    return value


def parse_fields_answer(text: str | None, vocab: dict) -> dict:
    """Разбор ответа второго запроса: название и параметры из обычного текста."""
    lines = _lines(str(text or ""))
    out = {
        "wine_name": lines.get("wine_name", "") or lines.get("название", "") or lines.get("name", ""),
        "sweetness": _enum("sweetness", lines.get("sweetness", "") or lines.get("сладость", "")),
        "color": _enum("color", lines.get("color", "") or lines.get("цвет", "")),
        "wine_type": _enum("wine_type", lines.get("wine_type", "") or lines.get("тип", "") or lines.get("вид", "")),
        "region": lines.get("region", "") or lines.get("регион", ""),
    }
    return {key: _clean(value) for key, value in out.items()}


def _fallback(observation: dict) -> dict:
    """Детерминированный фолбэк: транслит/омоглифы без LLM."""
    def norm(value: object) -> str:
        return homoglyphs_to_cyrillic(_clean(value))

    return {"winery": norm(observation.get("winery")), "brand": "",
            "wine_name": norm(observation.get("wine_name")), "sweetness": _enum("sweetness", _clean(observation.get("sweetness"))),
            "color": _enum("color", _clean(observation.get("color"))),
            "wine_type": _enum("wine_type", _clean(observation.get("wine_type"))),
            "region": norm(observation.get("region")), "source": "fallback"}


def normalize_reading(observation: dict, catalog: dict, *, model: str | None = None,
                      timeout_s: float = 30.0, max_tokens: int = 120, use_cache: bool = True,
                      num_gpu: int | None = 0) -> dict:
    """Нормализует чтение этикетки под каталог двумя текстовыми запросами к маленькой LLM.

    Args:
        observation: поля, прочитанные с этикетки (winery, wine_name, raw_text, sweetness, ...).
        catalog: записи каталога slug → record (источник словаря виноделен/регионов/перечислений).
        model: имя локальной модели (по умолчанию gemma4:e4b-mlx).
        timeout_s: таймаут одного запроса.
        max_tokens: лимит генерации ответа.
        use_cache: использовать кэш в памяти по сырому чтению.
        num_gpu: сколько слоёв отдать GPU (0 — CPU). По умолчанию 0, чтобы маленькая модель не
            вытесняла тяжёлого читателя (26b) из VRAM во время офлайн-прогона.

    Returns:
        Стандартизированное наблюдение: winery, brand, wine_name, sweetness, color, wine_type, region,
        raw_text и source ('llm' или 'fallback'). Никогда не поднимает исключение.
    """
    observed = observation or {}
    cache_key = "|".join(_clean(observed.get(key)) for key in
                         ("winery", "wine_name", "raw_text", "sweetness", "color", "wine_type", "region"))
    if use_cache and cache_key in _CACHE:
        return dict(_CACHE[cache_key])
    vocab = build_catalog_vocab(catalog)
    selected = str(model or DEFAULT_NORMALIZER_MODEL)
    winery_answer = ollama_native_chat(build_winery_prompt(observed, vocab), selected,
                                       max_tokens=max_tokens, timeout_s=timeout_s, fmt="text",
                                       num_gpu=num_gpu)
    if winery_answer is None:
        result = _fallback(observed)
        result["raw_text"] = _clean(observed.get("raw_text"))
        if use_cache and len(_CACHE) < _MAX_CACHE:
            _CACHE[cache_key] = dict(result)
        return result
    winery_part = parse_winery_answer(winery_answer, vocab)
    fields_answer = ollama_native_chat(build_fields_prompt(observed, vocab, winery_part["brand"]), selected,
                                       max_tokens=max_tokens, timeout_s=timeout_s, fmt="text",
                                       num_gpu=num_gpu)
    result: dict = {}
    if fields_answer is None:
        result = _fallback(observed)
    else:
        fields = parse_fields_answer(fields_answer, vocab)
        # Пустые поля не затирают исходное чтение: берём прочитанное, если модель не дала значения.
        for key in ("wine_name", "sweetness", "color", "wine_type", "region"):
            result[key] = fields.get(key) or _clean(observed.get(key))
        result["winery"] = winery_part["winery"] or _fallback(observed)["winery"]
        result["brand"] = winery_part["brand"]
        result["source"] = "llm"
    result["raw_text"] = _clean(observed.get("raw_text")) or _clean(fields_answer)
    if use_cache and len(_CACHE) < _MAX_CACHE:
        _CACHE[cache_key] = dict(result)
    return result
