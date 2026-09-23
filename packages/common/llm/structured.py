# START_MODULE_CONTRACT
# PURPOSE: Структурированный вывод моделей: восстановление JSON, нормализация ключей и значений, валидация по схемам
# SCOPE: SCHEMAS, repair_json, normalize_keys, normalize_values, validate, parse_structured,
#        provider_response_format, validation_stats, reset_validation_stats
# DEPENDS: json, re, collections, packages.common.text_norm, jsonschema (опционально)
# LINKS: M-VLM, M-GATEWAY, M-CONTRACTS
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# SCHEMAS - схемы ответов: поля этикетки, ответ сомелье, вердикт судьи
# ENUM_SYNONYMS - синонимы значений закрытых полей на русском и английском
# KEY_ALIASES - варианты написания ключей, включая латиницу и кириллицу
# repair_json - восстановление JSON из «грязного» ответа модели
# normalize_keys - приведение ключей к схеме без учёта регистра и алфавита
# normalize_values - приведение значений к допустимым: перечисления, типы, числа
# validate - проверка по схеме с нормализацией: ok, normalized, errors
# parse_structured - оркестратор: восстановление, нормализация, валидация, повтор
# provider_response_format - описание схемы для провайдера, где это поддерживается
# validation_stats - счётчики исходов валидации для телеметрии
# reset_validation_stats - сброс счётчиков (тесты и диагностика)
# END_MODULE_MAP
"""Структурированный вывод моделей.

Модель возвращает текст, а не данные: лишние слова вокруг, фенсы, одинарные и «умные» кавычки,
хвостовые запятые, другой регистр ключей, синонимы значений. Этот модуль приводит такой ответ к
схеме, а если не получается — сообщает об этом явно, не поднимая исключение.

Схема у провайдера (json_schema, grammar) гарантирует синтаксис, но не смысл, и при переходе на
запасного провайдера гарантия исчезает вовсе. Поэтому клиентская проверка выполняется всегда.
"""
from __future__ import annotations

import json
import re
from collections import Counter

from packages.common.text_norm import fold, similarity

try:  # схема проверяется строго, если библиотека доступна
    import jsonschema as _jsonschema
except Exception:  # noqa: BLE001 - без библиотеки работает компактный встроенный валидатор
    _jsonschema = None

__all__ = [
    "SCHEMAS",
    "ENUM_SYNONYMS",
    "KEY_ALIASES",
    "repair_json",
    "normalize_keys",
    "normalize_values",
    "validate",
    "parse_structured",
    "provider_response_format",
    "validation_stats",
    "reset_validation_stats",
]

COLORS = ("красное", "белое", "розовое", "игристое", "оранжевое")
SWEETNESS = ("сухое", "полусухое", "полусладкое", "сладкое", "брют", "полубрют", "экстрабрют")
TYPES = ("тихое", "игристое", "креплёное", "десертное")

SCHEMAS: dict[str, dict] = {
    # (а) поля этикетки: то, что модель читает с фотографии
    "label_fields": {
        "type": "object",
        "properties": {
            "winery": {"type": "string", "maxLength": 120},
            "wine_name": {"type": "string", "maxLength": 160},
            "vintage": {"type": ["integer", "string", "null"]},
            "grape": {"type": "string", "maxLength": 120},
            "sweetness": {"type": ["string", "null"], "enum": list(SWEETNESS) + [None]},
            "color": {"type": ["string", "null"], "enum": list(COLORS) + [None]},
            "wine_type": {"type": ["string", "null"], "enum": list(TYPES) + [None]},
            "region": {"type": "string", "maxLength": 120},
        },
        "required": [],
        "additionalProperties": True,
    },
    # (б) ответ сомелье: рекомендации и сочетания
    "sommelier": {
        "type": "object",
        "properties": {
            "recommendations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"slug": {"type": "string"}, "why": {"type": "string", "maxLength": 300}},
                    "required": ["slug"],
                },
            },
            "pairing": {"type": "array", "items": {"type": "string", "maxLength": 120}},
        },
        "required": [],
    },
    # (в) вердикт судьи: задел под оценку качества ответов
    "judge": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["pass", "fail", "uncertain"]},
            "score": {"type": "number", "minimum": 0, "maximum": 1},
            "reasons": {"type": "array", "items": {"type": "string", "maxLength": 200}},
        },
        "required": ["verdict"],
    },
}

KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "winery": ("winery", "producer", "винодельня", "производитель", "винодельни"),
    "wine_name": ("wine_name", "wine", "name", "title", "название", "наименование", "вино"),
    "vintage": ("vintage", "year", "millésime", "millesime", "год", "урожай"),
    "grape": ("grape", "grapes", "variety", "сорт", "сорта", "виноград"),
    "sweetness": ("sweetness", "sugar", "sweet", "сладость", "сахар", "содержание сахара"),
    "color": ("color", "colour", "цвет"),
    "wine_type": ("wine_type", "type", "kind", "style", "тип", "вид"),
    "region": ("region", "appellation", "регион", "местность"),
    "recommendations": ("recommendations", "items", "wines", "рекомендации"),
    "pairing": ("pairing", "pairings", "dishes", "сочетания", "блюда"),
    "verdict": ("verdict", "result", "вердикт", "итог"),
    "score": ("score", "rating", "оценка", "балл"),
    "reasons": ("reasons", "why", "notes", "причины", "обоснование"),
}

ENUM_SYNONYMS: dict[str, dict[str, str]] = {
    "color": {
        "red": "красное", "white": "белое", "rose": "розовое", "rosé": "розовое",
        "pink": "розовое", "sparkling": "игристое", "orange": "оранжевое",
        "красный": "красное", "белый": "белое", "розовый": "розовое",
        "белое сухое": "белое", "красное сухое": "красное",
    },
    "sweetness": {
        "dry": "сухое", "brut": "брют", "brut nature": "экстрабрют", "extra brut": "экстрабрют",
        "semi-dry": "полусухое", "semi dry": "полусухое", "off-dry": "полусухое",
        "semi-sweet": "полусладкое", "sweet": "сладкое", "demi-sec": "полубрют",
        "sec": "брют", "сухой": "сухое", "полусухой": "полусухое", "сладкий": "сладкое",
        "dry white": "сухое", "brut rose": "брют",
    },
    "wine_type": {
        "still": "тихое", "sparkling": "игристое", "fortified": "креплёное",
        "dessert": "десертное", "крепленое": "креплёное", "игристое вино": "игристое",
    },
    "verdict": {"passed": "pass", "ok": "pass", "failed": "fail", "maybe": "uncertain",
                "принято": "pass", "отказ": "fail"},
}

_STATS: Counter = Counter()


def validation_stats() -> dict:
    """Счётчики исходов разбора структурированных ответов."""
    return {"exact": _STATS["exact"], "repaired": _STATS["repaired"], "retried": _STATS["retried"],
            "failed": _STATS["failed"], "truncated": _STATS["truncated"], "calls": _STATS["calls"]}


def reset_validation_stats() -> None:
    """Сбрасывает счётчики (тесты и диагностика)."""
    _STATS.clear()


def _slice_object(text: str) -> str:
    """Вырезает первый сбалансированный объект, отбрасывая пояснения до и после него.

    Ответ модели часто обрамлён словами («Вот результат:» … «надеюсь, помог»), поэтому просто
    убрать начало недостаточно: хвост тоже мешает разбору. Строки учитываются, чтобы скобки
    внутри значений не сбивали подсчёт.
    """
    value = text or ""
    start = value.find("{")
    if start < 0:
        return value.strip()
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(value)):
        char = value[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return value[start:index + 1]
    return value[start:]  # обрыв ответа: закрывающие скобки достроит балансировка


def _strip_fences(text: str) -> str:
    """Убирает markdown-фенсы и пояснения вокруг JSON."""
    value = (text or "").strip()
    fence = re.search(r"```(?:json|JSON)?\s*(.+?)\s*```", value, re.S)
    if fence:
        value = fence.group(1).strip()
    return _slice_object(value).strip()


def _balance(text: str) -> str:
    """Достраивает незакрытые скобки и кавычки при обрыве ответа."""
    value = text.strip()
    if not value:
        return value
    if value.count('"') % 2 == 1:
        value += '"'
    for opener, closer in (("{", "}"), ("[", "]")):
        missing = value.count(opener) - value.count(closer)
        if missing > 0:
            value += closer * missing
    # незакрытая строка или оборванное значение: убираем хвост до последней запятой
    if re.search(r"[:,]\s*$", value):
        value = value.rstrip(",: \n") + '"' if value.rstrip().endswith(":") else value.rstrip(",: \n")
    return value


def repair_json(text: str) -> dict | None:
    """Восстанавливает объект JSON из «грязного» ответа модели.

    Обрабатываются типовые случаи: фенсы, текст вокруг, одинарные и «умные» кавычки, хвостовые
    запятые, обрыв ответа. Возвращается словарь или None, если восстановить не удалось.
    """
    value = _strip_fences(text)
    if not value:
        return None
    candidates = [value, _balance(value), _slice_object(value), _balance(_slice_object(value))]
    # кавычки: «умные» и одинарные приводим к обычным
    normalized = (value.replace("“", '"').replace("”", '"').replace("„", '"')
                  .replace("’", "'").replace("‘", "'"))
    candidates.append(normalized)
    candidates.append(_balance(normalized))
    if "'" in normalized and '"' not in normalized:
        candidates.append(_balance(normalized.replace("'", '"')))
    for candidate in candidates:
        for attempt in (candidate,
                        re.sub(r",\s*([}\]])", r"\1", candidate),  # хвостовые запятые
                        re.sub(r"(\w)'(\w)", r"\1\\'\2", candidate)):  # экранирование апострофов
            try:
                parsed = json.loads(attempt)
            except (ValueError, TypeError):
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def _canonical_key(key: str, schema: dict) -> str:
    """Приводит ключ к имени из схемы: регистр, латиница/кириллица, известные алиасы."""
    raw = str(key).strip()
    properties = (schema.get("properties") or {})
    if raw in properties:
        return raw
    folded = fold(raw).replace(" ", "_").replace("-", "_")
    for name in properties:
        if folded == fold(name).replace("-", "_"):
            return name
    for name, aliases in KEY_ALIASES.items():
        if name not in properties:
            continue
        if folded in {fold(alias).replace(" ", "_") for alias in aliases}:
            return name
    return raw


def normalize_keys(payload: object, schema: dict) -> object:
    """Рекурсивно приводит ключи к схеме (регистр, алфавит, алиасы)."""
    if isinstance(payload, list):
        item_schema = (schema.get("items") or {}) if isinstance(schema, dict) else {}
        return [normalize_keys(item, item_schema) for item in payload]
    if not isinstance(payload, dict):
        return payload
    properties = (schema.get("properties") or {}) if isinstance(schema, dict) else {}
    result: dict = {}
    for key, value in payload.items():
        name = _canonical_key(str(key), schema)
        child = properties.get(name, {})
        result[name] = normalize_keys(value, child)
    return result


def _enum_match(value: str, allowed: tuple[str, ...]) -> str | None:
    """Ищет допустимое значение перечисления по синонимам и похожести написания."""
    folded = fold(value)
    if not folded:
        return None
    if folded in allowed:
        return folded
    for name, synonyms in ENUM_SYNONYMS.items():
        if folded in {fold(key) for key in synonyms} and synonyms.get(value, synonyms.get(folded)) in allowed:
            return synonyms.get(folded) or synonyms.get(value)
    best, best_score = None, 0.0
    for candidate in allowed:
        score = similarity(value, candidate)
        if score > best_score:
            best, best_score = candidate, score
    return best if best_score >= 0.72 else None


def _coerce(value: object, spec: dict, field: str) -> object:
    """Приводит значение к объявленному типу и перечислению."""
    types = spec.get("type")
    types = [types] if isinstance(types, str) else list(types or [])
    if "null" in types and value in (None, "", "null", "None", "—", "-"):
        return None
    if value is None:
        return None
    if field in ("sweetness", "color", "wine_type", "verdict") and isinstance(value, str):
        allowed = tuple(spec.get("enum") or ())
        matched = _enum_match(value, tuple(item for item in allowed if item))
        return matched if matched else value
    # Год: строку с годом приводим к числу, но значение, уже числовое, не трогаем.
    if "integer" in types and isinstance(value, str):
        digits = re.search(r"\d{4}", value)
        if digits:
            return int(digits.group(0))
    if "number" in types and isinstance(value, str):
        try:
            return float(value.replace(",", "."))
        except ValueError:
            pass
    actual = ("integer" if isinstance(value, int) and not isinstance(value, bool) else
              "number" if isinstance(value, float) else
              "boolean" if isinstance(value, bool) else
              "array" if isinstance(value, list) else
              "object" if isinstance(value, dict) else "string")
    if actual in types or (actual == "integer" and "number" in types):
        return value  # значение уже соответствует схеме: не портим тип
    if "string" in types and not isinstance(value, str):
        return str(value)
    if "array" in types and not isinstance(value, list):
        return [value]
    return value


def normalize_values(payload: object, schema: dict) -> object:
    """Рекурсивно приводит значения к типам и перечислениям схемы."""
    if not isinstance(schema, dict):
        return payload
    if isinstance(payload, list):
        return [normalize_values(item, schema.get("items") or {}) for item in payload]
    if not isinstance(payload, dict):
        return payload
    properties = schema.get("properties") or {}
    result: dict = {}
    for key, value in payload.items():
        spec = properties.get(key) or {}
        if spec.get("type") == "object" or "properties" in spec:
            result[key] = normalize_values(value, spec)
        elif spec.get("type") == "array" and isinstance(spec.get("items"), dict):
            result[key] = normalize_values(value, spec)
        else:
            result[key] = _coerce(value, spec, key)
    return result


def _builtin_errors(payload: object, schema: dict, path: str = "") -> list[str]:
    """Компактная проверка по схеме: типы, перечисления, обязательные поля, границы."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return [f"{path or 'root'}: ожидался объект"]
    for name in schema.get("required") or []:
        if name not in payload:
            errors.append(f"{path}{name}: обязательное поле отсутствует")
    properties = schema.get("properties") or {}
    for key, value in payload.items():
        spec = properties.get(key)
        if spec is None:
            continue
        types = spec.get("type")
        types = [types] if isinstance(types, str) else list(types or [])
        actual = ("null" if value is None else
                  "integer" if isinstance(value, int) and not isinstance(value, bool) else
                  "number" if isinstance(value, float) else
                  "boolean" if isinstance(value, bool) else
                  "array" if isinstance(value, list) else
                  "object" if isinstance(value, dict) else "string")
        if types and actual not in types and not (actual == "integer" and "number" in types):
            errors.append(f"{path}{key}: тип {actual}, ожидался {'/'.join(types)}")
            continue
        if spec.get("enum") is not None and value not in spec["enum"]:
            errors.append(f"{path}{key}: значение {value!r} вне перечисления")
        if isinstance(value, (int, float)) and "minimum" in spec and value < spec["minimum"]:
            errors.append(f"{path}{key}: меньше минимума {spec['minimum']}")
        if isinstance(value, (int, float)) and "maximum" in spec and value > spec["maximum"]:
            errors.append(f"{path}{key}: больше максимума {spec['maximum']}")
        if isinstance(value, str) and "maxLength" in spec and len(value) > spec["maxLength"]:
            errors.append(f"{path}{key}: длиннее {spec['maxLength']}")
        if isinstance(value, dict) and "properties" in spec:
            errors.extend(_builtin_errors(value, spec, f"{path}{key}."))
        if isinstance(value, list) and isinstance(spec.get("items"), dict):
            item_spec = spec["items"]
            for index, item in enumerate(value):
                location = f"{path}{key}[{index}]."
                if item_spec.get("type") == "object" or "properties" in item_spec:
                    errors.extend(_builtin_errors(item, item_spec, location))
                    continue
                # Элемент простого типа проверяем как значение: требовать от строки объект нельзя.
                item_types = item_spec.get("type")
                item_types = [item_types] if isinstance(item_types, str) else list(item_types or [])
                actual = ("integer" if isinstance(item, int) and not isinstance(item, bool) else
                          "number" if isinstance(item, float) else
                          "array" if isinstance(item, list) else
                          "object" if isinstance(item, dict) else "string")
                if item_types and actual not in item_types:
                    errors.append(f"{location}тип {actual}, ожидался {'/'.join(item_types)}")
                if item_spec.get("enum") is not None and item not in item_spec["enum"]:
                    errors.append(f"{location}значение вне перечисления")
                if isinstance(item, str) and "maxLength" in item_spec and len(item) > item_spec["maxLength"]:
                    errors.append(f"{location}длиннее {item_spec['maxLength']}")
    return errors


def validate(payload: object, schema: dict, max_errors: int = 5) -> tuple[bool, object, list[str]]:
    """Проверяет структуру по схеме и возвращает (ok, normalized, errors).

    Избыточные ошибки отбрасываются: для решения достаточно факта и первых причин.
    """
    normalized = normalize_values(normalize_keys(payload, schema), schema)
    errors: list[str] = []
    if _jsonschema is not None:
        validator = _jsonschema.Draft202012Validator(schema)
        for error in sorted(validator.iter_errors(normalized), key=lambda item: list(item.path)):
            location = ".".join(str(part) for part in error.path)
            errors.append(f"{location or 'root'}: {error.message}")
            if len(errors) >= max_errors:
                break
    else:
        errors = _builtin_errors(normalized, schema)[:max_errors]
    return (not errors), normalized, errors


def parse_structured(text: str, schema_name: str, *, finish_reason: str = "",
                     retry=None, max_attempts: int = 2) -> dict:
    """Оркестратор разбора: восстановление, нормализация, валидация и один повтор.

    Args:
        text: сырой ответ модели.
        schema_name: имя схемы из SCHEMAS.
        finish_reason: причина остановки модели; ``length`` означает усечение ответа.
        retry: необязательная функция ``(подсказка) -> новый текст`` для повтора с усиленным промптом.
        max_attempts: сколько раз пробовать (по умолчанию два: исходная попытка и повтор).

    Returns:
        Словарь с данными и флагами: data, valid, repaired, truncated, errors, attempts.
    """
    schema = SCHEMAS.get(schema_name)
    if schema is None:
        raise KeyError(f"неизвестная схема: {schema_name}")
    truncated = finish_reason == "length"
    if truncated:
        _STATS["truncated"] += 1
    raw = text
    attempts = 0
    repaired_any = False
    errors: list[str] = []
    while attempts < max(1, max_attempts):
        attempts += 1
        _STATS["calls"] += 1
        payload = repair_json(raw)
        if payload is None:
            errors = ["json: не удалось восстановить объект"]
            if attempts < max_attempts and retry is not None:
                _STATS["retried"] += 1
                raw = retry(_retry_hint(schema_name)) or ""
                continue
            break
        repaired = raw.strip() != json.dumps(payload, ensure_ascii=False).strip()
        ok, normalized, errors = validate(payload, schema)
        if ok:
            repaired_any = repaired_any or repaired
            _STATS["repaired" if repaired_any else "exact"] += 1
            return {"data": normalized, "valid": True, "repaired": repaired_any,
                    "truncated": truncated, "errors": [], "attempts": attempts,
                    "schema": schema_name}
        if attempts < max_attempts and retry is not None:
            _STATS["retried"] += 1
            raw = retry(_retry_hint(schema_name, errors)) or ""
            continue
        break
    _STATS["failed"] += 1
    return {"data": {}, "valid": False, "repaired": repaired_any, "truncated": truncated,
            "errors": errors, "attempts": attempts, "schema": schema_name}


def _retry_hint(schema_name: str, errors: list[str] | None = None) -> str:
    """Подсказка для повтора: явные поля, перечисления и минимальный пример."""
    if schema_name == "label_fields":
        example = '{"winery": "Массандра", "wine_name": "Мускатель белый", "vintage": 2023, ' \
                  '"sweetness": "сладкое", "color": "белое"}'
        hint = ("Верни строго JSON без пояснений. Поля: winery, wine_name, vintage (год числом), "
                "grape, sweetness, color, wine_type, region. "
                f"Допустимые сладости: {', '.join(SWEETNESS)}. Допустимые цвета: {', '.join(COLORS)}. "
                f"Допустимые типы: {', '.join(TYPES)}. Пример: {example}")
    elif schema_name == "sommelier":
        hint = ('Верни строго JSON вида {"recommendations": [{"slug": "slug-вина", "why": "причина"}], '
                '"pairing": ["блюдо"]} без пояснений и без текста вокруг.')
    else:
        hint = 'Верни строго JSON вида {"verdict": "pass|fail|uncertain", "score": 0.0, "reasons": []}.'
    if errors:
        hint += f" Предыдущий ответ не прошёл проверку: {'; '.join(errors[:3])}."
    return hint


def provider_response_format(schema_name: str, provider: str = "openai") -> dict | None:
    """Описывает схему для провайдера, где это поддерживается.

    OpenAI-совместимые получают json_schema, Ollama — саму схему для грамматики. Возвращённое
    значение лишь подсказывает провайдеру форму ответа; клиентская проверка выполняется всегда,
    потому что запасной провайдер гарантий не даёт.
    """
    schema = SCHEMAS.get(schema_name)
    if schema is None:
        return None
    if provider == "ollama":
        return dict(schema)
    return {"type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema, "strict": True}}
