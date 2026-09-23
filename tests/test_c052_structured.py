"""Тесты C-052: разбор структурированных ответов моделей на реальных архетипах «грязного» вывода.

Каждый архетип прогоняется через настоящий `parse_structured`, а не через вспомогательные вызовы:
проверяется и результат, и флаги (repaired, valid, truncated), и устойчивость к отказу модели.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from packages.common.llm.structured import (  # noqa: E402
    SCHEMAS,
    normalize_keys,
    parse_structured,
    provider_response_format,
    repair_json,
    reset_validation_stats,
    validate,
    validation_stats,
)

CLEAN = '{"winery": "Массандра", "wine_name": "Мускатель белый", "vintage": 2023, "sweetness": "сладкое"}'


@pytest.fixture(autouse=True)
def _clean_stats():
    reset_validation_stats()
    yield


# --- архетипы ответа -----------------------------------------------------------

def test_exact_json() -> None:
    result = parse_structured(CLEAN, "label_fields")
    assert result["valid"] is True
    assert result["repaired"] is False
    assert result["data"]["winery"] == "Массандра"
    assert validation_stats()["exact"] == 1


def test_markdown_fence() -> None:
    text = f"```json\n{CLEAN}\n```"
    result = parse_structured(text, "label_fields")
    assert result["valid"] is True and result["data"]["vintage"] == 2023


def test_text_around_json() -> None:
    text = f"Вот результат анализа этикетки:\n{CLEAN}\nНадеюсь, помог!"
    result = parse_structured(text, "label_fields")
    assert result["valid"] is True and result["data"]["wine_name"] == "Мускатель белый"


def test_single_quotes() -> None:
    text = "{'winery': 'Массандра', 'wine_name': 'Мускатель белый', 'vintage': 2023}"
    result = parse_structured(text, "label_fields")
    assert result["valid"] is True and result["data"]["winery"] == "Массандра"


def test_smart_quotes() -> None:
    text = '{\u201cwinery\u201d: \u201cМассандра\u201d, \u201cwine_name\u201d: \u201cМускатель\u201d}'
    result = parse_structured(text, "label_fields")
    assert result["valid"] is True and result["data"]["winery"] == "Массандра"


def test_trailing_comma() -> None:
    text = '{"winery": "Массандра", "wine_name": "Мускатель", "vintage": 2023,}'
    result = parse_structured(text, "label_fields")
    assert result["valid"] is True and result["data"]["vintage"] == 2023


def test_uppercase_and_cyrillic_keys() -> None:
    text = '{"WINERY": "Массандра", "Wine_Name": "Мускатель", "Винодельня": "Массандра"}'
    result = parse_structured(text, "label_fields")
    assert result["valid"] is True
    assert result["data"]["winery"] == "Массандра"
    assert result["data"]["wine_name"] == "Мускатель"


def test_truncated_response_is_flagged() -> None:
    """Обрыв ответа: скобки достраиваются, факт усечения фиксируется отдельно."""
    text = '{"winery": "Массандра", "wine_name": "Мускатель бе'
    result = parse_structured(text, "label_fields", finish_reason="length")
    assert result["truncated"] is True
    assert validation_stats()["truncated"] == 1


def test_empty_and_none_are_safe() -> None:
    for text in ("", "   ", "модель отказалась отвечать", None):
        result = parse_structured(text or "", "label_fields")
        assert result["valid"] is False
        assert result["data"] == {}
        assert result["errors"]


def test_wrong_types_are_reported() -> None:
    text = '{"winery": 42, "recommendations": "не массив"}'
    result = parse_structured(text, "sommelier")
    assert result["valid"] is False
    assert any("recommendations" in error for error in result["errors"])


# --- нормализация значений ------------------------------------------------------

@pytest.mark.parametrize(("raw", "expected"), [
    ("Брют", "брют"), ("BRUT", "брют"), ("dry", "сухое"), ("DRY", "сухое"),
    ("полусухое", "полусухое"), ("semi-sweet", "полусладкое"),
])
def test_sweetness_synonyms(raw: str, expected: str) -> None:
    result = parse_structured(f'{{"sweetness": "{raw}"}}', "label_fields")
    assert result["valid"] is True, result["errors"]
    assert result["data"]["sweetness"] == expected


@pytest.mark.parametrize(("raw", "expected"), [
    ("красное", "красное"), ("RED", "красное"), ("белое сухое", "белое"), ("white", "белое"),
])
def test_color_synonyms(raw: str, expected: str) -> None:
    result = parse_structured(f'{{"color": "{raw}"}}', "label_fields")
    assert result["valid"] is True, result["errors"]
    assert result["data"]["color"] == expected


def test_vintage_is_coerced_to_year() -> None:
    result = parse_structured('{"vintage": "урожай 2021 года"}', "label_fields")
    assert result["data"]["vintage"] == 2021


def test_unknown_enum_value_is_rejected() -> None:
    """Неизвестное значение перечисления не выдумывается: оно фиксируется как ошибка."""
    # Похожее написание нормализуется намеренно, поэтому берём значение из другой области.
    result = parse_structured('{"sweetness": "газированное игристое"}', "label_fields")
    assert result["valid"] is False
    assert any("sweetness" in error for error in result["errors"])


# --- повтор и безопасный дефолт -------------------------------------------------

def test_retry_with_strengthened_prompt() -> None:
    """При невалидном ответе делается один повтор, и подсказка содержит поля и пример."""
    seen: list[str] = []

    def retry(hint: str) -> str:
        seen.append(hint)
        return CLEAN

    result = parse_structured("не json вовсе", "label_fields", retry=retry)
    assert result["valid"] is True
    assert result["attempts"] == 2
    assert validation_stats()["retried"] == 1
    assert seen and "winery" in seen[0] and "Пример" in seen[0]


def test_failed_after_retry_returns_safe_default() -> None:
    """После повтора возвращается безопасный дефолт, исключение не поднимается."""
    result = parse_structured("мусор", "sommelier", retry=lambda hint: "тоже мусор")
    assert result["valid"] is False
    assert result["data"] == {}
    assert result["attempts"] == 2
    assert validation_stats()["failed"] == 1


def test_model_refusal_is_not_an_exception() -> None:
    result = parse_structured("Извините, я не могу проанализировать это изображение.", "label_fields")
    assert result["valid"] is False and "json" in result["errors"][0]


# --- вспомогательные части ------------------------------------------------------

def test_schemas_are_closed_for_enum_fields() -> None:
    label = SCHEMAS["label_fields"]["properties"]
    assert "брют" in label["sweetness"]["enum"]
    assert "красное" in label["color"]["enum"]
    assert "игристое" in label["wine_type"]["enum"]
    assert SCHEMAS["judge"]["required"] == ["verdict"]


def test_repair_json_direct() -> None:
    assert repair_json("```json\n{\"a\": 1,}\n```") == {"a": 1}
    assert repair_json("текст без объекта") is None
    assert repair_json("") is None


def test_validate_returns_errors_not_exceptions() -> None:
    ok, normalized, errors = validate({"verdict": "may be"}, SCHEMAS["judge"])
    assert ok is False and normalized["verdict"] == "may be" and errors


def test_normalize_keys_is_recursive() -> None:
    payload = {"Recommendations": [{"Slug": "wine-1", "WHY": "подходит"}]}
    result = normalize_keys(payload, SCHEMAS["sommelier"])
    assert result["recommendations"][0]["slug"] == "wine-1"
    assert result["recommendations"][0]["why"] == "подходит"


def test_provider_response_format() -> None:
    """Схема передаётся провайдеру, где поддерживается, но валидация остаётся клиентской."""
    openai = provider_response_format("label_fields", "openai")
    assert openai["type"] == "json_schema" and openai["json_schema"]["strict"] is True
    ollama = provider_response_format("label_fields", "ollama")
    assert "properties" in ollama
    assert provider_response_format("нет-такой-схемы") is None


def test_stats_shape() -> None:
    parse_structured(CLEAN, "label_fields")
    stats = validation_stats()
    assert set(stats) == {"exact", "repaired", "retried", "failed", "truncated", "calls"}
    assert stats["calls"] == 1
