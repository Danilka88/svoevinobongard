# START_MODULE_CONTRACT
# PURPOSE: Тесты C-074 LLM-матчера: список кандидатов, промпт, выбор slug, отказ и деградация
# SCOPE: OBSERVATION, CANDIDATES, StubRouter, test_build_candidates_text_limits_and_formats,
#        test_build_matcher_prompt_contains_observation_and_candidates, test_parse_match_accepts_listed_slug,
#        test_parse_match_rejects_hallucinated_slug, test_parse_match_invalid_json_degrades,
#        test_match_candidates_uses_llm_choice, test_match_candidates_degrades_on_llm_failure,
#        test_match_candidates_empty_candidates
# DEPENDS: packages.common.llm.matcher
# LINKS: M-COMMON, M-VLM, M-GATEWAY
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# OBSERVATION - наблюдение этикетки для матчера
# CANDIDATES - синтетические кандидаты каталога
# StubRouter - подставной роутер: возвращает заранее заданный текст или падает
# test_build_candidates_text_limits_and_formats - список кандидатов нумерован и ограничен
# test_build_matcher_prompt_contains_observation_and_candidates - промпт несёт наблюдение и кандидатов
# test_parse_match_accepts_listed_slug - slug из списка принимается с вероятностью
# test_parse_match_rejects_hallucinated_slug - выдуманный slug сводится к отказу
# test_parse_match_invalid_json_degrades - невалидный ответ даёт degraded и slug None
# test_match_candidates_uses_llm_choice - матчер возвращает выбор LLM
# test_match_candidates_degrades_on_llm_failure - сбой LLM не роняет пайплайн
# test_match_candidates_empty_candidates - без кандидатов матчер сразу отказывается
# test_match_candidates_native_path_uses_ollama - локальное имя модели идёт нативным путём Ollama
# END_MODULE_MAP
"""Тесты LLM-матчера C-074 (без сети: LLM подменяется)."""
from __future__ import annotations

from types import SimpleNamespace

from packages.common.llm.matcher import (build_candidates_text, build_matcher_prompt, match_candidates,
                                         parse_match)

OBSERVATION = {"winery": "Скалистый берег", "wine_name": "Шёпот цветов", "vintage": "2024",
               "raw_text": "SHOPOT CVETOV 0.75 L"}
CANDIDATES = [
    {"slug": "skalistyy-bereg-shepot-czvetov", "name": "Шёпот цветов", "winery": "Скалистый берег",
     "year": 2024, "grape": "Рислинг", "sweetness": "брют"},
    {"slug": "massandra-muskatel", "name": "Мускатель белый", "winery": "Массандра", "year": 2023,
     "grape": "Мускатель", "sweetness": "сладкое"},
]

__all__ = ["CANDIDATES", "OBSERVATION", "StubRouter",
           "test_build_candidates_text_limits_and_formats",
           "test_build_matcher_prompt_contains_observation_and_candidates",
           "test_match_candidates_degrades_on_llm_failure", "test_match_candidates_empty_candidates",
           "test_match_candidates_native_path_uses_ollama", "test_match_candidates_uses_llm_choice",
           "test_parse_match_accepts_listed_slug", "test_parse_match_invalid_json_degrades",
           "test_parse_match_rejects_hallucinated_slug"]


class StubRouter:
    """Подставной роутер: отдаёт заданный текст или поднимает ошибку."""

    def __init__(self, text: str = "", error: Exception | None = None) -> None:
        self.text = text
        self.error = error

    def chat(self, messages, **kwargs):  # noqa: ANN001, ANN003 - совместимость с интерфейсом роутера
        if self.error is not None:
            raise self.error
        return SimpleNamespace(text=self.text, model="stub", provider="stub")


def test_build_candidates_text_limits_and_formats() -> None:
    """Список кандидатов нумерован, несёт slug и поля и уважает лимит."""
    text = build_candidates_text(CANDIDATES, limit=1)
    assert "1. skalistyy-bereg-shepot-czvetov" in text
    assert "Шёпот цветов" in text and "винодельня: Скалистый берег" in text
    assert "massandra-muskatel" not in text


def test_build_matcher_prompt_contains_observation_and_candidates() -> None:
    """Промпт матчера содержит и наблюдение, и список кандидатов."""
    prompt = build_matcher_prompt(OBSERVATION, CANDIDATES)
    assert "Прочитано с этикетки" in prompt and "Шёпот цветов" in prompt
    assert "Кандидаты каталога" in prompt and "massandra-muskatel" in prompt
    assert "null" in prompt


def test_parse_match_accepts_listed_slug() -> None:
    """Slug из списка принимается вместе с вероятностью."""
    outcome = parse_match('{"slug": "skalistyy-bereg-shepot-czvetov", "p_correct": 0.82, '
                          '"in_catalog": true, "reason": "совпали винодельня и название"}', CANDIDATES)
    assert outcome["slug"] == "skalistyy-bereg-shepot-czvetov"
    assert outcome["p_correct"] == 0.82
    assert outcome["in_catalog"] is True
    assert outcome["source"] == "llm"


def test_parse_match_rejects_hallucinated_slug() -> None:
    """Slug, которого нет в списке, сводится к честному отказу."""
    outcome = parse_match('{"slug": "vinotdelka-vymyshlennoe-vino", "p_correct": 0.9, '
                          '"in_catalog": true}', CANDIDATES)
    assert outcome["slug"] is None
    assert outcome["in_catalog"] is False


def test_parse_match_invalid_json_degrades() -> None:
    """Невалидный ответ даёт source=degraded и отказ, без исключения."""
    outcome = parse_match("Извините, не могу сопоставить.", CANDIDATES)
    assert outcome["slug"] is None
    assert outcome["source"] == "degraded"


def test_match_candidates_uses_llm_choice() -> None:
    """Матчер возвращает выбор LLM и связывает его с моделью."""
    router = StubRouter('{"slug": "massandra-muskatel", "p_correct": 0.6, "in_catalog": true}')
    outcome = match_candidates(OBSERVATION, CANDIDATES, router=router)
    assert outcome["slug"] == "massandra-muskatel" and outcome["source"] == "llm"
    assert outcome["model"] == "stub"


def test_match_candidates_degrades_on_llm_failure() -> None:
    """Недоступная LLM не роняет пайплайн: slug None, source degraded."""
    router = StubRouter(error=RuntimeError("connrefused"))
    outcome = match_candidates(OBSERVATION, CANDIDATES, router=router)
    assert outcome["slug"] is None and outcome["source"] == "degraded"
    assert "RuntimeError" in outcome["reason"]


def test_match_candidates_empty_candidates() -> None:
    """Без кандидатов матчер сразу отказывается, не вызывая LLM."""
    outcome = match_candidates(OBSERVATION, [], router=StubRouter(error=AssertionError("не должен")) )
    assert outcome["slug"] is None and outcome["source"] == "degraded"


def test_match_candidates_native_path_uses_ollama(monkeypatch) -> None:
    """Локальное имя модели (без слэша) идёт нативным путём Ollama; ответ связывается с моделью."""
    captured: dict = {}

    def fake_native(prompt, model, *, max_tokens, timeout_s):  # noqa: ANN001
        captured.update(prompt=prompt, model=model, max_tokens=max_tokens, timeout_s=timeout_s)
        return '{"slug": "skalistyy-bereg-shepot-czvetov", "p_correct": 0.7, "in_catalog": true}'

    monkeypatch.setattr("packages.common.llm.matcher._native_ollama_chat", fake_native)
    outcome = match_candidates(OBSERVATION, CANDIDATES, model="gemma4:12b-mlx")
    assert captured["model"] == "gemma4:12b-mlx" and "Кандидаты каталога" in captured["prompt"]
    assert outcome["slug"] == "skalistyy-bereg-shepot-czvetov"
    assert outcome["provider"] == "ollama-native" and outcome["source"] == "llm"
