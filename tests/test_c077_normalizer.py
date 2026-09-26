# START_MODULE_CONTRACT
# PURPOSE: Тесты C-077 LLM-нормализатора: словарь каталога, промпты, разбор текста, кэш и фолбэк
# SCOPE: CATALOG, test_build_catalog_vocab_from_catalog, test_decide_uses_normalize_cache, test_winery_prompt_lists_catalog,
#        test_parse_winery_answer_matches_catalog, test_parse_fields_answer_normalizes_enum,
#        test_normalize_reading_two_llm_calls, test_normalize_reading_falls_back_without_llm,
#        test_normalizer_cache, test_normalizer_never_returns_slug
# DEPENDS: packages.common.llm.normalizer
# LINKS: M-COMMON, M-VLM
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# CATALOG - синтетический каталог с винодельнями/регионами/перечислениями
# test_build_catalog_vocab_from_catalog - словарь собирается из каталога динамически
# test_winery_prompt_lists_catalog - промпт винодельни содержит список каталога и чтение
# test_parse_winery_answer_matches_catalog - ответ винодельни сводится к каталожному написанию
# test_parse_fields_answer_normalizes_enum - параметры приводятся к перечислениям по синонимам
# test_normalize_reading_two_llm_calls - нормализация делает два запроса и склеивает результат
# test_normalize_reading_falls_back_without_llm - недоступная LLM даёт фолбэк без исключения
# test_normalizer_cache - повторный вызов берётся из кэша
# test_normalizer_never_returns_slug - нормализатор не принимает решения о товаре (нет slug)
# test_decide_uses_normalize_cache - кэш нормализации избавляет от вызова LLM
# END_MODULE_MAP
"""Тесты LLM-нормализатора C-077 (LLM подменяется, без сети)."""
from __future__ import annotations

import packages.common.llm.normalizer as normalizer
from packages.common.llm.normalizer import (build_catalog_vocab, build_winery_prompt, normalize_reading,
                                            parse_fields_answer, parse_winery_answer,
                                            reset_normalizer_cache)

CATALOG = {
    "s1": {"winery": "Кубань-Вино", "name": "Шато Тамань Полусладкое", "region": "Кубань",
           "sweetness": "полусладкое", "color": "белое", "type": "тихое"},
    "s2": {"winery": "Абрау-Дюрсо", "name": "Абрау Дюрсо Резерв", "region": "Кубань",
           "sweetness": "брют", "color": "белое", "type": "игристое"},
}

__all__ = ["CATALOG", "test_build_catalog_vocab_from_catalog",
           "test_normalize_reading_falls_back_without_llm", "test_normalize_reading_two_llm_calls",
           "test_decide_uses_normalize_cache", "test_normalizer_cache", "test_normalizer_never_returns_slug",
           "test_parse_fields_answer_normalizes_enum", "test_parse_winery_answer_matches_catalog",
           "test_winery_prompt_lists_catalog"]


def test_build_catalog_vocab_from_catalog() -> None:
    """Словарь собирается из каталога: винодельни, регионы и перечисления без дублей."""
    vocab = build_catalog_vocab(CATALOG)
    assert "Кубань-Вино" in vocab["wineries"] and "Абрау-Дюрсо" in vocab["wineries"]
    assert vocab["regions"] == ["Кубань"]
    assert set(vocab["sweetness"]) == {"полусладкое", "брют"}


def test_winery_prompt_lists_catalog() -> None:
    """Промпт первого запроса содержит список виноделен каталога и прочитанное значение."""
    prompt = build_winery_prompt({"winery": "Château Tamagne"}, build_catalog_vocab(CATALOG))
    assert "Кубань-Вино" in prompt and "Château Tamagne" in prompt
    assert "winery:" in prompt and "brand:" in prompt


def test_parse_winery_answer_matches_catalog() -> None:
    """Ответ винодельни сводится к каталожному написанию, бренд сохраняется."""
    out = parse_winery_answer("winery: Кубань Вино\nbrand: Château Tamagne",
                              build_catalog_vocab(CATALOG))
    assert out["winery"] == "Кубань-Вино"          # частичное совпадение сведено к каталожному
    assert out["brand"] == "Château Tamagne"


def test_parse_fields_answer_normalizes_enum() -> None:
    """Параметры приводятся к перечислениям по синонимам (semi-sweet → полусладкое)."""
    out = parse_fields_answer("wine_name: Шато Тамань Полусладкое\nsweetness: semi-sweet\n"
                              "color: white\nwine_type: still\nregion: Кубань",
                              build_catalog_vocab(CATALOG))
    assert out["wine_name"] == "Шато Тамань Полусладкое"
    assert out["sweetness"] == "полусладкое" and out["color"] == "белое" and out["wine_type"] == "тихое"


def test_normalize_reading_two_llm_calls(monkeypatch) -> None:
    """Нормализация делает два запроса и склеивает винодельню и параметры."""
    reset_normalizer_cache()
    calls: list[str] = []
    answers = ["winery: Кубань-Вино\nbrand: Château Tamagne",
               "wine_name: Шато Тамань Полусладкое\nsweetness: semi-sweet"]

    def fake_native(prompt, model, **kwargs):  # noqa: ANN001
        calls.append(prompt)
        return answers[len(calls) - 1]

    monkeypatch.setattr(normalizer, "ollama_native_chat", fake_native)
    out = normalize_reading({"winery": "Château Tamagne", "raw_text": "SEMI-SWEET"}, CATALOG,
                            model="gemma4:e4b-mlx")
    assert len(calls) == 2
    assert out["winery"] == "Кубань-Вино" and out["wine_name"] == "Шато Тамань Полусладкое"
    assert out["sweetness"] == "полусладкое" and out["source"] == "llm"


def test_normalize_reading_falls_back_without_llm(monkeypatch) -> None:
    """Недоступная LLM даёт фолбэк (транслит) без исключения и с source=fallback."""
    reset_normalizer_cache()
    monkeypatch.setattr(normalizer, "ollama_native_chat", lambda *a, **k: None)
    out = normalize_reading({"winery": "МACCAНДРА", "wine_name": "Мускатель"}, CATALOG,
                            model="gemma4:e4b-mlx")
    assert out["source"] == "fallback"
    assert out["winery"] == "МАССАНДРА"  # латинские двойники сведены к кириллице


def test_normalizer_cache(monkeypatch) -> None:
    """Повторное чтение берётся из кэша: LLM не вызывается второй раз."""
    reset_normalizer_cache()
    calls: list[int] = [0]

    def fake_native(prompt, model, **kwargs):  # noqa: ANN001
        calls[0] += 1
        return "winery: Кубань-Вино\nbrand: Château Tamagne" if calls[0] == 1 else "wine_name: Вино"

    monkeypatch.setattr(normalizer, "ollama_native_chat", fake_native)
    obs = {"winery": "Château Tamagne", "raw_text": "SEMI-SWEET"}
    normalize_reading(obs, CATALOG, model="gemma4:e4b-mlx")
    normalize_reading(obs, CATALOG, model="gemma4:e4b-mlx")
    assert calls[0] == 2  # только первый вызов (два запроса), второй — из кэша


def test_normalizer_never_returns_slug() -> None:
    """Нормализатор не принимает решений о товаре: slug в результате отсутствует."""
    reset_normalizer_cache()
    out = normalize_reading({"winery": "X", "wine_name": "Y"}, CATALOG, model="gemma4:e4b-mlx",
                            use_cache=False)
    assert "slug" not in out


def test_decide_uses_normalize_cache(monkeypatch) -> None:
    """При наличии кэша нормализации LLM не вызывается: берётся готовая стандартизация (C-077)."""
    from packages.common.text_rank import text_index
    from scripts.eval.llm_match_eval import decide
    calls: list[int] = [0]

    def fake_match(observation, candidates, **kwargs):  # noqa: ANN001
        return {"slug": None, "p_correct": None, "in_catalog": False, "reason": "", "source": "llm"}

    def fake_normalize(*args, **kwargs):  # noqa: ANN002, ANN003
        calls[0] += 1
        return {"winery": "Кубань-Вино", "brand": "Шато Тамань", "wine_name": "Шато Тамань",
                "source": "llm"}

    monkeypatch.setattr("scripts.eval.llm_match_eval.match_candidates", fake_match)
    monkeypatch.setattr("scripts.eval.llm_match_eval.normalize_reading", fake_normalize)
    raw = "winery: CHATEAU TAMAGNE\nsweetness: SEMI-SWEET"
    cache = {raw: {"winery": "Кубань-Вино", "brand": "Шато Тамань", "wine_name": "Шато Тамань",
                   "sweetness": "полусладкое", "source": "llm"}}
    decide(raw, CATALOG, text_index(CATALOG), CATALOG, normalize=True, normalize_cache=cache)
    assert calls[0] == 0  # кэш сработал, LLM-нормализатор не вызывался
