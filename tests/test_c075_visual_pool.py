# START_MODULE_CONTRACT
# PURPOSE: Тесты C-075 визуального кандидат-пула: объединение текстовых и визуальных кандидатов
# SCOPE: test_union_text_and_visual, test_no_duplicates, test_limit_respected,
#        test_visual_only_when_no_text, test_empty_pool, test_decide_uses_visual_candidates
# DEPENDS: packages.common.observer, scripts.eval.llm_match_eval
# LINKS: M-EVAL, M-COMMON
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# CATALOG - синтетический каталог для проверки пула кандидатов
# test_union_text_and_visual - текстовые находки идут первыми, визуальные дополняют
# test_no_duplicates - дубликаты между источниками убираются
# test_limit_respected - лимит пула соблюдается
# test_visual_quota_reserves_slots - визуальным кандидатам резервируются слоты
# test_visual_only_when_no_text - без текстовых находок пул состоит из визуальных
# test_empty_pool - пустые источники дают пустой пул
# test_decide_uses_visual_candidates - матчер получает объединённый пул и выбирает из него
# END_MODULE_MAP
"""Тесты визуального кандидат-пула C-075 (без сети: матчер подменяется)."""
from __future__ import annotations

from scripts.eval.llm_match_eval import build_candidate_pool, decide

CATALOG = {
    "s1": {"slug": "s1", "name": "Вино 1", "winery": "В1", "year": 2024, "grape": "Рислинг",
           "sweetness": "брют"},
    "s2": {"slug": "s2", "name": "Вино 2", "winery": "В2", "year": 2023, "grape": "Мускат",
           "sweetness": "сладкое"},
    "s3": {"slug": "s3", "name": "Вино 3", "winery": "В3", "year": None, "grape": None,
           "sweetness": None},
}

__all__ = ["CATALOG", "test_decide_uses_visual_candidates", "test_empty_pool",
           "test_limit_respected", "test_no_duplicates", "test_union_text_and_visual",
           "test_visual_only_when_no_text", "test_visual_quota_reserves_slots"]


def test_union_text_and_visual() -> None:
    """Текстовые находки стоят первыми, визуальные дополняют пул."""
    pool = build_candidate_pool([("s1", 0.9), ("s2", 0.5)], ["s3"], limit=10)
    assert pool == ["s1", "s2", "s3"]


def test_no_duplicates() -> None:
    """Совпадение текстового и визуального кандидата не дублируется."""
    pool = build_candidate_pool([("s1", 0.9)], ["s1", "s2"], limit=10)
    assert pool == ["s1", "s2"]


def test_limit_respected() -> None:
    """Лимит пула соблюдается с приоритетом текстовых находок (квота)."""
    pool = build_candidate_pool([("s1", 0.9), ("s2", 0.8), ("s3", 0.7)], ["s4", "s5"], limit=2,
                                text_quota=1)
    assert pool[0] == "s1" and len(pool) == 2


def test_visual_quota_reserves_slots() -> None:
    """Визуальным кандидатам резервируются слоты: длинный текст не вытесняет визуального кандидата."""
    text = [("t1", 0.9), ("t2", 0.8), ("t3", 0.7), ("t4", 0.6)]
    pool = build_candidate_pool(text, ["v1"], limit=4, text_quota=2)
    assert "v1" in pool and pool[:2] == ["t1", "t2"]


def test_visual_only_when_no_text() -> None:
    """Без текстовых находок пул состоит из визуальных кандидатов."""
    assert build_candidate_pool([], ["s3", "s2"], limit=10) == ["s3", "s2"]


def test_empty_pool() -> None:
    """Пустые источники дают пустой пул."""
    assert build_candidate_pool([], None, limit=10) == []
    assert build_candidate_pool([], [], limit=10) == []


def test_decide_uses_visual_candidates(monkeypatch) -> None:
    """Матчер получает объединённый пул, включая визуального кандидата, которого нет в тексте."""
    seen: dict = {}

    def fake_match(observation, candidates, **kwargs):  # noqa: ANN001
        seen["slugs"] = [item["slug"] for item in candidates]
        return {"slug": "s3", "p_correct": 0.8, "in_catalog": True, "reason": "визуальный кандидат",
                "source": "llm"}

    monkeypatch.setattr("scripts.eval.llm_match_eval.match_candidates", fake_match)
    index = {"s1": {"name": "вино"}}
    decision = decide("winery: В1\nwine_name: Вино", CATALOG, index, CATALOG,
                      visual_slugs=["s3"])
    assert "s3" in seen["slugs"]
    assert decision["top1"] == "s3" and decision["source"] == "llm"
    assert "s3" in decision["candidates"]
