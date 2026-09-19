"""Тесты текстового фьюжна C-013: overlay обогащения, нормализация цвета, гейт флага."""
from __future__ import annotations

import json

import pytest

from services.matcher.app import textfusion
from services.matcher.app.merge import rerank_candidates


CATALOG = {
    "alpha-2021": {"slug": "alpha-2021", "name": "Альфа", "winery": "Тест", "category": "Красное",
                   "color": "красное сухое вино", "year": 2021, "sweetness": "сухое"},
    "alpha-2022": {"slug": "alpha-2022", "name": "Альфа", "winery": "Тест", "category": "Красное",
                   "color": "красное сухое вино", "year": 2022},
}

OVERLAY = {
    "alpha-2022": {"slug": "alpha-2022", "enriched": {"year": 2022, "sweetness": "сухое"},
                   "category_normalized": "Красное", "ambiguous": False},
}


def test_color_normalize_canonical() -> None:
    """Свободный color нормализуется к канонической категории."""
    assert textfusion.color_normalize({"color": "красное сухое вино"}) == "Красное"
    assert textfusion.color_normalize({"category": "Белое"}) == "Белое"
    assert textfusion.color_normalize({"color": "", "category": ""}) == ""


def test_load_enriched_missing_file_is_not_an_error(tmp_path) -> None:
    """Отсутствие обогащённого каталога — не ошибка: overlay пустой."""
    assert textfusion.load_enriched_catalog(tmp_path / "nope.jsonl") == {}


def test_load_enriched_reads_overlay(tmp_path) -> None:
    """Overlay читается по slug."""
    target = tmp_path / "catalog_enriched.jsonl"
    target.write_text(json.dumps(OVERLAY["alpha-2022"], ensure_ascii=False) + "\n", encoding="utf-8")
    overlay = textfusion.load_enriched_catalog(target)
    assert set(overlay) == {"alpha-2022"}
    assert overlay["alpha-2022"]["enriched"]["sweetness"] == "сухое"


def test_enriched_record_does_not_overwrite_existing() -> None:
    """Обогащение дозаполняет только пустые поля каталога."""
    record = dict(CATALOG["alpha-2021"])
    merged = textfusion.enriched_record(record, {"alpha-2021": {"enriched": {"year": 1999, "grape": "Рислинг"}}})
    assert merged["year"] == 2021
    assert merged["grape"] == "Рислинг"


def test_fusion_off_keeps_previous_path() -> None:
    """При выключенном флаге overlay не применяется — рейтинг как раньше."""
    fields = {"winery": "Тест", "wine_name": "Альфа", "vintage": "2022"}
    off = rerank_candidates([{"slug": "alpha-2021", "score": 0.5}, {"slug": "alpha-2022", "score": 0.5}],
                            fields, CATALOG, enriched=OVERLAY)
    top = rerank_candidates([{"slug": "alpha-2021", "score": 0.5}, {"slug": "alpha-2022", "score": 0.5}],
                            fields, CATALOG)
    assert [candidate.slug for candidate in off] == [candidate.slug for candidate in top]
    assert [candidate.score for candidate in off] == [candidate.score for candidate in top]


def test_fusion_on_uses_enriched_sweetness(monkeypatch: pytest.MonkeyPatch) -> None:
    """С флагом дозаполненная сладость разводит членов near-dup группы; без флага — tie."""
    catalog = {
        "a-1": {"slug": "a-1", "name": "Альфа", "winery": "Тест", "category": "Красное", "year": 2021},
        "a-2": {"slug": "a-2", "name": "Альфа", "winery": "Тест", "category": "Красное", "year": 2022},
    }
    overlay = {"a-2": {"slug": "a-2", "enriched": {"sweetness": "сухое"}, "category_normalized": "Красное"}}
    fields = {"winery": "Тест", "wine_name": "Альфа", "wine_type": "сухое"}
    candidates = [{"slug": "a-1", "score": 0.5}, {"slug": "a-2", "score": 0.5}]

    off = rerank_candidates(candidates, fields, catalog, enriched=overlay)
    assert all(candidate.sweetness_match is False for candidate in off), "без флага overlay не применяется"

    monkeypatch.setattr(textfusion, "text_fusion_enabled", lambda: True)
    on = rerank_candidates(candidates, fields, catalog, enriched=overlay)
    assert on[0].slug == "a-2", "дозаполненная сладость должна вывести a-2 вперёд"
    assert on[0].sweetness_match is True and on[1].sweetness_match is False


def test_fusion_on_uses_grape_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    """С флагом виноград входит в текстовый скор."""
    monkeypatch.setattr(textfusion, "text_fusion_enabled", lambda: True)
    catalog = {
        "a": {"slug": "a", "name": "Вино", "winery": "Тест", "grape": "Шардоне"},
        "b": {"slug": "b", "name": "Вино", "winery": "Тест", "grape": "Мерло"},
    }
    ranked = rerank_candidates([{"slug": "a", "score": 0.0}, {"slug": "b", "score": 0.0}],
                               {"winery": "Тест", "wine_name": "Вино", "grape": "Мерло"}, catalog)
    assert ranked[0].slug == "b"
    assert ranked[0].text_score > ranked[1].text_score


def test_fusion_weights_defaults() -> None:
    """Веса фьюжна отдаются согласованным словарём."""
    weights = textfusion.fusion_weights()
    assert set(weights) == {"text_weight", "year_bonus", "sweetness_bonus"}
    assert weights["year_bonus"] > 0
