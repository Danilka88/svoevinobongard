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

    off = rerank_candidates(candidates, fields, catalog, enriched=overlay, fusion="off")
    assert all(candidate.sweetness_match is False for candidate in off), "в режиме off overlay не применяется"

    monkeypatch.setattr(textfusion, "fusion_mode", lambda: "always")
    on = rerank_candidates(candidates, fields, catalog, enriched=overlay)
    assert on[0].slug == "a-2", "дозаполненная сладость должна вывести a-2 вперёд"
    assert on[0].sweetness_match is True and on[1].sweetness_match is False


def test_fusion_on_uses_grape_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    """С флагом виноград входит в текстовый скор."""
    monkeypatch.setattr(textfusion, "fusion_mode", lambda: "always")
    catalog = {
        "a": {"slug": "a", "name": "Вино", "winery": "Тест", "grape": "Шардоне"},
        "b": {"slug": "b", "name": "Вино", "winery": "Тест", "grape": "Мерло"},
    }
    ranked = rerank_candidates([{"slug": "a", "score": 0.0}, {"slug": "b", "score": 0.0}],
                               {"winery": "Тест", "wine_name": "Вино", "grape": "Мерло"}, catalog,
                               fusion="always")
    assert ranked[0].slug == "b"
    assert ranked[0].text_score > ranked[1].text_score


def test_fusion_weights_defaults() -> None:
    """Веса фьюжна отдаются согласованным словарём."""
    weights = textfusion.fusion_weights()
    assert set(weights) == {"text_weight", "year_bonus", "sweetness_bonus"}
    assert weights["year_bonus"] > 0


CATALOG_GROUPS = {
    "w-2021": {"slug": "w-2021", "name": "Вино", "winery": "Тест", "category": "Красное", "year": 2021},
    "w-2022": {"slug": "w-2022", "name": "Вино", "winery": "Тест", "category": "Красное", "year": 2022},
    "other": {"slug": "other", "name": "Другое", "winery": "Другой", "category": "Белое"},
}
OVERLAY_GROUP = {"w-2021": {"slug": "w-2021", "enriched": {"sweetness": "сухое"}},
                 "w-2022": {"slug": "w-2022", "enriched": {"sweetness": "сухое"}}}


def test_conditional_triggers_on_near_dup_pair() -> None:
    """Триггер срабатывает, когда top-1 и top-2 одной группы и скоры близки."""
    candidates = [{"slug": "w-2021", "score": 0.70}, {"slug": "w-2022", "score": 0.68}]
    assert textfusion.should_fuse(candidates, CATALOG_GROUPS) is True


def test_conditional_skips_when_top2_different_groups() -> None:
    """Не-near-dup запросы триггер не включают: выдачу трогать нельзя."""
    candidates = [{"slug": "w-2021", "score": 0.70}, {"slug": "other", "score": 0.68}]
    assert textfusion.should_fuse(candidates, CATALOG_GROUPS) is False


def test_conditional_skips_on_large_gap() -> None:
    """Если визуальный лидер уверенно оторвался, текст не нужен."""
    candidates = [{"slug": "w-2021", "score": 0.90}, {"slug": "w-2022", "score": 0.20}]
    assert textfusion.should_fuse(candidates, CATALOG_GROUPS) is False


def test_conditional_keeps_order_for_non_near_dup() -> None:
    """В режиме conditional на не-near-dup запросе порядок совпадает с off."""
    fields = {"winery": "Другой", "wine_name": "Другое", "wine_type": "сухое"}
    candidates = [{"slug": "other", "score": 0.90}, {"slug": "w-2021", "score": 0.30}]
    off = rerank_candidates(candidates, fields, CATALOG_GROUPS, enriched=OVERLAY_GROUP, fusion="off")
    conditional = rerank_candidates(candidates, fields, CATALOG_GROUPS, enriched=OVERLAY_GROUP,
                                    fusion="conditional")
    assert [c.slug for c in off] == [c.slug for c in conditional]
    assert [c.score for c in off] == [c.score for c in conditional]


def test_conditional_fuses_on_near_dup_pair() -> None:
    """На near-dup паре conditional даёт то же, что безусловный фьюжн."""
    fields = {"winery": "Тест", "wine_name": "Вино", "wine_type": "сухое"}
    candidates = [{"slug": "w-2021", "score": 0.70}, {"slug": "w-2022", "score": 0.68}]
    always = rerank_candidates(candidates, fields, CATALOG_GROUPS, enriched=OVERLAY_GROUP, fusion="always")
    conditional = rerank_candidates(candidates, fields, CATALOG_GROUPS, enriched=OVERLAY_GROUP,
                                    fusion="conditional")
    assert [c.slug for c in always] == [c.slug for c in conditional]
    assert [c.sweetness_match for c in always] == [c.sweetness_match for c in conditional]


def test_fusion_mode_parsing() -> None:
    """Режим разбирается из конфига, неизвестные значения сводятся к off."""
    settings = textfusion.settings
    before_enabled, before_mode = settings.text_fusion_enabled, settings.text_fusion_mode
    try:
        object.__setattr__(settings, "text_fusion_enabled", False)
        object.__setattr__(settings, "text_fusion_mode", "conditional")
        assert textfusion.fusion_mode() == "conditional"
        object.__setattr__(settings, "text_fusion_mode", "мусор")
        assert textfusion.fusion_mode() == "off"
        object.__setattr__(settings, "text_fusion_mode", "off")
        object.__setattr__(settings, "text_fusion_enabled", True)
        assert textfusion.fusion_mode() == "always", "TEXT_FUSION_ENABLED=1 равносилен режиму always"
    finally:
        object.__setattr__(settings, "text_fusion_enabled", before_enabled)
        object.__setattr__(settings, "text_fusion_mode", before_mode)


def test_default_fusion_mode_is_conditional(monkeypatch: pytest.MonkeyPatch) -> None:
    """Дефолт режима фьюжна — conditional: он не трогает запросы вне near-dup."""
    monkeypatch.delenv("TEXT_FUSION_MODE", raising=False)
    monkeypatch.delenv("TEXT_FUSION_ENABLED", raising=False)
    from packages.common import config as cfg_module

    fresh = type(cfg_module.settings)()
    assert fresh.text_fusion_mode == "conditional"
    assert fresh.text_fusion_enabled is False


def test_conditional_default_does_not_touch_non_near_dup(monkeypatch: pytest.MonkeyPatch) -> None:
    """С дефолтным conditional порядок вне near-dup совпадает с off."""
    monkeypatch.delenv("TEXT_FUSION_MODE", raising=False)
    monkeypatch.delenv("TEXT_FUSION_ENABLED", raising=False)
    from packages.common import config as cfg_module

    settings = textfusion.settings
    before = (settings.text_fusion_enabled, settings.text_fusion_mode)
    try:
        object.__setattr__(settings, "text_fusion_enabled", False)
        object.__setattr__(settings, "text_fusion_mode", type(cfg_module.settings)().text_fusion_mode)
        fields = {"winery": "Другой", "wine_name": "Другое", "wine_type": "сухое"}
        candidates = [{"slug": "other", "score": 0.9}, {"slug": "w-2021", "score": 0.3}]
        default = rerank_candidates(candidates, fields, CATALOG_GROUPS, enriched=OVERLAY_GROUP)
        off = rerank_candidates(candidates, fields, CATALOG_GROUPS, enriched=OVERLAY_GROUP, fusion="off")
        assert [c.slug for c in default] == [c.slug for c in off]
        assert [c.score for c in default] == [c.score for c in off]
    finally:
        object.__setattr__(settings, "text_fusion_enabled", before[0])
        object.__setattr__(settings, "text_fusion_mode", before[1])
