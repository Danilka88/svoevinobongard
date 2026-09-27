# START_MODULE_CONTRACT
# PURPOSE: Тесты C-092 генератора демо-стенда: сценарии, поля примеров, копирование фото
# SCOPE: test_scenario_of, test_clean_fields_labels, test_build_samples_shape, test_copy_photos
# DEPENDS: scripts.eval.build_demo_stand
# LINKS: M-EVAL, M-WEB
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# MATCH - синтетический отчёт LLM-пути для проверки сборки примеров
# GOLD - синтетический эталон с одним matched-фото
# test_scenario_of - сценарий по статусу эталона
# test_clean_fields_labels - поля наблюдения/нормализации в вид «подпись → значение» без пустых
# test_build_samples_shape - пример стенда несёт фото, чтение, кандидатов, итог, путь, latency
# test_copy_photos - копирование фото в public и карта имён
# END_MODULE_MAP
"""Тесты генератора демо-стенда C-092."""
from __future__ import annotations

from scripts.eval.build_demo_stand import _clean_fields, build_samples, copy_photos, scenario_of

__all__ = ["GOLD", "MATCH", "test_build_samples_shape", "test_clean_fields_labels",
           "test_copy_photos", "test_scenario_of"]

MATCH = {"results": [{
    "file": "a.jpg", "observation": "winery: ABRAU-DURSO\nwine_name: RESERVE",
    "fields": {"winery": "ABRAU-DURSO", "wine_name": "RESERVE", "grape": ""},
    "normalized": {"winery": "Абрау-Дюрсо", "source": "llm"},
    "visual_candidates": ["s1", "s2"], "text_candidates": ["s3"], "top1": "s1", "not_found": False,
    "path": "llm_llm", "latency_ms": 1234.0, "reader_ms": 500.0,
}], "metrics": {"accuracy_at_1": 0.4545}}
GOLD = {"files": {"a.jpg": {"status": "matched", "expected_slug": "s1", "human_label": "Abrau"}}}


def test_scenario_of() -> None:
    """Сценарий определяется по статусу эталона."""
    assert scenario_of("matched") == "matched"
    assert scenario_of("ambiguous") == "ambiguous"
    assert scenario_of("not_found") == "not_found"
    assert scenario_of("") == "not_found"


def test_clean_fields_labels() -> None:
    """Поля превращаются в список «подпись → значение», пустые пропускаются."""
    rows = _clean_fields({"winery": "Абрау-Дюрсо", "grape": "", "sweetness": None})
    labels = {row["label"] for row in rows}
    assert "Винодельня" in labels and "Сорт" not in labels and "Сладость" not in labels


def test_build_samples_shape() -> None:
    """Пример стенда несёт фото, чтение, нормализацию, кандидатов, итог, путь и время."""
    samples = build_samples(MATCH, GOLD, {"a.jpg": "img/01.jpg"})
    assert len(samples) == 1
    sample = samples[0]
    assert sample["photo"] == "img/01.jpg" and sample["scenario"] == "matched"
    assert sample["top1"] == "s1" and sample["not_found"] is False
    assert sample["candidates"] and sample["path"] == "llm_llm" and sample["latency_ms"] == 1234.0
    assert any(row["label"] == "Винодельня" for row in sample["reading_fields"])
    assert any(row["label"] == "Винодельня" for row in sample["normalized"])


def test_copy_photos(tmp_path) -> None:
    """Фото копируются в public под безопасными именами, карта имён возвращается."""
    field = tmp_path / "field"
    field.mkdir()
    (field / "вино 1.jpg").write_bytes(b"x")
    public = tmp_path / "public"
    mapping = copy_photos(field, public, ["вино 1.jpg", "нет.jpg"])
    assert mapping == {"вино 1.jpg": "img/01.jpg"}
    assert (public / "01.jpg").is_file()
