# START_MODULE_CONTRACT
# PURPOSE: Тесты C-073 бенча читателей: ожидаемая винодельня, качество чтения, сквозные метрики и отчёт
# SCOPE: CATALOG, GOLD, READINGS, test_expected_winery_matched_ambiguous_missing,
#        test_reading_quality_similarity_and_brand, test_e2e_accuracy_and_honest_not_found,
#        test_pick_best_worst_readers, test_conclusions_and_markdown_have_table_and_examples,
#        test_not_measured_reader_is_reported
# DEPENDS: packages.common.text_rank, scripts.eval.vision_bench
# LINKS: M-EVAL, M-OCR, M-VLM
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# CATALOG - синтетический каталог для сквозных метрик бенча
# GOLD - синтетический эталон с однозначным, неоднозначным и ненайденным фото
# READINGS - синтетические чтения читателя для проверки двух слоёв метрик
# test_expected_winery_matched_ambiguous_missing - винодельня эталона по статусу
# test_reading_quality_similarity_and_brand - слой качества чтения: похожесть и бренд
# test_e2e_accuracy_and_honest_not_found - сквозной слой: acc@1/@5 и честный not_found
# test_pick_best_worst_readers - выбор лучшего и худшего читателя
# test_conclusions_and_markdown_have_table_and_examples - выводы и Markdown содержат таблицу и примеры
# test_not_measured_reader_is_reported - читатель без измерения помечается «не измерен», а не None
# END_MODULE_MAP
"""Тесты бенча читателей C-073 (чистые функции, без вызова моделей)."""
from __future__ import annotations

from packages.common.text_rank import text_index
from scripts.eval.vision_bench import (build_conclusions, build_markdown, e2e_for_reader, expected_winery,
                                       pick_best_worst, reading_quality)

CATALOG = {
    "s1": {"winery": "Фанагория", "name": "5 Elements", "grape": "Пино Гри", "year": 2024,
           "sweetness": "брют"},
    "s2": {"winery": "Абрау-Дюрсо", "name": "Шардоне", "grape": "Шардоне", "year": None,
           "sweetness": "сухое"},
    "s3": {"winery": "Массандра", "name": "Мускатель белый", "grape": "Мускатель", "year": 2023,
           "sweetness": "сладкое"},
}
GOLD = {"files": {
    "a.jpg": {"status": "matched", "expected_slug": "s1", "human_label": "FANAGORIA 5 ELEMENTS"},
    "b.jpg": {"status": "ambiguous", "candidates": [{"slug": "s2"}], "human_label": "Abrau Durso"},
    "c.jpg": {"status": "not_found", "human_label": "Bosca"},
}, "counts": {"total": 3, "matched": 1, "ambiguous": 1, "not_found": 1, "evaluable": 2}}
READINGS = {
    "a.jpg": {"file": "a.jpg", "raw_text": "winery: Фанагория\nwine_name: 5 Elements", "ok": True,
              "latency_ms": 100.0, "wall_ms": 120.0, "reason": ""},
    "b.jpg": {"file": "b.jpg", "raw_text": "winery: Абрау-Дюрсо\nwine_name: Шардоне", "ok": True,
              "latency_ms": 200.0, "wall_ms": 220.0, "reason": ""},
    "c.jpg": {"file": "c.jpg", "raw_text": "winery: неразборчиво", "ok": True,
              "latency_ms": 300.0, "wall_ms": 320.0, "reason": ""},
}

__all__ = ["CATALOG", "GOLD", "READINGS", "test_conclusions_and_markdown_have_table_and_examples",
           "test_e2e_accuracy_and_honest_not_found", "test_expected_winery_matched_ambiguous_missing",
           "test_not_measured_reader_is_reported", "test_pick_best_worst_readers",
           "test_reading_quality_similarity_and_brand"]


def test_expected_winery_matched_ambiguous_missing() -> None:
    """Винодельня эталона берётся по ожидаемому slug (matched) или первому кандидату (ambiguous)."""
    assert expected_winery(GOLD["files"]["a.jpg"], CATALOG) == "Фанагория"
    assert expected_winery(GOLD["files"]["b.jpg"], CATALOG) == "Абрау-Дюрсо"
    assert expected_winery(GOLD["files"]["c.jpg"], CATALOG) == ""


def test_reading_quality_similarity_and_brand() -> None:
    """Слой чтения считает похожесть на название и долю прочитанных брендов."""
    quality = reading_quality(list(READINGS.values()), GOLD, CATALOG)
    assert quality["samples"] == 3
    assert quality["observations_nonempty"] == 3
    assert quality["title_similarity_mean"] > 0.5
    assert quality["brand_evaluated"] == 2          # not_found фото бренд не оценивает
    assert quality["brand_read_rate"] == 1.0        # оба бренда прочитаны
    assert quality["winery_similarity_mean"] > 0.8


def test_e2e_accuracy_and_honest_not_found() -> None:
    """Сквозной слой: наблюдение → поиск по каталогу → slug, accurate и честный not_found."""
    index = text_index(CATALOG)
    results = e2e_for_reader("test-reader", READINGS, GOLD, CATALOG, index)
    by_file = {row["file"]: row for row in results}
    assert by_file["a.jpg"]["top1"] == "s1"
    assert by_file["b.jpg"]["top1"] == "s2"
    assert by_file["c.jpg"]["not_found"] is True and by_file["c.jpg"]["top1"] is None


def test_pick_best_worst_readers() -> None:
    """Лучший и худший читатель выбираются по сквозной точности."""
    e2e = {"good": {"accuracy_at_1": 0.9}, "bad": {"accuracy_at_1": 0.1}}
    quality = {"good": {"title_similarity_mean": 0.8}, "bad": {"title_similarity_mean": 0.2}}
    assert pick_best_worst(e2e, quality) == ("good", "bad")


def test_conclusions_and_markdown_have_table_and_examples() -> None:
    """Выводы и Markdown содержат таблицу читателей и примеры лучшего/худшего."""
    index = text_index(CATALOG)
    results = e2e_for_reader("test-reader", READINGS, GOLD, CATALOG, index)
    quality = {"test-reader": reading_quality(list(READINGS.values()), GOLD, CATALOG)}
    from scripts.eval.field21_abc import compute_metrics
    e2e = {"test-reader": compute_metrics(results, GOLD)}
    timing = {"test-reader": {"n": 3, "p50": 100.0, "p95": 200.0, "mean": 120.0, "failures": 0}}
    payload = {
        "readers": ["test-reader"],
        "gold": GOLD,
        "quality": quality,
        "e2e": e2e,
        "timing": timing,
        "timing_limit": 3,
        "timing_dir": "data/samples/real100",
        "readings": {"test-reader": READINGS},
        "results": {"test-reader": results},
        "best_reader": "test-reader",
        "worst_reader": "test-reader",
        "conclusions": [],
    }
    payload["conclusions"] = build_conclusions(payload)
    assert any("Лучший онлайн" in note or "онлайн" in note for note in payload["conclusions"])
    markdown = build_markdown(payload)
    assert "test-reader" in markdown
    assert "a.jpg" in markdown and "title_sim" in markdown


def test_not_measured_reader_is_reported() -> None:
    """Читатель без измерения помечается «не измерен» с причиной, а не None-метриками."""
    quality = {"test-reader": {"title_similarity_mean": 0.5, "brand_read_rate": 1.0,
                               "brand_evaluated": 1}}
    e2e = {"test-reader": {"accuracy_at_1": 0.5, "accuracy_at_5": 0.5, "not_found_rate": 0.0,
                           "latency_ms": {"p50": 100.0, "p95": 200.0}}}
    payload = {
        "readers": ["test-reader", "paddleocr-vl"],
        "gold": GOLD,
        "quality": quality,
        "e2e": e2e,
        "timing": {"test-reader": {"n": 3, "p50": 100.0, "p95": 200.0, "mean": 120.0, "failures": 0}},
        "timing_limit": 3,
        "timing_dir": "data/samples/real100",
        "readings": {"test-reader": READINGS},
        "results": {"test-reader": []},
        "best_reader": "test-reader",
        "worst_reader": "test-reader",
        "conclusions": [],
    }
    payload["conclusions"] = build_conclusions(payload)
    assert any("не измерен" in note for note in payload["conclusions"])
    markdown = build_markdown(payload)
    assert "не измерен" in markdown
    assert "PaddleOCR-VL" in markdown
