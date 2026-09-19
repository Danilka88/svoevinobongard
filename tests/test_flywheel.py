# START_MODULE_CONTRACT
# PURPOSE: Верификация flywheel: лог без PII, очередь ревью, human-in-the-loop решения, экспорт подтверждённых, реестр
# SCOPE: test_build_log_record_has_no_pii, test_append_query_log_writes_jsonl, test_store_records_and_builds_queue, test_review_rejects_invalid_decision, test_queue_excludes_reviewed, test_export_only_confirmed, test_export_corrected_uses_corrected_slug, test_ingest_jsonl_dedup
# DEPENDS: services.gateway.app.flywheel_log, services.active_learning.app.store, services.active_learning.app.dataset, packages.common.catalog
# LINKS: M-ACTIVE-LEARNING, M-GATEWAY
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# test_build_log_record_has_no_pii - запись лога содержит хеш изображения и скоры, но не сырые байты
# test_append_query_log_writes_jsonl - запись лога дописывается в JSONL построчно
# test_store_records_and_builds_queue - неуверенные кейсы идут в очереди первыми
# test_review_rejects_invalid_decision - недопустимое решение отклоняется
# test_queue_excludes_reviewed - отревьюенные кейсы уходят из очереди
# test_export_only_confirmed - в датасет попадают только подтверждённые метки
# test_export_corrected_uses_corrected_slug - исправленная метка экспортируется с исправленным slug
# test_ingest_jsonl_dedup - повторный ingest не создает дублей
# END_MODULE_MAP
"""V-M-ACTIVE-LEARNING: flywheel с human-in-the-loop."""
from __future__ import annotations

import json

import pytest

from packages.common.catalog import CatalogStore
from services.active_learning.app.dataset import export_confirmed
from services.active_learning.app.store import FlywheelStore, ingest_jsonl
from services.gateway.app.flywheel_log import append_query_log, build_log_record

__all__ = [
    "test_build_log_record_has_no_pii",
    "test_append_query_log_writes_jsonl",
    "test_store_records_and_builds_queue",
    "test_review_rejects_invalid_decision",
    "test_queue_excludes_reviewed",
    "test_export_only_confirmed",
    "test_export_corrected_uses_corrected_slug",
    "test_ingest_jsonl_dedup",
]

_CATALOG = {
    "kokur-suhoe-2025": {"slug": "kokur-suhoe-2025", "name": "Кокур Сухое",
                         "winery": "Захарьин", "grape": "Кокур", "region": "Крым",
                         "image": "a.webp"},
    "kokur-2025": {"slug": "kokur-2025", "name": "Кокур", "winery": "Захарьин",
                   "grape": "Кокур", "region": "Крым", "image": "b.webp"},
}


class _Response:
    """Минимальный двойник ScanResponse."""

    def __init__(self, slug: str = "kokur-suhoe-2025", margin: float = 0.5) -> None:
        self.slug = slug
        self.confidence = 0.42
        self.margin = margin
        self.path = "retrieval"
        self.latency_ms = 123.4
        self.not_found = False
        self.top5 = [type("Item", (), {"slug": slug, "score": 0.81})(),
                     type("Item", (), {"slug": "kokur-2025", "score": 0.52})()]


def _store(tmp_path) -> FlywheelStore:
    return FlywheelStore(tmp_path / "flywheel.db")


def _entry(slug: str = "kokur-suhoe-2025", margin: float = 0.5, sha: str = "deadbeef") -> dict:
    return {
        "timestamp": "2026-09-19T10:00:00+00:00",
        "image_sha256": sha,
        "slug": slug,
        "confidence": 0.42,
        "margin": margin,
        "path": "retrieval",
        "latency_ms": 123.4,
        "not_found": False,
        "uncertain": margin < 0.0659,
        "top_k": [{"slug": slug, "score": 0.81}],
        "normalize_method": "largest_bbox",
    }


def test_build_log_record_has_no_pii() -> None:
    raw = b"\x89PNG\r\n\x1a\n fake image bytes"
    record = build_log_record(raw, _Response(margin=0.01), margin_threshold=0.0659,
                              normalize_method="contour_quad")
    assert record["image_sha256"] and len(record["image_sha256"]) == 64
    assert record["uncertain"] is True
    assert record["top_k"][0]["slug"] == "kokur-suhoe-2025"
    assert record["normalize_method"] == "contour_quad"
    # сырых байт изображения в записи нет
    assert all(isinstance(value, (str, int, float, bool, list, type(None)))
               for value in record.values())
    assert "fake image bytes" not in json.dumps(record)


def test_append_query_log_writes_jsonl(tmp_path) -> None:
    path = tmp_path / "queries.jsonl"
    assert append_query_log(_entry(), path) is True
    assert append_query_log(_entry(sha="cafebabe"), path) is True
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["image_sha256"] == "deadbeef"


def test_store_records_and_builds_queue(tmp_path) -> None:
    store = _store(tmp_path)
    try:
        store.record(_entry(margin=0.30, sha="aa"))
        store.record(_entry(margin=0.01, sha="bb"))
        assert store.stats().total == 2
        queue = store.queue(limit=5)
        assert queue[0]["image_sha256"] == "bb"  # неуверенный кейс впереди
        assert queue[0]["uncertain"] == 1
    finally:
        store.close()


def test_review_rejects_invalid_decision(tmp_path) -> None:
    store = _store(tmp_path)
    try:
        store.record(_entry())
        with pytest.raises(ValueError):
            store.review(1, "maybe")
        with pytest.raises(ValueError):
            store.review(999, "confirmed")
    finally:
        store.close()


def test_queue_excludes_reviewed(tmp_path) -> None:
    store = _store(tmp_path)
    try:
        store.record(_entry(sha="aa"))
        store.record(_entry(sha="bb"))
        store.review(1, "confirmed", reviewer="qa")
        queue = store.queue(limit=10)
        assert [item["id"] for item in queue] == [2]
        assert store.stats().confirmed == 1
        assert store.stats().pending == 1
    finally:
        store.close()


def test_export_only_confirmed(tmp_path) -> None:
    store = _store(tmp_path)
    try:
        store.record(_entry(sha="aa"))
        store.record(_entry(slug="kokur-2025", sha="bb"))
        store.review(1, "confirmed", reviewer="qa")
        out = tmp_path / "confirmed.jsonl"
        summary = export_confirmed(store, CatalogStore(_CATALOG), out)
        assert summary["unique_slugs"] == 1
        assert summary["confirmed"] == 1
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        assert [row["slug"] for row in rows] == ["kokur-suhoe-2025"]
    finally:
        store.close()


def test_export_corrected_uses_corrected_slug(tmp_path) -> None:
    store = _store(tmp_path)
    try:
        store.record(_entry(slug="kokur-2025", sha="cc"))
        store.review(1, "corrected", corrected_slug="kokur-suhoe-2025", reviewer="qa")
        out = tmp_path / "confirmed.jsonl"
        summary = export_confirmed(store, CatalogStore(_CATALOG), out)
        assert summary["corrected"] == 1
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        assert rows[0]["slug"] == "kokur-suhoe-2025"
    finally:
        store.close()


def test_ingest_jsonl_dedup(tmp_path) -> None:
    log = tmp_path / "queries.jsonl"
    append_query_log(_entry(), log)
    store = _store(tmp_path)
    try:
        assert ingest_jsonl(store, log) == 1
        assert ingest_jsonl(store, log) == 0  # повторный ingest не создает дублей
        assert store.stats().total == 1
    finally:
        store.close()
