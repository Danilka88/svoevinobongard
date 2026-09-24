# START_MODULE_CONTRACT
# PURPOSE: Обезличенная телеметрия интерфейса, санитайз и сводка
# SCOPE: test_sanitize_drops_unknown_fields, test_sanitize_rejects_unknown_type_and_non_objects, test_sanitize_truncates_long_strings_and_route, test_sanitize_keeps_only_valid_latency_and_seq, test_sanitize_accepts_all_declared_types, test_append_events_writes_only_valid_and_stamps, test_append_events_rotates_by_size, test_append_events_best_effort_on_broken_path, test_summary_structure_and_scenarios, test_summary_on_missing_file_is_empty, test_telemetry_endpoint_writes_and_reports, test_telemetry_endpoint_enforces_batch_limit, test_telemetry_endpoint_tolerates_garbage, test_telemetry_summary_endpoint, test_scan_accepts_session_header_without_changing_contract
# DEPENDS: packages.common.telemetry (ALLOWED_TYPES, MAX_EVENTS_PER_BATCH, append_events, load_events, sanitize_event, telemetry_summary), services.gateway.app (main), packages.contracts.schemas (ScanResponse)
# LINKS: M-GATEWAY, M-DATA
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# client - константа теста
# test_sanitize_drops_unknown_fields - проверяет: sanitize drops unknown fields
# test_sanitize_rejects_unknown_type_and_non_objects - проверяет: sanitize rejects unknown type and non objects
# test_sanitize_truncates_long_strings_and_route - проверяет: sanitize truncates long strings and route
# test_sanitize_keeps_only_valid_latency_and_seq - проверяет: sanitize keeps only valid latency and seq
# test_sanitize_accepts_all_declared_types - проверяет: sanitize accepts all declared types
# test_append_events_writes_only_valid_and_stamps - проверяет: append events writes only valid and stamps
# test_append_events_rotates_by_size - проверяет: append events rotates by size
# test_append_events_best_effort_on_broken_path - Плохой путь не поднимает исключение: телеметрия не ломает основной сценарий
# test_summary_structure_and_scenarios - проверяет: summary structure and scenarios
# test_summary_on_missing_file_is_empty - проверяет: summary on missing file is empty
# test_telemetry_endpoint_writes_and_reports - проверяет: telemetry endpoint writes and reports
# test_telemetry_endpoint_enforces_batch_limit - проверяет: telemetry endpoint enforces batch limit
# test_telemetry_endpoint_tolerates_garbage - Мусорный батч не приводит к ошибке: приём всегда успешен
# test_telemetry_summary_endpoint - проверяет: telemetry summary endpoint
# test_scan_accepts_session_header_without_changing_contract - Заголовок сессии не меняет форму ответа сканера
# END_MODULE_MAP
"""Тесты телеметрии C-044: санитайз, лимиты, ротация, сводка и best-effort приём.

Ключевая гарантия — приватность: всё, что не входит в белый список, не попадает в лог.
Вторая гарантия — устойчивость: сбой телеметрии не влияет ни на интерфейс, ни на API.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from packages.common.telemetry import (  # noqa: E402
    ALLOWED_TYPES,
    MAX_EVENTS_PER_BATCH,
    append_events,
    load_events,
    sanitize_event,
    telemetry_summary,
)


from services.gateway.app import main as gw  # noqa: E402

__all__ = [
    "client",
    "test_sanitize_drops_unknown_fields",
    "test_sanitize_rejects_unknown_type_and_non_objects",
    "test_sanitize_truncates_long_strings_and_route",
    "test_sanitize_keeps_only_valid_latency_and_seq",
    "test_sanitize_accepts_all_declared_types",
    "test_append_events_writes_only_valid_and_stamps",
    "test_append_events_rotates_by_size",
    "test_append_events_best_effort_on_broken_path",
    "test_summary_structure_and_scenarios",
    "test_summary_on_missing_file_is_empty",
    "test_telemetry_endpoint_writes_and_reports",
    "test_telemetry_endpoint_enforces_batch_limit",
    "test_telemetry_endpoint_tolerates_garbage",
    "test_telemetry_summary_endpoint",
    "test_scan_accepts_session_header_without_changing_contract",
]

client = TestClient(gw.app)


# --- санитайз ------------------------------------------------------------------

def test_sanitize_drops_unknown_fields() -> None:
    clean = sanitize_event({
        "type": "scan", "action": "ok", "latency_ms": 123.45, "ok": True,
        "image": "base64...", "email": "user@example.com", "comment": "свободный текст",
        "winery": "Массандра", "nested": {"a": 1},
    })
    assert clean == {"type": "scan", "action": "ok", "latency_ms": 123.5, "ok": True}


def test_sanitize_rejects_unknown_type_and_non_objects() -> None:
    assert sanitize_event({"type": "keylogger", "action": "x"}) is None
    assert sanitize_event("scan") is None
    assert sanitize_event(None) is None
    assert sanitize_event({"action": "x"}) is None


def test_sanitize_truncates_long_strings_and_route() -> None:
    clean = sanitize_event({"type": "action", "action": "a" * 300, "detail": "b" * 300,
                            "route": "/catalog/" + "c" * 300})
    assert clean is not None
    assert len(clean["action"]) == 64
    assert len(clean["detail"]) == 64
    assert len(clean["route"]) <= 120


def test_sanitize_keeps_only_valid_latency_and_seq() -> None:
    clean = sanitize_event({"type": "action", "latency_ms": "not-a-number", "seq": "7"})
    assert clean is not None
    assert "latency_ms" not in clean
    assert clean["seq"] == 7


@pytest.mark.parametrize("kind", ALLOWED_TYPES)
def test_sanitize_accepts_all_declared_types(kind: str) -> None:
    assert sanitize_event({"type": kind}) == {"type": kind}


# --- запись и ротация -----------------------------------------------------------

def test_append_events_writes_only_valid_and_stamps(tmp_path: Path) -> None:
    target = tmp_path / "ui.jsonl"
    written = append_events([
        {"type": "scan", "action": "start"},
        {"type": "unknown", "action": "drop-me"},
        {"type": "page_view", "route": "/"},
    ], "session-1", target)
    assert written == 2
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert all(row["session_id"] == "session-1" for row in rows)
    assert all(isinstance(row["ts"], str) and row["recv"] > 0 for row in rows)


def test_append_events_rotates_by_size(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "ui.jsonl"
    append_events([{"type": "page_view", "route": "/old"}], "s1", target)
    monkeypatch.setattr("packages.common.telemetry.MAX_FILE_BYTES", 1)
    append_events([{"type": "page_view", "route": "/new"}], "s1", target)
    rotated = target.with_suffix(target.suffix + ".1")
    assert rotated.is_file()  # старый файл ушёл в архив
    assert "old" in rotated.read_text(encoding="utf-8")
    assert "new" in target.read_text(encoding="utf-8")


def test_append_events_best_effort_on_broken_path(tmp_path: Path) -> None:
    """Плохой путь не поднимает исключение: телеметрия не ломает основной сценарий."""
    blocker = tmp_path / "file.txt"
    blocker.write_text("not a directory")
    assert append_events([{"type": "page_view"}], "s1", blocker / "nested" / "ui.jsonl") == 0


# --- сводка --------------------------------------------------------------------

def test_summary_structure_and_scenarios(tmp_path: Path) -> None:
    target = tmp_path / "ui.jsonl"
    append_events([
        {"type": "page_view", "route": "/", "action": "view"},
        {"type": "scan", "action": "scan_start"},
        {"type": "scan", "action": "scan_done", "ok": True, "latency_ms": 1200},
        {"type": "scan", "action": "scan_not_found", "ok": False, "latency_ms": 900},
        {"type": "sommelier", "action": "step", "value": 2},
        {"type": "catalog", "action": "search", "latency_ms": 80},
        {"type": "theme", "action": "switch", "value": "dark"},
        {"type": "js_error", "action": "window_error", "detail": "runtime"},
        {"type": "api_error", "action": "scan", "detail": "failed"},
    ], "session-a", target)
    append_events([{"type": "page_view", "route": "/catalog", "action": "view"}], "session-b", target)
    summary = telemetry_summary(target)
    assert summary["events"] == 10
    assert summary["sessions"] == 2
    assert summary["by_route"]["/"] == 1
    assert summary["errors"] == {"js_error": 1, "api_error": 1}
    assert summary["scenarios"] == {"scans": 3, "scans_not_found": 1, "sommelier_steps": 1,
                                    "catalog_searches": 1, "theme_switches": 1}
    assert summary["latency"]["samples"] == 3
    assert summary["latency"]["p50_ms"] > 0
    assert len(summary["last_errors"]) == 2


def test_summary_on_missing_file_is_empty(tmp_path: Path) -> None:
    summary = telemetry_summary(tmp_path / "absent.jsonl")
    assert summary["events"] == 0 and summary["sessions"] == 0
    assert summary["latency"] == {"p50_ms": 0.0, "p95_ms": 0.0, "samples": 0}


# --- эндпоинты -----------------------------------------------------------------

def test_telemetry_endpoint_writes_and_reports(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "ui.jsonl"
    monkeypatch.setattr(gw, "_telemetry_path", lambda: str(target))
    response = client.post("/v1/telemetry", json={
        "session_id": "abc-123",
        "events": [{"type": "page_view", "route": "/"}, {"type": "scan", "action": "scan_start"},
                   {"type": "evil", "secret": "leak"}],
    })
    assert response.status_code == 200
    assert response.json()["accepted"] == 2
    rows = load_events(target)
    assert len(rows) == 2
    assert all("secret" not in row for row in rows)


def test_telemetry_endpoint_enforces_batch_limit(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "ui.jsonl"
    monkeypatch.setattr(gw, "_telemetry_path", lambda: str(target))
    events = [{"type": "page_view", "route": "/"} for _ in range(MAX_EVENTS_PER_BATCH + 50)]
    response = client.post("/v1/telemetry", json={"session_id": "s", "events": events})
    assert response.status_code == 200
    assert response.json()["accepted"] == MAX_EVENTS_PER_BATCH
    assert len(load_events(target)) == MAX_EVENTS_PER_BATCH


def test_telemetry_endpoint_tolerates_garbage(monkeypatch, tmp_path: Path) -> None:
    """Мусорный батч не приводит к ошибке: приём всегда успешен."""
    monkeypatch.setattr(gw, "_telemetry_path", lambda: str(tmp_path / "ui.jsonl"))
    for payload in ({"events": "not-a-list"}, {"events": [1, "x", None]}, {}, {"session_id": "s"}):
        response = client.post("/v1/telemetry", json=payload)
        assert response.status_code in {200, 204}


def test_telemetry_summary_endpoint(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "ui.jsonl"
    monkeypatch.setattr(gw, "_telemetry_path", lambda: str(target))
    client.post("/v1/telemetry", json={"session_id": "s1", "events": [
        {"type": "scan", "action": "scan_done", "ok": True, "latency_ms": 500},
        {"type": "api_error", "action": "catalog", "detail": "failed"},
    ]})
    body = client.get("/v1/telemetry/summary").json()
    assert body["events"] == 2
    assert body["sessions"] == 1
    assert body["scenarios"]["scans"] == 1
    assert body["errors"]["api_error"] == 1
    assert body["last_errors"][0]["action"] == "catalog"


def test_scan_accepts_session_header_without_changing_contract(monkeypatch) -> None:
    """Заголовок сессии не меняет форму ответа сканера."""
    from packages.contracts.schemas import ScanResponse

    async def fake_scan(*args, **kwargs):
        assert kwargs.get("session_id") == "sess-42"
        return ScanResponse(slug="lesnaya-proseka", confidence=0.9)

    monkeypatch.setattr(gw, "run_scan", fake_scan)
    response = client.post("/v1/scan", files={"image": ("q.jpg", b"\xff\xd8\xff" + b"0" * 64, "image/jpeg")},
                           headers={"X-Session-Id": "sess-42"})
    assert response.status_code == 200
    assert response.json()["slug"] == "lesnaya-proseka"
