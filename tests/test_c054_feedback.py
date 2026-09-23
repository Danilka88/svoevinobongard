"""Тесты C-054: правила доверия к голосам, дедуп, лимит частоты и очередь модерации.

Проверяется главная гарантия анти-накрутки: одиночный голос не влияет никогда, влияние появляется
только при согласии независимых сессий и консенсуса агентов. Модели и сеть не участвуют.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from packages.common.learning import (  # noqa: E402
    FEEDBACK_REASONS,
    MIN_TRUSTED_SESSIONS,
    RATE_LIMIT_PER_MINUTE,
    TRUST_AGREEMENT,
    append_feedback,
    dedupe_votes,
    evaluate_trust,
    feedback_summary,
    load_feedback,
    moderation_queue,
    rate_limit_exceeded,
    sanitize_feedback,
    vote_key,
)
from services.gateway.app import main as gw  # noqa: E402

client = TestClient(gw.app)
SHA = "a" * 64


def _vote(session: str, vote: str = "up", *, source: str = "user", slug: str = "wine-1",
          sha: str = SHA, reason: list[str] | None = None, timestamp: str | None = None) -> dict:
    return {"vote": vote, "reason": reason or [], "slug": slug, "image_sha256": sha,
            "session_id": session, "source": source,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds")}


# --- санитайз и дедуп -----------------------------------------------------------

def test_sanitize_accepts_only_known_votes_and_reasons() -> None:
    good = sanitize_feedback({"vote": "UP", "reason": ["не то вино", "мусор"], "slug": "w"})
    assert good is not None and good["vote"] == "up" and good["reason"] == ["wrong_wine"]
    assert sanitize_feedback({"vote": "maybe"}) is None
    assert sanitize_feedback("up") is None
    assert sanitize_feedback({"vote": "down", "reason": ["a"] * 10})["reason"].__len__() <= 3


def test_vote_key_uses_session_and_image() -> None:
    assert vote_key(_vote("s1")) == ("s1", SHA)
    assert vote_key(_vote("s1", sha="", slug="wine-2")) == ("s1", "slug:wine-2")


def test_dedupe_keeps_one_vote_per_session_and_image() -> None:
    votes = [_vote("s1", "up"), _vote("s1", "down"), _vote("s2", "up")]
    assert len(dedupe_votes(votes)) == 2


# --- лимит частоты --------------------------------------------------------------

def test_rate_limit_blocks_flood_from_one_session() -> None:
    now = datetime.now(timezone.utc)
    votes = [_vote("s1", timestamp=(now - timedelta(seconds=5)).isoformat(timespec="seconds"))
             for _ in range(RATE_LIMIT_PER_MINUTE)]
    assert rate_limit_exceeded(votes, "s1", now=now) is True
    assert rate_limit_exceeded(votes, "s2", now=now) is False
    old = [_vote("s1", timestamp=(now - timedelta(minutes=5)).isoformat(timespec="seconds"))
           for _ in range(RATE_LIMIT_PER_MINUTE)]
    assert rate_limit_exceeded(old, "s1", now=now) is False


# --- правила доверия ------------------------------------------------------------

def test_single_vote_never_influences() -> None:
    verdict = evaluate_trust([_vote("s1")], consensus={"vote": "up"}, image_sha256=SHA)
    assert verdict["trusted"] is False and verdict["applied"] is False
    assert any("независимых сессий" in reason for reason in verdict["reasons"])


def test_three_independent_sessions_with_agent_consensus_are_trusted() -> None:
    votes = [_vote(f"s{index}") for index in range(MIN_TRUSTED_SESSIONS)]
    verdict = evaluate_trust(votes, consensus={"vote": "up"}, image_sha256=SHA)
    assert verdict["trusted"] is True and verdict["applied"] is True
    assert verdict["sessions"] == MIN_TRUSTED_SESSIONS
    assert verdict["agreement"] >= TRUST_AGREEMENT


def test_three_sessions_without_agent_consensus_are_not_trusted() -> None:
    votes = [_vote(f"s{index}") for index in range(MIN_TRUSTED_SESSIONS)]
    assert evaluate_trust(votes, consensus=None, image_sha256=SHA)["trusted"] is False
    disagreeing = evaluate_trust(votes, consensus={"vote": "down"}, image_sha256=SHA)
    assert disagreeing["trusted"] is False
    assert any("расходится" in reason for reason in disagreeing["reasons"])


def test_split_vote_fails_agreement_threshold() -> None:
    votes = [_vote("s1"), _vote("s2"), _vote("s3", "down"), _vote("s4", "down")]
    verdict = evaluate_trust(votes, consensus={"vote": "up"}, image_sha256=SHA)
    assert verdict["trusted"] is False
    assert any("согласие" in reason for reason in verdict["reasons"])


def test_single_outlier_is_ignored() -> None:
    """Одиночный голос против сильного консенсуса агентов влияния не получает."""
    votes = [_vote("outlier", "down")]
    verdict = evaluate_trust(votes, consensus={"vote": "up"}, image_sha256=SHA)
    assert verdict["trusted"] is False and verdict["applied"] is False
    assert verdict["agent_vote"] == "up" and verdict["agent_agrees"] is False


def test_duplicate_votes_from_one_session_do_not_add_sessions() -> None:
    votes = [_vote("s1") for _ in range(5)]
    verdict = evaluate_trust(votes, consensus={"vote": "up"}, image_sha256=SHA)
    assert verdict["sessions"] == 1 and verdict["trusted"] is False


# --- хранилище, сводка и очередь ------------------------------------------------

def test_append_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "feedback.jsonl"
    assert append_feedback([_vote("s1")], path) == 1
    stored = load_feedback(path)
    assert len(stored) == 1 and stored[0]["vote"] == "up"


def test_broken_storage_is_non_fatal(tmp_path: Path) -> None:
    blocker = tmp_path / "file.txt"
    blocker.write_text("not a directory")
    assert append_feedback([_vote("s1")], blocker / "nested" / "feedback.jsonl") == 0


def test_summary_counts_and_queue(tmp_path: Path) -> None:
    votes = [_vote(f"s{index}") for index in range(MIN_TRUSTED_SESSIONS)]
    votes.append(_vote("solo", "down", slug="wine-2", sha="b" * 64, reason=["wrong_wine"]))
    votes.append(_vote("agent-1", "up", source="agent"))
    summary = feedback_summary(votes)
    assert summary["unique_votes"] == 5
    assert summary["up"] == 4 and summary["down"] == 1
    assert summary["trusted_targets"] == 1
    assert summary["queued_targets"] == 1
    assert summary["rules"]["min_sessions"] == MIN_TRUSTED_SESSIONS
    queue = moderation_queue(votes)
    assert len(queue) == 1
    assert queue[0]["slug"] == "wine-2" and queue[0]["needs_moderator"] is True
    assert queue[0]["provenance"][0]["reason"] == ["wrong_wine"]


def test_wrong_wine_reason_goes_to_queue() -> None:
    votes = [_vote("s1", "down", reason=["wrong_wine"])]
    queue = moderation_queue(votes)
    assert len(queue) == 1
    assert "wrong_wine" in queue[0]["provenance"][0]["reason"]


# --- эндпоинты ------------------------------------------------------------------

def test_feedback_endpoint_single_vote_goes_to_queue(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gw, "_feedback_path", lambda: str(tmp_path / "feedback.jsonl"))
    response = client.post("/v1/feedback", json={"vote": "up", "slug": "wine-1", "image_sha256": SHA},
                           headers={"X-Session-Id": "session-a"})
    assert response.status_code == 200
    body = response.json()
    assert body["accepted"] is True
    assert body["trusted"] is False and body["applied"] is False
    assert body["sessions"] == 1


def test_feedback_endpoint_three_sessions_become_trusted(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(gw, "_feedback_path", lambda: str(path))
    for session in ("session-a", "session-b", "session-c"):
        response = client.post("/v1/feedback", json={"vote": "up", "slug": "wine-1", "image_sha256": SHA},
                               headers={"X-Session-Id": session})
        assert response.status_code == 200
    body = client.post("/v1/feedback", json={"vote": "up", "slug": "wine-1", "image_sha256": SHA},
                       headers={"X-Session-Id": "session-d"}).json()
    assert body["trusted"] is True and body["applied"] is True
    assert body["sessions"] >= MIN_TRUSTED_SESSIONS


def test_feedback_endpoint_rejects_unknown_vote(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gw, "_feedback_path", lambda: str(tmp_path / "feedback.jsonl"))
    response = client.post("/v1/feedback", json={"vote": "great"}, headers={"X-Session-Id": "s"})
    assert response.status_code == 400


def test_feedback_reason_codes_are_known() -> None:
    assert "wrong_wine" in FEEDBACK_REASONS and "bad_photo" in FEEDBACK_REASONS


def test_learning_summary_and_queue_endpoints(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(gw, "_feedback_path", lambda: str(path))
    client.post("/v1/feedback", json={"vote": "down", "slug": "wine-9", "image_sha256": "c" * 64,
                                      "reason": ["wrong_year"]},
                headers={"X-Session-Id": "solo"})
    summary = client.get("/v1/learning/summary").json()
    assert summary["votes"] == 1 and summary["sessions"] == 1
    assert summary["queued_targets"] == 1
    assert summary["rules"]["min_sessions"] == MIN_TRUSTED_SESSIONS
    queue = client.get("/v1/learning/queue").json()
    assert queue["items"] and queue["items"][0]["slug"] == "wine-9"


def test_feedback_writes_telemetry_event(monkeypatch, tmp_path: Path) -> None:
    """Событие фидбека попадает в телеметрию: видно, что пользователь голосовал."""
    feedback_path = tmp_path / "feedback.jsonl"
    telemetry_path = tmp_path / "ui.jsonl"
    monkeypatch.setattr(gw, "_feedback_path", lambda: str(feedback_path))
    monkeypatch.setattr(gw, "_telemetry_path", lambda: str(telemetry_path))
    client.post("/v1/feedback", json={"vote": "up", "slug": "wine-1"}, headers={"X-Session-Id": "s-tel"})
    events = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(event.get("action") == "feedback" for event in events)
