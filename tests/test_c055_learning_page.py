"""Тесты C-055: страница обучения, решения модератора, счётчики датасета и навигация «5 + Ещё».

Проверяется аддитивность решения модератора (оно фиксируется с провенансом и отражается в сводке),
счётчики датасета по разбиениям и то, что навигация остаётся масштабируемой и доступной.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from services.gateway.app import main as gw  # noqa: E402

client = TestClient(gw.app)
WEB = _ROOT / "apps" / "web"
SHA = "d" * 64


@pytest.fixture(autouse=True)
def _paths(tmp_path: Path, monkeypatch):
    """Изолируем хранилища: тесты не трогают рабочие файлы."""
    monkeypatch.setattr(gw, "_feedback_path", lambda: str(tmp_path / "feedback.jsonl"))
    monkeypatch.setattr(gw, "_decision_path", lambda: str(tmp_path / "decisions.jsonl"))
    monkeypatch.setattr(gw, "_telemetry_path", lambda: str(tmp_path / "ui.jsonl"))
    monkeypatch.setattr(gw, "_annotation_dir", lambda: tmp_path / "annotation")
    monkeypatch.setattr(gw, "_dataset_dirs", lambda: [tmp_path / "annotation"])
    yield


def _vote(session: str, vote: str = "down", *, slug: str = "nesushchestvuyushchee-vino") -> dict:
    return {"vote": vote, "slug": slug, "image_sha256": SHA, "reason": ["wrong_wine"]}


# --- решения модератора ---------------------------------------------------------

def test_decision_requires_known_choice() -> None:
    response = client.post("/v1/learning/decision", json={"decision": "maybe", "slug": "w"})
    assert response.status_code == 400


def test_decision_is_recorded_with_provenance() -> None:
    response = client.post("/v1/learning/decision",
                           json={"decision": "accept", "slug": "wine-1", "image_sha256": SHA,
                                 "reason": "подтверждено модератором"},
                           headers={"X-Session-Id": "moderator-demo"})
    assert response.status_code == 200
    body = response.json()
    assert body["accepted"] is True
    assert body["decision"] == "accept"
    assert body["provenance"]["session_id"] == "moderator-demo"
    assert body["provenance"]["timestamp"]
    rows = [json.loads(line) for line in Path(gw._decision_path()).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows and rows[0]["decision"] == "accept"


def test_decision_removes_case_from_queue_and_updates_summary() -> None:
    """После решения модератора кейс уходит из очереди, а сводка это отражает."""
    client.post("/v1/feedback", json=_vote("solo"), headers={"X-Session-Id": "solo"})
    assert client.get("/v1/learning/queue").json()["total"] == 1
    client.post("/v1/learning/decision", json={"decision": "reject", "slug": "nesushchestvuyushchee-vino",
                                               "image_sha256": SHA})
    queue = client.get("/v1/learning/queue").json()
    assert queue["total"] == 0
    summary = client.get("/v1/learning/summary").json()
    assert summary["decisions"]["total"] == 1 and summary["decisions"]["rejected"] == 1


def test_decision_without_target_is_rejected() -> None:
    response = client.post("/v1/learning/decision", json={"decision": "accept"})
    assert response.status_code == 400


# --- счётчики датасета ----------------------------------------------------------

def test_dataset_endpoint_counts_splits(tmp_path: Path) -> None:
    annotation = tmp_path / "annotation"
    annotation.mkdir(parents=True, exist_ok=True)
    (annotation / "consensus.jsonl").write_text(
        json.dumps({"file": "a", "split": "eval"}) + "\n" + json.dumps({"file": "b", "split": "eval"}) + "\n",
        encoding="utf-8")
    (annotation / "escalations.jsonl").write_text(json.dumps({"file": "c", "split": "holdout"}) + "\n",
                                                  encoding="utf-8")
    body = client.get("/v1/learning/dataset").json()
    splits = {row["split"]: row for row in body["splits"]}
    assert splits["eval"]["records"] == 2
    assert splits["holdout"]["records"] == 1
    assert body["total_records"] == 3


def test_dataset_endpoint_is_empty_safe(tmp_path: Path) -> None:
    body = client.get("/v1/learning/dataset").json()
    assert body["total_records"] == 0 and body["splits"] == []


# --- страница и навигация -------------------------------------------------------

def test_learning_page_exists_and_uses_endpoints() -> None:
    page = (WEB / "pages" / "learning.vue").read_text(encoding="utf-8")
    for call in ("learningSummary", "learningQueue", "telemetrySummary", "learningDataset", "learningDecision"):
        assert call in page, f"страница не использует {call}"
    assert "Раздел для модератора" in page
    assert "<h1>" in page
    assert "ServiceUnavailable" in page  # пустое состояние вместо пустоты при сбое


def test_navigation_keeps_five_primary_and_more_sheet() -> None:
    layout = (WEB / "layouts" / "default.vue").read_text(encoding="utf-8")
    assert "primaryNav" in layout and "moreNav" in layout
    assert "aria-expanded" in layout and "aria-current" in layout
    assert "'/learning'" in layout
    assert "Escape" in layout  # закрытие листа с клавиатуры
    assert "role=\"dialog\"" in layout


def test_all_pages_are_reachable_from_navigation() -> None:
    """Страница не должна теряться: каждая есть либо в основных пунктах, либо в «Ещё»."""
    layout = (WEB / "layouts" / "default.vue").read_text(encoding="utf-8")
    pages = sorted(path.stem for path in (WEB / "pages").glob("*.vue") if path.stem != "index")
    for page in pages:
        assert f"'/{page}'" in layout, f"страница /{page} недостижима из навигации"
    assert "'/'" in layout  # сканер


def test_primary_nav_has_exactly_five_items() -> None:
    layout = (WEB / "layouts" / "default.vue").read_text(encoding="utf-8")
    block = re.search(r"const primaryNav = \[(.*?)\];", layout, re.S)
    assert block, "не найден список основных пунктов"
    assert block.group(1).count("to:") == 5


def test_gateway_client_exposes_learning_calls() -> None:
    source = (WEB / "composables" / "useGateway.ts").read_text(encoding="utf-8")
    for call in ("learningSummary", "learningQueue", "learningDataset", "learningDecision", "telemetrySummary"):
        assert call in source, f"клиент не отдаёт {call}"
    calls = re.findall(r"\$fetch<", source)
    signals = re.findall(r"signal: requestTimeout\(", source)
    assert len(signals) == len(calls), "у каждого вызова должен быть таймаут"
