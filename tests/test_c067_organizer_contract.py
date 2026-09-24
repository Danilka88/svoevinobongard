# START_MODULE_CONTRACT
# PURPOSE: Страж контракта организатора: плоский slug на любом входе и метрики прогона настоящего скрипта
# SCOPE: ROOT, gw, participant, client, test_eval_predict_flat_contract_only_slug, test_eval_predict_empty_file_is_null_200, test_eval_predict_non_image_is_null_200, test_eval_predict_broken_image_is_null_200, test_eval_predict_internal_error_is_null_200, test_eval_predict_retrieval_unavailable_is_503, test_eval_response_schema_has_only_slug, test_participant_summarize_metrics, test_participant_http_errors_from_gateway_log, test_participant_markdown_reports_compatibility
# DEPENDS: services.gateway.app.main (eval_predict), packages.contracts.schemas (EvalPredictResponse), scripts.eval.participant_run (summarize, render_markdown)
# LINKS: M-GATEWAY, M-EVAL, M-CONTRACTS
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# ROOT - корень проекта
# participant - загруженный модуль прогона скрипта организатора
# client - тестовый HTTP-клиент gateway
# test_eval_predict_flat_contract_only_slug - ответ содержит ровно ключ slug
# test_eval_predict_empty_file_is_null_200 - пустой файл даёт плоский null без ошибки
# test_eval_predict_non_image_is_null_200 - не-изображение даёт плоский null без ошибки
# test_eval_predict_broken_image_is_null_200 - битый файл с валидной подписью даёт null, а не 5xx
# test_eval_predict_internal_error_is_null_200 - внутренний сбой даёт плоский null
# test_eval_predict_retrieval_unavailable_is_503 - недоступный retrieval остаётся явным 503
# test_eval_response_schema_has_only_slug - схема ответа не содержит других полей
# test_participant_summarize_metrics - метрики прогона считают долю непустых, перцентили и лимит
# test_participant_http_errors_from_gateway_log - HTTP-ошибки берутся из лога gateway по кодам
# test_participant_markdown_reports_compatibility - отчёт помечает совместимость при чистых числах
# END_MODULE_MAP
"""C-067: жёсткий контракт организатора и метрики прогона его скрипта.

Первая часть проверяет плоский ответ `POST /v1/eval/predict` на широком наборе плохих
входов: он обязан оставаться объектом ровно с ключом `slug` и не превращаться в 5xx.
Вторая часть проверяет, что отчёт прогона считает именно то, что важно организатору:
долю непустых ответов, перцентили задержки, долю в лимите и число HTTP-ошибок.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from packages.contracts.schemas import EvalPredictResponse  # noqa: E402
from services.gateway.app import main as gw  # noqa: E402

__all__ = [
    "ROOT",
    "gw",
    "participant",
    "client",
    "test_eval_predict_flat_contract_only_slug",
    "test_eval_predict_empty_file_is_null_200",
    "test_eval_predict_non_image_is_null_200",
    "test_eval_predict_broken_image_is_null_200",
    "test_eval_predict_internal_error_is_null_200",
    "test_eval_predict_retrieval_unavailable_is_503",
    "test_eval_response_schema_has_only_slug",
    "test_participant_summarize_metrics",
    "test_participant_http_errors_from_gateway_log",
    "test_participant_markdown_reports_compatibility",
]


def _load_participant():
    """Загружает скрипт прогона организатора как модуль без запуска main."""
    spec = importlib.util.spec_from_file_location(
        "svoe_participant_run", ROOT / "scripts" / "eval" / "participant_run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


participant = _load_participant()
client = TestClient(gw.app)


def _png(side: int = 32) -> bytes:
    """Валидное изображение достаточного размера, чтобы дойти до пайплайна."""
    buffer = io.BytesIO()
    Image.new("RGB", (side, side), (200, 30, 40)).save(buffer, format="PNG")
    return buffer.getvalue()


def _broken_jpeg() -> bytes:
    """Файл с валидной JPEG-подписью, но повреждённым телом (частый случай обрыва загрузки)."""
    return b"\xff\xd8\xff\xe0" + b"garbage-truncated-body"


# --- плоский контракт ----------------------------------------------------------

def test_eval_predict_flat_contract_only_slug(monkeypatch) -> None:
    """На валидном изображении ответ — объект ровно с ключом slug."""
    async def fake_scan(*args, **kwargs):
        return type("R", (), {"slug": "lesnaya-proseka"})()

    monkeypatch.setattr(gw, "run_scan", fake_scan)
    response = client.post("/v1/eval/predict", files={"image": ("q.png", _png(), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"slug"}
    assert body["slug"] == "lesnaya-proseka"


def test_eval_predict_empty_file_is_null_200() -> None:
    response = client.post("/v1/eval/predict", files={"image": ("e.jpg", b"", "image/jpeg")})
    assert response.status_code == 200
    assert response.json() == {"slug": None}


def test_eval_predict_non_image_is_null_200() -> None:
    response = client.post("/v1/eval/predict",
                           files={"image": ("x.png", b"definitely not an image", "image/png")})
    assert response.status_code == 200
    assert response.json() == {"slug": None}


def test_eval_predict_broken_image_is_null_200() -> None:
    """Битый файл раньше давал 503 через retrieval; теперь это плоский null."""
    response = client.post("/v1/eval/predict",
                           files={"image": ("broken.jpg", _broken_jpeg(), "image/jpeg")})
    assert response.status_code == 200
    assert response.json() == {"slug": None}


def test_eval_predict_internal_error_is_null_200(monkeypatch) -> None:
    async def broken(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(gw, "run_scan", broken)
    response = client.post("/v1/eval/predict", files={"image": ("q.png", _png(), "image/png")})
    assert response.status_code == 200
    assert response.json() == {"slug": None}


def test_eval_predict_retrieval_unavailable_is_503(monkeypatch) -> None:
    """Недоступность инфраструктуры остаётся явным 503 — это не плохой вход."""
    async def unavailable(*args, **kwargs):
        raise RuntimeError("retrieval service unavailable")

    monkeypatch.setattr(gw, "run_scan", unavailable)
    response = client.post("/v1/eval/predict", files={"image": ("q.png", _png(), "image/png")})
    assert response.status_code == 503


def test_eval_response_schema_has_only_slug() -> None:
    assert set(EvalPredictResponse.model_fields.keys()) == {"slug"}


# --- метрики прогона организатора ---------------------------------------------

def test_participant_summarize_metrics() -> None:
    rows = [
        {"predicted_slug": "a", "latency_ms": 100},
        {"predicted_slug": "b", "latency_ms": 200},
        {"predicted_slug": None, "latency_ms": 300},
        {"predicted_slug": "d", "latency_ms": 400},
        {"predicted_slug": "e", "latency_ms": 10000},
    ]
    summary = participant.summarize(list(rows), 0, "", None)
    assert summary["processed"] == 5
    assert summary["non_null"] == 4
    assert summary["non_null_ratio"] == 0.8
    assert summary["latency_ms"] == {"p50": 300.0, "p95": 8080.0, "max": 10000.0}
    assert summary["timeouts_at_organizer_cap"] == 1
    assert summary["within_limit"] == 4


def test_participant_http_errors_from_gateway_log(tmp_path: Path) -> None:
    log = tmp_path / "gw.log"
    log.write_text(
        '127.0.0.1 - "POST /v1/eval/predict HTTP/1.1" 200 OK\n'
        '127.0.0.1 - "POST /v1/eval/predict HTTP/1.1" 200 OK\n'
        '127.0.0.1 - "POST /v1/eval/predict HTTP/1.1" 503 Service Unavailable\n'
        '127.0.0.1 - "GET /v1/health HTTP/1.1" 200 OK\n',
        encoding="utf-8",
    )
    summary = participant.summarize(
        [{"predicted_slug": "a", "latency_ms": 120}], 0, "", log)
    assert summary["gateway_status"] == {"200": 2, "503": 1}
    assert summary["http_errors"] == 1


def test_participant_markdown_reports_compatibility(tmp_path: Path) -> None:
    log = tmp_path / "gw.log"
    log.write_text('127.0.0.1 - "POST /v1/eval/predict HTTP/1.1" 200 OK\n', encoding="utf-8")
    rows = [{"predicted_slug": "a", "latency_ms": 900, "image_sha256": "x"}]
    summary = participant.summarize(rows, 0, "", log)
    md = participant.render_markdown(summary, Path("reports/participant_queries.tsv"),
                                     "http://127.0.0.1:8080/v1/eval/predict")
    assert "СОВМЕСТИМ" in md
    assert "90" in md or "900" in md
    assert json.dumps(summary)  # summary сериализуем
