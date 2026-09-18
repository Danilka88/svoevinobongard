# START_MODULE_CONTRACT
# PURPOSE: Верификация gateway — /v1/health отвечает 200, pipeline-эндпоинты осознанно 501 в M0
# SCOPE: test_health_ok, test_pipeline_not_implemented_in_m0
# DEPENDS: M-GATEWAY (services/gateway/app/main.py), M-CONTRACTS
# LINKS: M-GATEWAY
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# ROOT - корень проекта
# MAIN_PY - путь к FastAPI-приложению gateway
# app_module - загруженный модуль gateway
# client - TestClient поверх приложения
# test_health_ok - GET /v1/health отвечает 200
# test_pipeline_not_implemented_in_m0 - pipeline-эндпоинты отдают 501
# END_MODULE_MAP
"""V-M-003: gateway-контракт.

Загружает FastAPI-приложение напрямую из файла и проверяет реальные HTTP-ответы
через TestClient (без запуска uvicorn).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# FastAPI требует python-multipart для File(...) (Form data). Он заявлен в
# services/gateway/requirements.txt; в голом окружении — корректный skip, не падение.
pytest.importorskip("multipart", reason="python-multipart не установлен (см. services/gateway/requirements.txt)")

ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = ROOT / "services" / "gateway" / "app" / "main.py"


def _load_gateway_app():
    spec = importlib.util.spec_from_file_location("svoe_gateway_main", MAIN_PY)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


app_module = _load_gateway_app()
client = TestClient(app_module.app)


def test_health_ok() -> None:
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "gateway"
    assert set(body["models"]) == {"retrieval", "vlm"}


def test_pipeline_not_implemented_in_m0() -> None:
    for endpoint in ("/v1/eval/predict", "/v1/scan"):
        resp = client.post(endpoint, files={"image": ("q.png", b"\x89PNG\r\n", "image/png")})
        assert resp.status_code == 501, f"{endpoint} должен возвращать 501 в M0"
        assert "detail" in resp.json(), f"{endpoint} должен отдавать контракт ошибки {{detail}}"
