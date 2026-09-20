# START_MODULE_CONTRACT
# PURPOSE: Верификация gateway: /v1/health отвечает 200, эндпоинты требуют изображение
# SCOPE: test_health_ok, test_scan_requires_image, test_health_lists_endpoints
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
# test_health_ok - GET /v1/health отвечает 200 и отдает модели
# test_health_lists_endpoints - health перечисляет публичные эндпоинты
# test_scan_requires_image - POST /v1/scan без файла дает 422
# END_MODULE_MAP
"""V-M-GATEWAY: health и валидация входа."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

__all__ = [
    "ROOT",
    "MAIN_PY",
    "app_module",
    "client",
    "test_health_ok",
    "test_health_lists_endpoints",
    "test_scan_requires_image",
]

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


def test_health_lists_endpoints() -> None:
    body = client.get("/v1/health").json()
    assert body["endpoints"] == ["/v1/health", "/v1/warmup", "/v1/eval/predict", "/v1/scan",
                                   "/v1/similar/{slug}", "/v1/sommelier", "/v1/enotourism/{slug}"]


def test_scan_requires_image() -> None:
    assert client.post("/v1/scan").status_code == 422
    assert client.post("/v1/eval/predict").status_code == 422
