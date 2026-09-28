# START_MODULE_CONTRACT
# PURPOSE: Переключатель движка сканирования: classic (только Tier-1) против LLM-ядра
# SCOPE: test_classic_mode_skips_llm_core, test_llm_mode_runs_full_core,
#        test_scan_endpoint_mode_classic_skips_observer, test_scan_endpoint_mode_llm_runs_observer
# DEPENDS: asyncio, base64, json, numpy, pytest, fastapi, packages.common.catalog,
#          services.gateway.app.clients, services.gateway.app.main, services.gateway.app.pipeline
# LINKS: M-GATEWAY
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# test_classic_mode_skips_llm_core - llm_mode=False не вызывает observe/normalize_reading/match
# test_llm_mode_runs_full_core - llm_mode=True проходит чтение, нормализацию и выбор → path=llm_match
# test_scan_endpoint_mode_classic_skips_observer - mode=classic в /v1/scan не вызывает observer
# test_scan_endpoint_mode_llm_runs_observer - mode=llm в /v1/scan вызывает observer
# END_MODULE_MAP
"""C-096: режим сканирования выбирается параметром mode (llm | classic).

Проверяем, что классический режим не трогает LLM-сервисы вовсе, а режим LLM проходит
полный контур чтение → нормализация → выбор. Это машинный контракт переключателя в UI.
"""
from __future__ import annotations

import asyncio
import base64
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from packages.common.catalog import CatalogStore
from packages.common.images import encode_png
from services.gateway.app import main as gateway_main
from services.gateway.app import pipeline as pipeline_module
from services.gateway.app.clients import ServiceClients
from services.gateway.app.pipeline import run_scan

__all__ = [
    "test_classic_mode_skips_llm_core",
    "test_llm_mode_runs_full_core",
    "test_scan_endpoint_mode_classic_skips_observer",
    "test_scan_endpoint_mode_llm_runs_observer",
]

pytest.importorskip("multipart", reason="python-multipart required by FastAPI File uploads")

_RECORD = {
    "slug": "kokur-suhoe-2025",
    "name": "Кокур Сухое, 2025",
    "winery": "Валерий Захарьин",
    "region": "Крым",
    "category": "Белое",
    "color": "белое",
    "grape": "Кокур",
    "year": 2025,
    "sweetness": "сухое",
}
_TOP_K = [
    {"slug": "kokur-suhoe-2025", "score": 0.81, "name": "Кокур Сухое, 2025", "winery": "Валерий Захарьин"},
    {"slug": "kokur-2025", "score": 0.52, "name": "Кокур, 2025", "winery": "Валерий Захарьин"},
]


def _store(tmp_path) -> CatalogStore:
    catalog = tmp_path / "catalog.jsonl"
    second = dict(_RECORD, slug="kokur-2025", name="Кокур, 2025")
    catalog.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in (_RECORD, second)),
                       encoding="utf-8")
    return CatalogStore.from_jsonl(catalog)


def _png_bytes() -> bytes:
    return encode_png(np.full((32, 32, 3), 200, dtype=np.uint8))


class _Calls:
    """Счётчик вызовов LLM-методов, чтобы проверить их отсутствие в классическом режиме."""

    def __init__(self) -> None:
        self.observe = 0
        self.normalize = 0
        self.match = 0


def _patch_clients(monkeypatch) -> _Calls:
    calls = _Calls()

    async def fake_normalize(self, data: bytes, filename: str = "image.jpg"):
        return {"method": "none", "count": 0, "crops": [], "metrics": {}}

    async def fake_search(self, data: bytes, filename: str = "image.jpg"):
        return {"encoder": "test", "model": "test", "device": "cpu", "latency_ms": 1.0, "top_k": _TOP_K}

    async def fake_observe(self, data: bytes, filename: str = "query.png", timeout=None):
        calls.observe += 1
        return {"ok": True, "raw_text": "Валерий Захарьин Кокур"}

    async def fake_normalize_reading(self, observation, timeout=None):
        calls.normalize += 1
        return {"source": "llm", "winery": "Валерий Захарьин", "wine_name": "Кокур"}

    async def fake_match(self, observation, candidates, timeout=None):
        calls.match += 1
        return {"slug": "kokur-suhoe-2025", "p_correct": 0.9, "source": "llm"}

    monkeypatch.setattr(ServiceClients, "normalize", fake_normalize)
    monkeypatch.setattr(ServiceClients, "search", fake_search)
    monkeypatch.setattr(ServiceClients, "observe", fake_observe)
    monkeypatch.setattr(ServiceClients, "normalize_reading", fake_normalize_reading)
    monkeypatch.setattr(ServiceClients, "match", fake_match)
    monkeypatch.setattr(pipeline_module, "parse_observation",
                        lambda raw: {"winery": "Валерий Захарьин", "wine_name": "Кокур"})
    return calls


def test_classic_mode_skips_llm_core(tmp_path, monkeypatch) -> None:
    calls = _patch_clients(monkeypatch)
    result = asyncio.run(run_scan(_png_bytes(), _store(tmp_path), llm_mode=False))
    assert calls.observe == 0
    assert calls.normalize == 0
    assert calls.match == 0
    assert result.path == "retrieval"
    assert result.observation == {}
    assert result.normalization == {}


def test_llm_mode_runs_full_core(tmp_path, monkeypatch) -> None:
    calls = _patch_clients(monkeypatch)
    result = asyncio.run(run_scan(_png_bytes(), _store(tmp_path), llm_mode=True))
    assert calls.observe == 1
    assert calls.match == 1
    assert result.path == "llm_match"
    assert result.slug == "kokur-suhoe-2025"
    assert result.observation.get("winery") == "Валерий Захарьин"


def test_scan_endpoint_mode_classic_skips_observer(tmp_path, monkeypatch) -> None:
    calls = _patch_clients(monkeypatch)
    monkeypatch.setattr(gateway_main, "_store", _store(tmp_path))
    client = TestClient(gateway_main.app)
    response = client.post("/v1/scan", files={"image": ("q.png", _png_bytes(), "image/png")},
                           data={"mode": "classic"})
    assert response.status_code == 200
    assert calls.observe == 0
    assert response.json()["path"] == "retrieval"


def test_scan_endpoint_mode_llm_runs_observer(tmp_path, monkeypatch) -> None:
    calls = _patch_clients(monkeypatch)
    monkeypatch.setattr(gateway_main, "_store", _store(tmp_path))
    client = TestClient(gateway_main.app)
    response = client.post("/v1/scan", files={"image": ("q.png", _png_bytes(), "image/png")},
                           data={"mode": "llm"})
    assert response.status_code == 200
    assert calls.observe == 1
    assert response.json()["path"] == "llm_match"
