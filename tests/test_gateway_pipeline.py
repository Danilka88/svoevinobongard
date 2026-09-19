# START_MODULE_CONTRACT
# PURPOSE: Верификация пайплайна gateway: rich-ответ, плоский контракт организатора, not_found и деградация
# SCOPE: test_run_scan_rich_response, test_run_scan_degraded_when_normalize_down, test_run_scan_not_found, test_eval_predict_flat_contract, test_scan_endpoint_rich, test_retrieval_unavailable
# DEPENDS: fastapi, packages.common.catalog, packages.contracts.schemas, services.gateway.app.pipeline, services.gateway.app.clients
# LINKS: M-GATEWAY
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# test_run_scan_rich_response - пайплайн собирает ScanResponse с top5, margin и карточкой
# test_run_scan_degraded_when_normalize_down - без normalize используется исходное изображение
# test_run_scan_not_found - низкий скор помечается not_found
# test_eval_predict_flat_contract - эндпоинт отдает ровно {"slug": "..."}
# test_scan_endpoint_rich - эндпоинт отдает rich-ответ контракта
# test_calibrated_confidence_monotonic - уверенность растет с отрывом и корректна на краях
# test_ocr_branch_uses_matcher - неуверенный Tier-1 уходит в OCR и matcher с path=ocr_fallback
# test_retrieval_unavailable - недоступный retrieval дает HTTP 503, а не 5xx-падение
# END_MODULE_MAP
"""V-M-GATEWAY: оркестрация пайплайна и контракт организатора."""
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
from services.gateway.app.clients import ServiceClients
from services.gateway.app.pipeline import run_scan

__all__ = [
    "test_run_scan_rich_response",
    "test_run_scan_degraded_when_normalize_down",
    "test_run_scan_not_found",
    "test_eval_predict_flat_contract",
    "test_scan_endpoint_rich",
    "test_retrieval_unavailable",
]

pytest.importorskip("multipart", reason="python-multipart required by FastAPI File uploads")

_CARD_RECORD = {
    "slug": "kokur-suhoe-2025",
    "name": "Кокур Сухое, 2025",
    "winery": "Валерий Захарьин",
    "region": "Крым",
    "category": "Белое",
    "color": "белое",
    "grape": "Кокур",
    "description": "Сухое белое вино",
    "year": 2025,
    "sweetness": "сухое",
    "image": "DSC_09173_4a9ff95cc2.webp",
    "has_image": True,
}

_DEFAULT_TOP_K = [
    {"slug": "kokur-suhoe-2025", "score": 0.81, "name": "Кокур Сухое, 2025",
     "winery": "Валерий Захарьин", "region": "Крым", "year": 2025},
    {"slug": "kokur-2025", "score": 0.52, "name": "Кокур, 2025",
     "winery": "Валерий Захарьин", "region": "Крым", "year": 2025},
]


def _store(tmp_path) -> CatalogStore:
    """Каталог из двух записей."""
    catalog = tmp_path / "catalog.jsonl"
    second = dict(_CARD_RECORD, slug="kokur-2025", name="Кокур, 2025")
    catalog.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in (_CARD_RECORD, second)),
        encoding="utf-8",
    )
    return CatalogStore.from_jsonl(catalog)


def _png_bytes() -> bytes:
    """Небольшой PNG для multipart."""
    return encode_png(np.full((32, 32, 3), 200, dtype=np.uint8))


def _patch_clients(monkeypatch, top_k: list[dict] | None = None, normalize_ok: bool = True) -> None:
    """Подменяет HTTP-клиенты сервисов детерминированными ответами."""
    async def fake_normalize(self, data: bytes, filename: str = "image.jpg"):
        if not normalize_ok:
            return None
        return {
            "method": "contour_quad",
            "count": 1,
            "metrics": {"detection": "contour_quad", "variants": 1},
            "crops": [{"index": 0, "width": 32, "height": 32,
                       "png_base64": base64.b64encode(_png_bytes()).decode("ascii")}],
        }

    async def fake_search(self, data: bytes, filename: str = "image.jpg"):
        return {"encoder": "test", "model": "test", "device": "cpu", "latency_ms": 1.0,
                "top_k": top_k if top_k is not None else _DEFAULT_TOP_K}

    monkeypatch.setattr(ServiceClients, "normalize", fake_normalize)
    monkeypatch.setattr(ServiceClients, "search", fake_search)


def test_run_scan_rich_response(tmp_path, monkeypatch) -> None:
    _patch_clients(monkeypatch)
    result = asyncio.run(run_scan(_png_bytes(), _store(tmp_path)))
    assert result.slug == "kokur-suhoe-2025"
    assert result.path == "retrieval"
    # confidence — калиброванный softmax по top-K (C-003), а не сырой косинус
    assert 0.0 < result.confidence <= 1.0
    assert result.f1_top1 == pytest.approx(0.81, abs=1e-6)
    assert result.margin == pytest.approx(0.29, abs=1e-6)
    assert len(result.top5) == 2
    assert result.not_found is False
    assert result.card is not None and result.card.region == "Крым"
    assert result.latency_ms >= 0.0


def test_run_scan_degraded_when_normalize_down(tmp_path, monkeypatch) -> None:
    _patch_clients(monkeypatch, normalize_ok=False)
    result = asyncio.run(run_scan(_png_bytes(), _store(tmp_path)))
    assert result.slug == "kokur-suhoe-2025"


def test_run_scan_not_found(tmp_path, monkeypatch) -> None:
    _patch_clients(monkeypatch, top_k=[{"slug": "low", "score": 0.10, "name": "x", "winery": "y"}])
    result = asyncio.run(run_scan(_png_bytes(), _store(tmp_path)))
    assert result.not_found is True
    assert result.slug == "low"


def test_eval_predict_flat_contract(tmp_path, monkeypatch) -> None:
    _patch_clients(monkeypatch)
    monkeypatch.setattr(gateway_main, "_store", _store(tmp_path))
    client = TestClient(gateway_main.app)
    response = client.post("/v1/eval/predict", files={"image": ("q.png", _png_bytes(), "image/png")})
    assert response.status_code == 200
    assert response.json() == {"slug": "kokur-suhoe-2025"}


def test_scan_endpoint_rich(tmp_path, monkeypatch) -> None:
    _patch_clients(monkeypatch)
    monkeypatch.setattr(gateway_main, "_store", _store(tmp_path))
    client = TestClient(gateway_main.app)
    response = client.post("/v1/scan", files={"image": ("q.png", _png_bytes(), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["slug"] == "kokur-suhoe-2025"
    assert body["path"] == "retrieval"
    assert len(body["top5"]) == 2
    assert body["card"]["winery"] == "Валерий Захарьин"


def test_calibrated_confidence_monotonic() -> None:
    """Калиброванная уверенность выше при большем отрыве и корректна на краях."""
    from services.gateway.app.pipeline import calibrated_confidence

    assert calibrated_confidence([]) == 0.0
    assert calibrated_confidence([0.9, 0.1]) > calibrated_confidence([0.6, 0.55])
    assert 0.0 < calibrated_confidence([0.9, 0.1], temperature=0.2) <= 1.0
    # при одинаковых скорах вероятность делится поровну
    assert abs(calibrated_confidence([0.5, 0.5], temperature=0.2) - 0.5) < 1e-9


def test_ocr_branch_uses_matcher(tmp_path, monkeypatch) -> None:
    """Неуверенный Tier-1 → OCR-поля → matcher → path=ocr_fallback."""
    async def fake_search(self, data: bytes, filename: str = "image.jpg"):
        return {"encoder": "test", "model": "test", "device": "cpu", "latency_ms": 1.0,
                "top_k": [{"slug": "kokur-2025", "score": 0.40, "name": "Кокур, 2025",
                           "winery": "Валерий Захарьин"},
                          {"slug": "kokur-suhoe-2025", "score": 0.39,
                           "name": "Кокур Сухое, 2025", "winery": "Валерий Захарьин"}]}

    async def fake_ocr(self, data: bytes, filename: str = "query.png"):
        return {"ok": True, "engine": "stub", "lines": ["КОКУР СУХОЕ 2025"],
                "fields": {"winery": "Валерий Захарьин", "vintage": "2025",
                           "wine_type": "сухое"}, "ocr_latency_ms": 12.0}

    async def fake_rerank(self, candidates: list[dict], fields: dict):
        return {"items": [{"slug": "kokur-suhoe-2025", "score": 0.95,
                           "name": "Кокур Сухое, 2025", "winery": "Валерий Захарьин"},
                          {"slug": "kokur-2025", "score": 0.50, "name": "Кокур, 2025",
                           "winery": "Валерий Захарьин"}]}

    monkeypatch.setattr(ServiceClients, "search", fake_search)
    monkeypatch.setattr(ServiceClients, "ocr", fake_ocr)
    monkeypatch.setattr(ServiceClients, "rerank", fake_rerank)

    result = asyncio.run(run_scan(_png_bytes(), _store(tmp_path)))
    assert result.path == "ocr_fallback"
    assert result.ocr_used is True
    assert result.slug == "kokur-suhoe-2025"
    assert result.ocr_latency_ms >= 0.0


def test_retrieval_unavailable(tmp_path, monkeypatch) -> None:
    async def broken_search(self, data: bytes, filename: str = "image.jpg"):
        return None

    monkeypatch.setattr(ServiceClients, "search", broken_search)
    monkeypatch.setattr(gateway_main, "_store", _store(tmp_path))
    client = TestClient(gateway_main.app)
    response = client.post("/v1/eval/predict", files={"image": ("q.png", _png_bytes(), "image/png")})
    assert response.status_code == 503
    assert "detail" in response.json()
