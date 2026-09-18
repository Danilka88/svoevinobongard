# START_MODULE_CONTRACT
# PURPOSE: Публичный API gateway — health, контракт организатора /v1/eval/predict и rich-ответ /v1/scan
# SCOPE: app, get_store, health, eval_predict, scan
# DEPENDS: fastapi, packages.common.catalog, packages.contracts.schemas, services.gateway.app.pipeline
# LINKS: M-GATEWAY, M-CONTRACTS
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# app - FastAPI-приложение gateway
# VLM_MODEL - имя VLM из окружения
# ENCODER - имя image-энкодера из окружения
# get_store - ленивая загрузка read-only каталога
# health - GET /v1/health
# eval_predict - POST /v1/eval/predict для скрипта организатора
# scan - POST /v1/scan с rich-ответом
# END_MODULE_MAP
"""Gateway — единственная публичная точка входа.

Контракт организатора: плоский ``{"slug": "..."}`` на верхнем уровне.
Rich-ответ: slug, confidence, f1_top1, f1_top5, margin, path, latency_ms, top5, card.
Внутренние сервисы наружу не публикуются.
"""
from __future__ import annotations

import os

from fastapi import FastAPI, File, HTTPException, UploadFile

from packages.common.catalog import CatalogStore
from packages.common.config import settings
from packages.common.logging import log_marker
from packages.contracts.schemas import EvalPredictResponse, ScanResponse
from services.gateway.app.pipeline import run_scan

__all__ = [
    "app",
    "VLM_MODEL",
    "ENCODER",
    "get_store",
    "health",
    "eval_predict",
    "scan",
]

app = FastAPI(title="Svoe Vino Scanner — Gateway", version="0.2.0")

VLM_MODEL = os.getenv("VLM_MODEL", "gemma4:12b")
ENCODER = os.getenv("ENCODER", "siglip2")

_store: CatalogStore | None = None


def get_store() -> CatalogStore:
    """Ленивая загрузка read-only каталога.

    Raises:
        HTTPException: если файл каталога не найден.
    """
    global _store
    if _store is None:
        try:
            _store = CatalogStore.from_jsonl(settings.catalog_jsonl)
        except FileNotFoundError as error:
            log_marker("M-GATEWAY", "get_store", "CATALOG_MISSING", path=settings.catalog_jsonl)
            raise HTTPException(status_code=503, detail=str(error)) from error
        log_marker("M-GATEWAY", "get_store", "READY", records=len(_store))
    return _store


@app.get("/v1/health")
def health() -> dict:
    """Проба живости gateway."""
    log_marker("M-GATEWAY", "health", "OK")
    return {
        "status": "ok",
        "service": "gateway",
        "stage": "M1-tier1",
        "models": {"retrieval": ENCODER, "vlm": VLM_MODEL},
        "endpoints": ["/v1/health", "/v1/eval/predict", "/v1/scan"],
    }


@app.post("/v1/eval/predict")
async def eval_predict(image: UploadFile = File(...)) -> EvalPredictResponse:
    """Контракт организатора: image → плоский slug."""
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty image")
    try:
        result = await run_scan(data, get_store(), filename=image.filename or "query.jpg")
    except RuntimeError as error:
        log_marker("M-GATEWAY", "eval_predict", "PIPELINE_UNAVAILABLE", error=str(error))
        raise HTTPException(status_code=503, detail=str(error)) from error
    return EvalPredictResponse(slug=result.slug)


@app.post("/v1/scan")
async def scan(image: UploadFile = File(...)) -> ScanResponse:
    """Rich-ответ основного сценария сканирования."""
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty image")
    try:
        return await run_scan(data, get_store(), filename=image.filename or "query.jpg")
    except RuntimeError as error:
        log_marker("M-GATEWAY", "scan", "PIPELINE_UNAVAILABLE", error=str(error))
        raise HTTPException(status_code=503, detail=str(error)) from error
