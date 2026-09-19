# START_MODULE_CONTRACT
# PURPOSE: FastAPI-сервис VLM: POST /extract собирает letterbox-базу и P&S-тайлы полного кадра и отдаёт поля
# SCOPE: app, health, extract_endpoint
# DEPENDS: fastapi, packages.common.config, packages.common.images, packages.common.logging,
#          packages.common.preprocess, services.normalize.app.tta, services.vlm.app.extract
# LINKS: M-VLM, M-NORMALIZE
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# app - FastAPI-приложение VLM-сервиса
# health - GET /health с параметрами препроцессинга и провайдера
# extract_endpoint - POST /extract (multipart image) → поля этикетки с полного кадра
# END_MODULE_MAP
"""Сервис VLM для полного кадра.

Этикетка не обрезается: сервис получает полный кадр, строит letterbox-базу и
pan-and-scan тайлы по всей площади, и отправляет их одним запросом. Число тайлов
зависит от неуверенности (PAS_TILES против PAS_TILES_UNCERTAIN).
"""
from __future__ import annotations

import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from packages.common.config import settings
from packages.common.images import load_bgr
from packages.common.logging import log_marker
from packages.common.preprocess import PreprocessConfig
from services.normalize.app.tta import build_vlm_inputs
from services.vlm.app.extract import extract_fields

app = FastAPI(title="Svoe Vino Scanner — VLM", version="0.2.0")

__all__ = ["app", "health", "extract_endpoint"]


@app.get("/health")
def health() -> dict:
    """Проба живости с параметрами полного кадра."""
    return {
        "status": "ok",
        "service": "vlm",
        "role": "vision_model",
        "online_model": settings.vlm_online_model,
        "full_frame": True,
        "pas_tiles": settings.pas_tiles,
        "pas_tiles_uncertain": settings.pas_tiles_uncertain,
        "num_ctx": settings.vlm_num_ctx,
        "max_tokens": settings.vlm_max_tokens,
        "timeout_s": settings.vlm_timeout_s,
        "fallback_enabled": settings.vlm_fallback_enabled,
    }


@app.post("/extract")
async def extract_endpoint(image: UploadFile = File(...), uncertain: bool = False) -> dict:
    """Читает поля этикетки с полного кадра; при сбое отдаёт 200 с ok=false."""
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty image")
    try:
        array = load_bgr(data)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    tiles = settings.pas_tiles_uncertain if uncertain else settings.pas_tiles
    config = PreprocessConfig(pas_tiles=tiles, enhance=settings.normalize_enhance, crop=False)
    images, kinds = build_vlm_inputs(array, config)

    started = time.perf_counter()
    fields = await run_in_threadpool(
        extract_fields,
        images,
        None,
        model=settings.vlm_online_model or None,
        max_tokens=settings.vlm_max_tokens,
        timeout_s=settings.vlm_timeout_s,
        num_ctx=settings.vlm_num_ctx,
        reasoning_effort=settings.vlm_reasoning_effort,
    )
    total_ms = (time.perf_counter() - started) * 1000.0
    log_marker("M-VLM", "extract", "DONE", ok=str(fields.ok), images=len(images),
               tiles=tiles, latency_ms=round(total_ms, 1))
    return {
        "ok": fields.ok,
        "reason": fields.reason,
        "provider": fields.provider,
        "model": fields.model,
        "finish_reason": fields.finish_reason,
        "images": len(images),
        "image_kinds": kinds,
        "filled_fields": fields.filled(),
        "vlm_latency_ms": round(fields.latency_ms, 1),
        "total_latency_ms": round(total_ms, 1),
        "fields": fields.as_dict(),
    }
