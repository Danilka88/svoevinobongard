# START_MODULE_CONTRACT
# PURPOSE: FastAPI-сервис VLM-fallback: POST /extract отдает строгие JSON-поля этикетки
# SCOPE: app, health, extract_endpoint
# DEPENDS: fastapi, numpy, packages.common.config, packages.common.images, packages.common.logging,
#          services.vlm.app.extract
# LINKS: M-VLM
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# app - FastAPI-приложение VLM-сервиса
# health - GET /health со статусом провайдера и модели
# extract_endpoint - POST /extract (multipart image) со строгими полями этикетки
# END_MODULE_MAP
"""Сервис VLM-извлечения полей этикетки.

Онлайн-путь использует быструю модель с отключенными рассуждениями и жёстким
таймаутом; тяжёлая модель (gemma4:12b) остаётся для офлайн-обогащения. При сбое
сервис отвечает 200 с ``ok=false``: это контракт деградации, а не ошибка связи.
"""
from __future__ import annotations

import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from packages.common.config import settings
from packages.common.images import load_bgr
from packages.common.logging import log_marker
from services.vlm.app.extract import extract_fields

app = FastAPI(title="Svoe Vino Scanner — VLM", version="0.1.0")

__all__ = ["app", "health", "extract_endpoint"]


@app.get("/health")
def health() -> dict:
    """Проба живости сервиса VLM."""
    return {
        "status": "ok",
        "service": "vlm",
        "role": "vision_model",
        "online_model": settings.vlm_online_model,
        "max_tokens": settings.vlm_max_tokens,
        "timeout_s": settings.vlm_timeout_s,
        "fallback_enabled": settings.vlm_fallback_enabled,
    }


@app.post("/extract")
async def extract_endpoint(image: UploadFile = File(...)) -> dict:
    """Извлекает поля этикетки; при сбое VLM отвечает 200 с ok=false (деградация)."""
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty image")
    try:
        array = load_bgr(data)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    started = time.perf_counter()
    fields = await run_in_threadpool(
        extract_fields,
        array,
        None,
        model=settings.vlm_online_model or None,
        max_tokens=settings.vlm_max_tokens,
        timeout_s=settings.vlm_timeout_s,
        extra={"reasoning_effort": settings.vlm_reasoning_effort},
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    log_marker("M-VLM", "extract", "DONE", ok=str(fields.ok), latency_ms=round(latency_ms, 1))
    return {
        "ok": fields.ok,
        "reason": fields.reason,
        "provider": fields.provider,
        "model": fields.model,
        "finish_reason": fields.finish_reason,
        "vlm_latency_ms": round(fields.latency_ms, 1),
        "total_latency_ms": round(latency_ms, 1),
        "fields": fields.as_dict(),
    }
