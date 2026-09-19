# START_MODULE_CONTRACT
# PURPOSE: FastAPI-сервис OCR: POST /ocr отдает строки и нормализованные поля (год, сладость)
# SCOPE: app, health, ocr_endpoint
# DEPENDS: fastapi, packages.common.config, packages.common.images, packages.common.logging, packages.common.ocr
# LINKS: M-OCR
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# app - FastAPI-приложение OCR-сервиса
# health - GET /health с выбранным движком и бюджетом времени
# ocr_endpoint - POST /ocr (multipart image) со строками и полями
# END_MODULE_MAP
"""Сервис OCR-сигнала.

Дешёвая альтернатива VLM-fallback: сотни миллисекунд на кроп, детерминированный
результат, никакой генерации. Провал движка не является ошибкой связи — сервис
отвечает 200 с ``ok=false``, а вызывающий код деградирует к Tier-1.
"""
from __future__ import annotations

import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from packages.common.config import settings
from packages.common.images import load_bgr
from packages.common.logging import log_marker
from packages.common.ocr import resolve_engine, run_ocr

app = FastAPI(title="Svoe Vino Scanner — OCR", version="0.1.0")

__all__ = ["app", "health", "ocr_endpoint"]


@app.get("/health")
def health() -> dict:
    """Проба живости сервиса OCR."""
    return {"status": "ok", "service": "ocr", "engine": resolve_engine(),
            "max_side": settings.ocr_max_side, "budget_ms": int(settings.ocr_timeout_s * 1000)}


@app.post("/ocr")
async def ocr_endpoint(image: UploadFile = File(...)) -> dict:
    """Распознает текст и возвращает нормализованные поля этикетки."""
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty image")
    try:
        array = load_bgr(data)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    started = time.perf_counter()
    result = await run_in_threadpool(run_ocr, array, settings.ocr_max_side)
    total_ms = (time.perf_counter() - started) * 1000.0
    log_marker("M-OCR", "ocr", "RESPONSE", ok=str(result.ok), engine=result.engine,
               latency_ms=round(total_ms, 1))
    return {
        "ok": result.ok,
        "reason": result.reason,
        "engine": result.engine,
        "lines": result.lines,
        "fields": result.as_fields(),
        "ocr_latency_ms": result.latency_ms,
        "total_latency_ms": round(total_ms, 1),
    }
