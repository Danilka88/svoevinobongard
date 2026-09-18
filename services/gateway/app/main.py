# START_MODULE_CONTRACT
# PURPOSE: Публичный API gateway — health, контракт организатора /v1/eval/predict и rich-ответ /v1/scan
# SCOPE: FastAPI app, /v1/health, /v1/eval/predict, /v1/scan
# DEPENDS: M-CONTRACTS
# LINKS: M-GATEWAY, M-CONTRACTS
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# app - FastAPI-приложение gateway
# VLM_MODEL - имя VLM из окружения
# ENCODER - имя image-энкодера из окружения
# health - GET /v1/health
# eval_predict - POST /v1/eval/predict для скрипта организатора
# scan - POST /v1/scan с rich-ответом
# END_MODULE_MAP
"""Gateway skeleton (M0).

Public API entrypoint for the wine scanner. Real pipeline wiring
(normalize → retrieval → vlm → matcher) is added in M1+.
"""
from __future__ import annotations

import sys
import os
from fastapi import FastAPI, File, UploadFile, HTTPException

# START_BLOCK_GATEWAY_INIT
app = FastAPI(title="Svoe Vino Scanner — Gateway", version="0.1.0")

VLM_MODEL = os.getenv("VLM_MODEL", "gemma4:12b")
ENCODER = os.getenv("ENCODER", "siglip2")
# END_BLOCK_GATEWAY_INIT


# START_BLOCK_GATEWAY_HEALTH
@app.get("/v1/health")
def health() -> dict:
    print("[M-GATEWAY][HEALTH][OK]", file=sys.stderr)
    return {
        "status": "ok",
        "service": "gateway",
        "stage": "M0-skeleton",
        "models": {"retrieval": ENCODER, "vlm": VLM_MODEL},
    }
# END_BLOCK_GATEWAY_HEALTH


# START_BLOCK_GATEWAY_EVAL
@app.post("/v1/eval/predict")
async def eval_predict(image: UploadFile = File(...)) -> dict:
    """Organizer eval contract. Pipeline not wired yet (M0)."""
    print("[M-GATEWAY][PIPELINE][NOT_IMPLEMENTED] endpoint=/v1/eval/predict", file=sys.stderr)
    raise HTTPException(status_code=501, detail="pipeline not implemented (M0 scaffold)")
# END_BLOCK_GATEWAY_EVAL


# START_BLOCK_GATEWAY_SCAN
@app.post("/v1/scan")
async def scan(image: UploadFile = File(...)) -> dict:
    """Rich scan response. Pipeline not wired yet (M0)."""
    print("[M-GATEWAY][PIPELINE][NOT_IMPLEMENTED] endpoint=/v1/scan", file=sys.stderr)
    raise HTTPException(status_code=501, detail="pipeline not implemented (M0 scaffold)")
# END_BLOCK_GATEWAY_SCAN
