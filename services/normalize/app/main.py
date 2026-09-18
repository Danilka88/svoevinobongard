# START_MODULE_CONTRACT
# PURPOSE: FastAPI-сервис нормализации: POST /normalize возвращает список нормализованных изображений
# SCOPE: normalize_bytes, health, normalize_endpoint
# DEPENDS: fastapi, numpy, cv2, services.normalize.app.exif, services.normalize.app.detect, services.normalize.app.tta
# LINKS: M-NORMALIZE
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# app - FastAPI-приложение normalize
# normalize_bytes - байты фото в список нормализованных BGR-вариантов и метод детекции
# health - GET /health
# normalize_endpoint - POST /normalize (multipart image)
# END_MODULE_MAP
"""Сервис нормализации.

Границы: сервис не импортирует другие сервисы. Вход — байты изображения,
выход — список нормализованных вариантов (TTA) плюс метод детекции.
"""
from __future__ import annotations

import base64

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile

from packages.common.images import encode_png
from packages.common.logging import log_marker
from services.normalize.app.detect import detect_label
from services.normalize.app.exif import load_image
from services.normalize.app.tta import build_variants

__all__ = [
    "app",
    "normalize_bytes",
    "health",
    "normalize_endpoint",
]

app = FastAPI(title="Svoe Vino Scanner — Normalize", version="0.1.0")


def normalize_bytes(data: bytes) -> tuple[list, str, dict]:
    """Нормализует фото: EXIF, детекция этикетки, выпрямление, TTA.

    Args:
        data: байты входного изображения.

    Returns:
        Кортеж (варианты BGR, метод детекции, метрики).

    Raises:
        ValueError: если изображение не декодируется.
    """
    image = load_image(data)
    detection = detect_label(image)
    variants = build_variants(image, detection.quad)
    metrics = {
        "detection": detection.method,
        "area_ratio": round(detection.area_ratio, 4),
        "variants": len(variants),
        "source_size": [int(image.shape[1]), int(image.shape[0])],
    }
    return variants, detection.method, metrics


@app.get("/health")
def health() -> dict:
    """Проба живости сервиса."""
    log_marker("M-NORMALIZE", "health", "OK")
    return {"status": "ok", "service": "normalize"}


@app.post("/normalize")
async def normalize_endpoint(image: UploadFile = File(...)) -> dict:
    """Нормализует загруженное фото и возвращает TTA-варианты в base64 PNG."""
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty image")
    try:
        variants, method, metrics = normalize_bytes(data)
    except ValueError as error:
        log_marker("M-NORMALIZE", "normalize", "DECODE_FAILED", error=str(error))
        raise HTTPException(status_code=400, detail=str(error)) from error

    log_marker("M-NORMALIZE", "normalize", "CROPS", method=method, variants=len(variants))
    crops = []
    for index, variant in enumerate(variants):
        height, width = variant.shape[:2]
        crops.append({
            "index": index,
            "width": int(width),
            "height": int(height),
            "png_base64": base64.b64encode(encode_png(variant)).decode("ascii"),
        })
    return {"method": method, "count": len(crops), "metrics": metrics, "crops": crops}
