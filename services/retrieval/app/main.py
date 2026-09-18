# START_MODULE_CONTRACT
# PURPOSE: FastAPI-сервис Tier-1 retrieval: POST /search (image) → Top-K slug со скорами и метаданными
# SCOPE: read_index_meta, Runtime, get_meta, get_runtime, health, search_endpoint
# DEPENDS: fastapi, numpy, packages.common.catalog, services.retrieval.app.encoder, services.retrieval.app.index, services.retrieval.app.search
# LINKS: M-RETRIEVAL
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# app - FastAPI-приложение retrieval
# read_index_meta - чтение meta.json индекса (авторитетная модель энкодера)
# Runtime - ленивый контейнер энкодера, индекса, каталога и метаданных
# get_meta - кэшированные метаданные индекса
# get_runtime - ленивая инициализация Runtime
# health - GET /health со статусом индекса
# search_endpoint - POST /search (multipart image)
# END_MODULE_MAP
"""Сервис retrieval (Tier-1).

Границы: сервис не импортирует другие сервисы. Индекс читается из
``INDEX_DIR`` (data/index), метаданные — из ``CATALOG_JSONL``. Модель и индекс
загружаются лениво при первом запросе.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from packages.common.config import settings
from packages.common.images import load_bgr
from packages.common.logging import log_marker
from packages.common.catalog import CatalogStore
from services.retrieval.app.encoder import ImageEncoder, load_encoder
from services.retrieval.app.head import describe_head, load_head, read_head_meta
from services.retrieval.app.index import VectorIndex, load_index
from services.retrieval.app.search import search_images

__all__ = [
    "app",
    "read_index_meta",
    "Runtime",
    "get_meta",
    "get_runtime",
    "health",
    "search_endpoint",
]

app = FastAPI(title="Svoe Vino Scanner — Retrieval", version="0.1.0")


def read_index_meta(directory: str | Path) -> dict:
    """Читает meta.json индекса (энкодер и модель — авторитет для запросов).

    Запросный энкодер ОБЯЗАН совпадать с тем, которым построен индекс, иначе
    косинусная близость между запросом и каталогом теряет смысл. Поэтому модель
    берётся из meta.json, а не из ENCODER/ENCODER_MODEL окружения.
    """
    meta_path = Path(directory) / "meta.json"
    if not meta_path.is_file():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except ValueError:
        return {}


@dataclass
class Runtime:
    """Лениво загруженные зависимости сервиса."""

    encoder: ImageEncoder
    index: VectorIndex
    store: CatalogStore
    meta: dict
    head: object | None = None


_runtime: Runtime | None = None
_meta_cache: dict | None = None


def get_meta() -> dict:
    """Кэшированные метаданные индекса."""
    global _meta_cache
    if _meta_cache is None:
        _meta_cache = read_index_meta(settings.index_dir)
    return _meta_cache


def get_runtime() -> Runtime:
    """Загружает энкодер, индекс и каталог при первом обращении.

    Raises:
        HTTPException: если индекс не собран.
    """
    global _runtime
    if _runtime is not None:
        return _runtime
    meta = get_meta()
    try:
        index = load_index(settings.index_dir, backend=settings.vector_backend)
    except FileNotFoundError as error:
        log_marker("M-RETRIEVAL", "get_runtime", "INDEX_MISSING", dir=settings.index_dir)
        raise HTTPException(
            status_code=503,
            detail=f"index not built: {error}. Run scripts/catalog/build_index.py",
        ) from error
    model_id = meta.get("model") or None
    if model_id and meta.get("encoder") and meta["encoder"] != settings.encoder:
        log_marker("M-RETRIEVAL", "get_runtime", "ENCODER_OVERRIDE",
                   env=settings.encoder, index=meta["encoder"], model=model_id)
    head = None
    head_payload = read_head_meta(meta)
    if head_payload:
        head = load_head(Path(settings.index_dir) / head_payload["file"], device="cpu")
        log_marker("M-RETRIEVAL", "get_runtime", "HEAD_LOADED", shape=describe_head(meta))
    _runtime = Runtime(
        encoder=load_encoder(model_id=model_id, device=settings.encoder_device),
        index=index,
        store=CatalogStore.from_jsonl(settings.catalog_jsonl),
        meta=meta,
        head=head,
    )
    log_marker("M-RETRIEVAL", "get_runtime", "READY", vectors=len(_runtime.index.slugs),
               catalog=len(_runtime.store), model=_runtime.encoder.model_id)
    return _runtime


@app.get("/health")
def health() -> dict:
    """Проба живости и состояния индекса."""
    meta = get_meta()
    return {
        "status": "ok",
        "service": "retrieval",
        "encoder": meta.get("encoder") or settings.encoder,
        "model": meta.get("model") or settings.resolved_encoder_model,
        "vector_backend": meta.get("backend") or settings.vector_backend,
        "head": describe_head(meta),
        "index_loaded": _runtime is not None,
        "index_size": len(_runtime.index.slugs) if _runtime else 0,
    }


@app.post("/search")
async def search_endpoint(image: UploadFile = File(...)) -> dict:
    """Ищет Top-K по каталогу для загруженного изображения."""
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty image")
    runtime = get_runtime()
    try:
        array = load_bgr(data)
    except ValueError as error:
        log_marker("M-RETRIEVAL", "search", "DECODE_FAILED", error=str(error))
        raise HTTPException(status_code=400, detail=str(error)) from error

    started = time.perf_counter()
    hits = await run_in_threadpool(
        search_images, runtime.encoder, runtime.index, runtime.store,
        [np.asarray(array)], settings.top_k, runtime.head,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0

    return {
        "encoder": runtime.meta.get("encoder") or settings.encoder,
        "model": runtime.encoder.model_id,
        "device": runtime.encoder.device,
        "latency_ms": round(latency_ms, 1),
        "top_k": [
            {
                "slug": hit.slug,
                "score": round(hit.score, 6),
                "name": hit.name,
                "winery": hit.winery,
                "region": hit.region,
                "year": hit.year,
            }
            for hit in hits
        ],
    }
