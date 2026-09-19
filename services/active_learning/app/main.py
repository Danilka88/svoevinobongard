# START_MODULE_CONTRACT
# PURPOSE: FastAPI-сервис flywheel: приём логов, очередь ревью, решения человека, статистика и экспорт датасета
# SCOPE: app, LogRequest, ReviewRequest, health, ingest_endpoint, stats_endpoint, queue_endpoint, review_endpoint, export_endpoint
# DEPENDS: fastapi, pydantic, packages.common.catalog, packages.common.config,
#          services.active_learning.app.dataset, services.active_learning.app.store
# LINKS: M-ACTIVE-LEARNING
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# app - FastAPI-приложение flywheel
# LogRequest - тело запроса на запись лога
# ReviewRequest - решение человека по записи очереди
# health - GET /health со статусом хранилища
# ingest_endpoint - POST /ingest: принять JSONL-лог gateway
# stats_endpoint - GET /stats: сводка по хранилищу
# queue_endpoint - GET /queue: очередь ревью
# review_endpoint - POST /review/{entry_id}: подтвердить, исправить или отклонить метку
# export_endpoint - POST /export: выгрузить подтверждённые метки в датасет
# END_MODULE_MAP
"""Сервис flywheel (human-in-the-loop).

Автоматической разметки нет: сервис хранит логи, показывает очередь неуверенных
кейсов и принимает решения человека. Только после решения метка попадает в датасет.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from packages.common.catalog import CatalogStore
from packages.common.config import settings
from packages.common.logging import log_marker
from services.active_learning.app.dataset import export_confirmed
from services.active_learning.app.store import FlywheelStore, ingest_jsonl

app = FastAPI(title="Svoe Vino Scanner — Flywheel", version="0.1.0")

__all__ = ["app", "LogRequest", "ReviewRequest", "health", "ingest_endpoint",
           "stats_endpoint", "queue_endpoint", "review_endpoint", "export_endpoint"]

_store: FlywheelStore | None = None
_catalog: CatalogStore | None = None


class LogRequest(BaseModel):
    """Одна запись лога запроса."""

    timestamp: str = ""
    image_sha256: str = ""
    slug: str | None = None
    confidence: float = 0.0
    margin: float = 0.0
    path: str = ""
    latency_ms: float = 0.0
    not_found: bool = False
    uncertain: bool = False
    top_k: list[dict] = Field(default_factory=list)
    normalize_method: str = ""


class ReviewRequest(BaseModel):
    """Решение человека по записи очереди."""

    decision: str
    corrected_slug: str = ""
    reviewer: str = ""


def get_store() -> FlywheelStore:
    """Ленивая инициализация хранилища."""
    global _store
    if _store is None:
        _store = FlywheelStore(settings.flywheel_db_path)
    return _store


def get_catalog() -> CatalogStore:
    """Ленивый read-only каталог.

    Raises:
        HTTPException: если файл каталога не найден.
    """
    global _catalog
    if _catalog is None:
        try:
            _catalog = CatalogStore.from_jsonl(settings.catalog_jsonl)
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
    return _catalog


@app.get("/health")
def health() -> dict:
    """Проба живости и сводка хранилища."""
    return {"status": "ok", "service": "flywheel",
            "db": settings.flywheel_db_path,
            "human_in_the_loop": True,
            "stats": get_store().stats().as_dict()}


@app.post("/ingest")
def ingest_endpoint() -> dict:
    """Принимает JSONL-лог gateway в хранилище."""
    added = ingest_jsonl(get_store(), settings.flywheel_log_path)
    log_marker("M-ACTIVE-LEARNING", "ingest", "RESPONSE", added=added)
    return {"added": added, "stats": get_store().stats().as_dict()}


@app.get("/stats")
def stats_endpoint() -> dict:
    """Сводка по хранилищу."""
    return get_store().stats().as_dict()


@app.get("/queue")
def queue_endpoint(limit: int = 20, only_pending: bool = True) -> dict:
    """Очередь ревью: неуверенные кейсы впереди."""
    items = get_store().queue(limit=limit, only_pending=only_pending)
    return {"count": len(items), "items": items}


@app.post("/review/{entry_id}")
def review_endpoint(entry_id: int, request: ReviewRequest) -> dict:
    """Выносит решение человека по записи: confirmed | corrected | rejected."""
    try:
        get_store().review(entry_id, request.decision, request.corrected_slug, request.reviewer)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"status": "ok", "id": entry_id, "decision": request.decision,
            "stats": get_store().stats().as_dict()}


@app.post("/export")
def export_endpoint() -> dict:
    """Выгружает подтверждённые метки в датасет для scripts/train."""
    return export_confirmed(get_store(), get_catalog(), settings.flywheel_confirmed_path)
