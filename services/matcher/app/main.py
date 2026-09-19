# START_MODULE_CONTRACT
# PURPOSE: FastAPI-сервис matcher: POST /rerank переранжирует Top-K по полям VLM и каталогу
# SCOPE: app, health, rerank_endpoint
# DEPENDS: fastapi, pydantic, packages.common.catalog, packages.common.config, packages.common.logging,
#          services.matcher.app.merge
# LINKS: M-MATCHER
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# app - FastAPI-приложение matcher
# RerankRequest - тело запроса: кандидаты и поля этикетки
# get_store - ленивая загрузка read-only каталога
# health - GET /health
# rerank_endpoint - POST /rerank: переранжированные кандидаты с объяснением
# END_MODULE_MAP
"""Сервис сопоставления полей VLM с каталогом.

Отдельный HTTP-сервис: gateway не импортирует чужой код, а вызывает /rerank по
JSON. Метаданные каталога читаются из read-only стора.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from packages.common.catalog import CatalogStore
from packages.common.config import settings
from packages.common.logging import log_marker
from services.matcher.app.merge import rerank_candidates

app = FastAPI(title="Svoe Vino Scanner — Matcher", version="0.1.0")

__all__ = ["app", "RerankRequest", "get_store", "health", "rerank_endpoint"]

_store: CatalogStore | None = None


class RerankRequest(BaseModel):
    """Запрос на переранжирование."""

    candidates: list[dict] = Field(default_factory=list)
    fields: dict = Field(default_factory=dict)


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
            raise HTTPException(status_code=503, detail=str(error)) from error
    return _store


@app.get("/health")
def health() -> dict:
    """Проба живости matcher."""
    return {"status": "ok", "service": "matcher",
            "text_weight": settings.matcher_text_weight,
            "vintage_bonus": settings.matcher_vintage_bonus,
            "sweetness_bonus": settings.matcher_sweetness_bonus}


@app.post("/rerank")
async def rerank_endpoint(request: RerankRequest) -> dict:
    """Переранжирует кандидатов по полям этикетки."""
    if not request.candidates:
        raise HTTPException(status_code=400, detail="empty candidates")
    reranked = rerank_candidates(
        request.candidates,
        request.fields,
        get_store(),
        text_weight=settings.matcher_text_weight,
        vintage_bonus=settings.matcher_vintage_bonus,
        sweetness_bonus=settings.matcher_sweetness_bonus,
    )
    log_marker("M-MATCHER", "rerank", "RESPONSE", candidates=len(reranked))
    return {
        "items": [
            {
                "slug": item.slug,
                "score": item.score,
                "base_score": item.base_score,
                "text_score": item.text_score,
                "vintage_match": item.vintage_match,
                "sweetness_match": item.sweetness_match,
                "color_match": item.color_match,
                "name": item.name,
                "winery": item.winery,
                "notes": item.notes,
            }
            for item in reranked
        ]
    }
