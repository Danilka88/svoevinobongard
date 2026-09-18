# START_MODULE_CONTRACT
# PURPOSE: Единый источник истины для API-контрактов сканера: scan, card, eval и ошибки
# SCOPE: TopKItem, Card, ScanResponse, EvalPredictResponse, ErrorResponse
# DEPENDS: pydantic
# LINKS: M-CONTRACTS
# ROLE: TYPES
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# TopKItem - один кандидат выдачи: slug, score, name, winery
# Card - карточка вина для rich-ответа UI
# ScanResponse - rich-ответ POST /v1/scan
# EvalPredictResponse - плоский ответ POST /v1/eval/predict для скрипта организатора
# ErrorResponse - контракт ошибки: error, detail
# END_MODULE_MAP
"""API-контракты сканера «Своё Вино».

Единый источник истины для gateway, UI и ML-сервисов (ADR: контракты первыми).
"""
from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = [
    "TopKItem",
    "Card",
    "ScanResponse",
    "EvalPredictResponse",
    "ErrorResponse",
]


class TopKItem(BaseModel):
    """Один кандидат поисковой выдачи."""

    slug: str
    score: float
    name: str | None = None
    winery: str | None = None


class Card(BaseModel):
    """Карточка вина для мобильного UI."""

    name: str = ""
    winery: str = ""
    region: str = ""
    category: str = ""
    color: str = ""
    grape: str = ""
    description: str = ""
    year: int | None = None
    sweetness: str | None = None
    image: str | None = None
    image_url: str | None = None


class ScanResponse(BaseModel):
    """Rich-ответ основного сценария сканирования."""

    slug: str | None
    confidence: float = 0.0
    f1_top1: float | None = None
    f1_top5: float | None = None
    margin: float = 0.0
    path: str = "retrieval"
    latency_ms: float = 0.0
    top5: list[TopKItem] = Field(default_factory=list)
    not_found: bool = False
    card: Card | None = None


class EvalPredictResponse(BaseModel):
    """Контракт организатора: строго плоский slug на верхнем уровне."""

    slug: str | None


class ErrorResponse(BaseModel):
    """Контракт ошибки API."""

    error: str
    detail: str = ""
