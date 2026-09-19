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
# SimilarItem - похожее вино из другой винодельни с причиной совпадения
# SimilarResponse - ответ GET /v1/similar/{slug}
# SommelierRequest - запрос сомелье: slug и ответы пользователя
# SommelierItem - рекомендация сомелье со объяснением
# SommelierResponse - ответ POST /v1/sommelier: рекомендации, пары, источник
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
    "SimilarItem",
    "SimilarResponse",
    "SommelierRequest",
    "SommelierItem",
    "SommelierResponse",
]


class TopKItem(BaseModel):
    """Один кандидат поисковой выдачи."""

    slug: str
    score: float
    name: str | None = None
    winery: str | None = None
    geo_score: float = 0.0
    inliers: int = 0


class Card(BaseModel):
    """Карточка вина для мобильного UI (C-018: поля как на портале)."""

    name: str = ""
    winery: str = ""
    winery_slug: str = ""
    winery_url: str = ""
    region: str = ""
    category: str = ""
    color: str = ""
    grape: str = ""
    grapes: list[str] = Field(default_factory=list)
    description: str = ""
    year: int | None = None
    sweetness: str | None = None
    alcohol_pct: float | None = None
    temperature: str = ""
    dishes: list[str] = Field(default_factory=list)
    image: str | None = None
    image_url: str | None = None
    api_match: bool = False


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
    confident: bool = False
    vlm_used: bool = False
    vlm_latency_ms: float = 0.0
    ocr_used: bool = False
    ocr_latency_ms: float = 0.0
    degraded: bool = False
    card: Card | None = None


class EvalPredictResponse(BaseModel):
    """Контракт организатора: строго плоский slug на верхнем уровне."""

    slug: str | None


class ErrorResponse(BaseModel):
    """Контракт ошибки API."""

    error: str
    detail: str = ""


class SimilarItem(BaseModel):
    """Похожее вино из другой винодельни."""

    slug: str
    name: str = ""
    winery: str = ""
    region: str = ""
    category: str = ""
    grape: str = ""
    score: float = 0.0
    reason: str = ""
    image_url: str = ""
    source: str = "catalog"


class SimilarResponse(BaseModel):
    """Ответ GET /v1/similar/{slug}."""

    base_slug: str
    items: list[SimilarItem] = Field(default_factory=list)


class SommelierRequest(BaseModel):
    """Ответы пользователя на вопросы цифрового сомелье."""

    slug: str
    answers: dict[str, str] = Field(default_factory=dict)


class SommelierItem(BaseModel):
    """Рекомендация сомелье."""

    slug: str
    name: str = ""
    winery: str = ""
    why: str = ""


class SommelierResponse(BaseModel):
    """Ответ POST /v1/sommelier."""

    slug: str
    recommendations: list[SommelierItem] = Field(default_factory=list)
    pairing: list[str] = Field(default_factory=list)
    official_dishes: list[str] = Field(default_factory=list)
    source: str = "catalog"
    note: str = ""
