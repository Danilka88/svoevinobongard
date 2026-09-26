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
    # C-034: медиа-поля портала (аддитивно; пустые значения, если данных нет)
    category_gradient: str = ""
    region_image_url: str = ""
    region_geojson_url: str = ""
    grape_images: list[dict] = Field(default_factory=list)
    dish_images: list[dict] = Field(default_factory=list)


class ScanResponse(BaseModel):
    """Rich-ответ основного сценария сканирования."""

    slug: str | None
    confidence: float = 0.0
    # C-061: калиброванная вероятность правильного ответа; None, если калибровка выключена.
    p_correct: float | None = None
    # C-062: наблюдение модели-наблюдателя (пусто, если режим выключен).
    observation: dict = Field(default_factory=dict)
    # C-077: результат LLM-нормализации чтения под каталог (пусто, если режим выключен).
    normalization: dict = Field(default_factory=dict)
    f1_top1: float | None = None
    f1_top5: float | None = None
    margin: float = 0.0
    path: str = "retrieval"
    latency_ms: float = 0.0
    top5: list[TopKItem] = Field(default_factory=list)
    # C-065: расширенная выдача до десяти кандидатов (top5 сохранён для совместимости).
    top10: list[TopKItem] = Field(default_factory=list)
    ambiguous: bool = False
    alternatives: list[TopKItem] = Field(default_factory=list)
    not_found: bool = False
    confident: bool = False
    vlm_used: bool = False
    vlm_latency_ms: float = 0.0
    ocr_used: bool = False
    ocr_latency_ms: float = 0.0
    degraded: bool = False
    card: Card | None = None
    trace: dict = Field(default_factory=dict)  # C-034: витрина пути решения, на решение не влияет
    # C-043: что прочитано с этикетки и почему нет точного ответа (аддитивно)
    recognized_fields: dict = Field(default_factory=dict)
    reason: str = ""


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
    """Ответы пользователя на вопросы цифрового сомелье.

    C-037: slug необязателен — без него подбор идёт по всему каталогу по предпочтениям.
    """

    slug: str = ""
    answers: dict[str, str] = Field(default_factory=dict)
    likes: list[str] = Field(default_factory=list)  # история вкуса из localStorage


class SommelierItem(BaseModel):
    """Рекомендация сомелье."""

    slug: str
    name: str = ""
    winery: str = ""
    why: str = ""
    # C-037: аддитивные поля для экрана сомелье
    image_url: str = ""
    sensory_note: str = ""
    grape: str = ""
    region: str = ""
    category: str = ""
    year: int | None = None
    pairing_cards: list[dict] = Field(default_factory=list)


class SommelierResponse(BaseModel):
    """Ответ POST /v1/sommelier.

    C-071: ``source`` принимает канонические значения ``official|deterministic|llm``,
    а ``llm_degraded`` показывает, что LLM-путь не дал ответа и сработал детерминированный
    подбор по каталогу. Оба поля аддитивны: старые клиенты продолжают работать.
    """

    slug: str
    recommendations: list[SommelierItem] = Field(default_factory=list)
    pairing: list[str] = Field(default_factory=list)
    official_dishes: list[str] = Field(default_factory=list)
    source: str = "catalog"
    note: str = ""
    # C-034: аддитивные поля для UI (pairing не меняется, чтобы не ломать существующих клиентов)
    pairing_cards: list[dict] = Field(default_factory=list)
    alternatives: list[dict] = Field(default_factory=list)
    sensory_note: str = ""
    # C-071: витрина деградации LLM-пути (аддитивно)
    llm_degraded: bool = False


class EnotourismRegion(BaseModel):
    """Факт о регионе вина из официального API."""

    name: str = ""
    fact: str = ""
    events_count: int = 0
    events_description: str = ""
    # C-034: медиа и счётчики региона (аддитивно)
    image_url: str = ""
    geojson_url: str = ""
    wines_count: int = 0
    manufacturers_count: int = 0
    tours_count: int = 0


class EnotourismTour(BaseModel):
    """Винный тур портала."""

    slug: str
    title: str = ""
    address: str = ""
    price: int | None = None
    image_url: str = ""
    url: str = ""
    match: str = ""


class EnotourismArticle(BaseModel):
    """Статья портала для блока «Истории в бокале»."""

    slug: str
    title: str = ""
    category: str = ""
    author: str = ""
    published_at: str = ""
    image_url: str = ""
    url: str = ""
    relevance: str = "latest"


class EnotourismResponse(BaseModel):
    """Ответ GET /v1/enotourism/{slug}."""

    slug: str
    region: EnotourismRegion = Field(default_factory=EnotourismRegion)
    tours: list[EnotourismTour] = Field(default_factory=list)
    articles: list[EnotourismArticle] = Field(default_factory=list)
    source: str = "official-api"
    note: str = ""
