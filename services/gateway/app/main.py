# START_MODULE_CONTRACT
# PURPOSE: Публичный API gateway — health, контракт организатора /v1/eval/predict и rich-ответ /v1/scan
# SCOPE: app, get_store, health, warmup, eval_predict, scan, similar, sommelier
# DEPENDS: fastapi, packages.common.catalog, packages.contracts.schemas, services.gateway.app.pipeline
# LINKS: M-GATEWAY, M-CONTRACTS
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# app - FastAPI-приложение gateway
# VLM_MODEL - имя VLM из окружения
# ENCODER - имя image-энкодера из окружения
# enotourism - GET /v1/enotourism/{slug}: регион, туры и статьи портала
# get_official - ленивая загрузка официального обогащения v3 (C-017)
# get_store - ленивая загрузка read-only каталога
# health - GET /v1/health
# warmup - POST /v1/warmup (прогрев retrieval)
# eval_predict - POST /v1/eval/predict для скрипта организатора
# scan - POST /v1/scan с rich-ответом
# similar - GET /v1/similar/{slug}: аналоги из других виноделен
# sommelier - POST /v1/sommelier: цифровой сомелье и гастрономические пары
# END_MODULE_MAP
"""Gateway — единственная публичная точка входа.

Контракт организатора: плоский ``{"slug": "..."}`` на верхнем уровне.
Rich-ответ: slug, confidence, f1_top1, f1_top5, margin, path, latency_ms, top5, card.
Внутренние сервисы наружу не публикуются.
"""
from __future__ import annotations

import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from packages.common.catalog import CatalogStore, load_official_v3
from packages.common.config import settings
from packages.common.llm.errors import LLMError
from packages.common.llm.router import LLMRouter
from packages.common.logging import log_marker
from packages.contracts.schemas import (
    EnotourismResponse,
    EvalPredictResponse,
    ScanResponse,
    SimilarItem,
    SimilarResponse,
    SommelierItem,
    SommelierRequest,
    SommelierResponse,
)
from services.gateway.app.clients import ServiceClients
from services.gateway.app.enotourism import enotourism_for
from services.gateway.app.pipeline import run_scan
from services.gateway.app.recommend import (
    build_sommelier_prompt,
    catalog_recommendations,
    find_similar,
    find_similar_official,
    official_dishes,
    pairing_for,
    parse_sommelier_reply,
)

__all__ = [
    "app",
    "enotourism",
    "get_official",
    "VLM_MODEL",
    "ENCODER",
    "get_store",
    "health",
    "warmup",
    "eval_predict",
    "scan",
    "similar",
    "sommelier",
]

app = FastAPI(title="Svoe Vino Scanner — Gateway", version="0.3.0")

VLM_MODEL = os.getenv("VLM_MODEL", "gemma4:12b")
ENCODER = os.getenv("ENCODER", "siglip2")

_store: CatalogStore | None = None


_official: dict[str, dict] | None = None


def get_official() -> dict[str, dict]:
    """Официальное обогащение v3 (C-017): карточка, сомелье и похожие берут данные отсюда.

    Файла может не быть — тогда карточка работает на данных CSV, как раньше.
    """
    global _official
    if _official is None:
        _official = load_official_v3()
        log_marker("M-GATEWAY", "get_official", "READY", slugs=len(_official))
    return _official


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
            log_marker("M-GATEWAY", "get_store", "CATALOG_MISSING", path=settings.catalog_jsonl)
            raise HTTPException(status_code=503, detail=str(error)) from error
        log_marker("M-GATEWAY", "get_store", "READY", records=len(_store))
    return _store


@app.get("/v1/health")
def health() -> dict:
    """Проба живости gateway."""
    log_marker("M-GATEWAY", "health", "OK")
    return {
        "status": "ok",
        "service": "gateway",
        "stage": "M1-tier1",
        "models": {"retrieval": ENCODER, "vlm": VLM_MODEL},
        "endpoints": ["/v1/health", "/v1/eval/predict", "/v1/scan"],
    }


@app.post("/v1/warmup")
async def warmup() -> dict:
    """Прогревает retrieval, чтобы первый скан не платил за загрузку модели (SLA)."""
    clients = ServiceClients()
    try:
        retrieval = await clients.warmup()
    finally:
        await clients.aclose()
    if retrieval is None:
        log_marker("M-GATEWAY", "warmup", "UNAVAILABLE")
        raise HTTPException(status_code=503, detail="retrieval warmup unavailable")
    log_marker("M-GATEWAY", "warmup", "OK", model=retrieval.get("model"))
    return {"status": "ok", "gateway": "ready", "retrieval": retrieval}


@app.get("/v1/similar/{slug}")
def similar(slug: str, limit: int = 6) -> SimilarResponse:
    """Похожие вина: сначала официальный API портала, иначе метаданные каталога."""
    items = find_similar_official(slug, limit=limit)
    source = "official"
    if not items:
        items = find_similar(get_store(), slug, limit=limit)
        source = "catalog"
    official = get_official().get(slug) or {}
    # дополняем официальные карточки локальными полями (цвет/сорт), если их нет
    for item in items:
        extra = get_official().get(item.get("slug", "")) or {}
        if source == "catalog":
            item.setdefault("image_url", str(extra.get("image_url") or ""))
            item.setdefault("source", "catalog")
        if not item.get("grape"):
            item["grape"] = str(extra.get("grapes", [""])[0] if extra.get("grapes") else "")
    log_marker("M-GATEWAY", "similar", "DONE", slug=slug, items=len(items), source=source)
    return SimilarResponse(base_slug=slug,
                           items=[SimilarItem(**item) for item in items])


@app.get("/v1/enotourism/{slug}")
def enotourism(slug: str) -> EnotourismResponse:
    """Энотуризм и статьи для вина: регион с фактом, туры региона и подборка материалов.

    Всё содержимое — из официального API портала (туры, регионы, статьи). Если кэш
    пуст, возвращаются пустые списки: блок удержания не ломает основной сценарий.
    """
    store = get_store()
    record = store.get(slug) or {}
    payload = enotourism_for(
        slug,
        official=get_official().get(slug),
        fallback_region=str(record.get("region") or ""),
        fallback_winery=str(record.get("winery") or ""),
    )
    log_marker("M-GATEWAY", "enotourism", "DONE", slug=slug,
               tours=len(payload["tours"]), articles=len(payload["articles"]))
    return EnotourismResponse(**payload)


@app.post("/v1/sommelier")
async def sommelier(request: SommelierRequest) -> SommelierResponse:
    """Цифровой сомелье: рекомендации и гастрономические пары.

    Сначала пробуем текстовую модель через LLM Router; при недоступности или
    битом ответе откатываемся к детерминированному подбору по каталогу, чтобы
    фича оставалась рабочей без LLM.
    """
    store = get_store()
    record = store.get(request.slug)
    if not record:
        raise HTTPException(status_code=404, detail=f"unknown slug: {request.slug}")

    fallback = catalog_recommendations(store, request.slug, request.answers)
    allowed = {item["slug"] for item in fallback}
    # пары берём из официальных гастросочетаний портала; LLM только формулирует
    dishes = official_dishes(get_official().get(request.slug) or {})
    pairing = pairing_for(record, dishes)
    source = "catalog"
    note = ""

    if fallback:
        try:
            router = LLMRouter()
            result = await run_in_threadpool(
                router.chat,
                build_sommelier_prompt(record, request.answers, fallback),
                role="text_model",
                max_tokens=320,
                temperature=0.0,
                response_format={"type": "json_object"},
                extra={"reasoning_effort": "none"},
                timeout_s=8.0,
            )
            parsed = parse_sommelier_reply(result.text, allowed)
            if parsed:
                by_slug = {item["slug"]: item for item in fallback}
                recommendations = [
                    SommelierItem(
                        slug=item["slug"],
                        name=by_slug.get(item["slug"], {}).get("name", ""),
                        winery=by_slug.get(item["slug"], {}).get("winery", ""),
                        why=item["why"] or by_slug.get(item["slug"], {}).get("why", ""),
                    )
                    for item in parsed["recommendations"]
                ]
                if parsed.get("pairing"):
                    pairing = parsed["pairing"]
                if recommendations:
                    log_marker("M-GATEWAY", "sommelier", "LLM_OK", items=len(recommendations))
                    return SommelierResponse(slug=request.slug, recommendations=recommendations,
                                             pairing=pairing, official_dishes=dishes,
                                             source="llm" if not dishes else "llm+official-dishes")
        except LLMError as error:
            note = f"llm unavailable: {type(error).__name__}"
            log_marker("M-GATEWAY", "sommelier", "LLM_FAILED", error=type(error).__name__)
        except Exception as error:  # noqa: BLE001 - сбой LLM не должен ломать фичу
            note = f"llm error: {type(error).__name__}"
            log_marker("M-GATEWAY", "sommelier", "LLM_ERROR", error=type(error).__name__)

    log_marker("M-GATEWAY", "sommelier", "CATALOG_FALLBACK", items=len(fallback))
    return SommelierResponse(
        slug=request.slug,
        recommendations=[SommelierItem(**item) for item in fallback],
        pairing=pairing,
        official_dishes=dishes,
        source="official-dishes" if dishes else source,
        note=note,
    )


@app.post("/v1/eval/predict")
async def eval_predict(image: UploadFile = File(...)) -> EvalPredictResponse:
    """Контракт организатора: image → плоский slug."""
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty image")
    try:
        result = await run_scan(data, get_store(), filename=image.filename or "query.jpg")
    except RuntimeError as error:
        log_marker("M-GATEWAY", "eval_predict", "PIPELINE_UNAVAILABLE", error=str(error))
        raise HTTPException(status_code=503, detail=str(error)) from error
    return EvalPredictResponse(slug=result.slug)


@app.post("/v1/scan")
async def scan(image: UploadFile = File(...)) -> ScanResponse:
    """Rich-ответ основного сценария сканирования."""
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty image")
    try:
        store = get_store()
        return await run_scan(data, store, filename=image.filename or "query.jpg",
                              official_index=get_official())
    except RuntimeError as error:
        log_marker("M-GATEWAY", "scan", "PIPELINE_UNAVAILABLE", error=str(error))
        raise HTTPException(status_code=503, detail=str(error)) from error
