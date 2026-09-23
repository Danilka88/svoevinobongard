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
# catalog - GET /v1/catalog: браузер каталога с фильтрами
# demo - GET /v1/demo/metrics: метрики для дашборда
# wine - GET /v1/wine/{slug}: полная карточка вина для каталога
# get_media_index - медиа-индекс каталога (C-034)
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

import io
import os
import time

import json
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from fastapi.concurrency import run_in_threadpool

from packages.common.catalog import CatalogStore, load_official_v3
from packages.common.config import settings
from packages.common.llm.errors import LLMError
from packages.common.llm.router import LLMRouter
from packages.common.learning import (append_feedback, evaluate_trust, feedback_summary,
                                      load_feedback, moderation_queue, rate_limit_exceeded,
                                      sanitize_feedback, vote_key)
from packages.common.llm.structured import validation_stats
from packages.common.logging import log_marker
from packages.common.telemetry import MAX_EVENTS_PER_BATCH, append_events, telemetry_summary
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
from services.gateway.app.enotourism import enotourism_for, load_collections
from services.gateway.app.media import (catalog_search, demo_metrics, load_media,
                                        media_for, warm_catalog_index)
from services.gateway.app.pipeline import build_card, build_trace, run_scan
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
    "demo",
    "wine",
    "enotourism",
    "get_media_index",
    "get_official",
    "VLM_MODEL",
    "ENCODER",
    "get_store",
    "health",
    "warmup",
    "eval_predict",
    "scan",
    "similar",
    "catalog",
    "sommelier",
]

app = FastAPI(title="Svoe Vino Scanner — Gateway", version="0.3.0")

# C-046/BUG-005: интерфейс и организаторский скрипт ходят с других источников, поэтому CORS нужен.
# Список источников настраивается через окружение, по умолчанию — только локальная разработка.
_CORS_ORIGINS = [origin.strip() for origin in
                 os.environ.get("CORS_ORIGINS", "http://127.0.0.1:3000,http://localhost:3000").split(",")
                 if origin.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_CORS_ORIGINS, allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])

# C-046/BUG-002: вырожденные изображения (например 1x1) ломали ресайз в поиске и давали 503.
# Такие кадры отсекаются до обращения к retrieval: ответ остаётся корректным отказом.
_MIN_IMAGE_SIDE = 8


def _image_too_small(data: bytes) -> bool:
    """Проверяет, что изображение пригодно для поиска по размеру сторон."""
    try:
        with Image.open(io.BytesIO(data)) as probe:
            width, height = probe.size
        return min(width, height) < _MIN_IMAGE_SIDE
    except Exception:  # noqa: BLE001 - нечитаемое изображение обработается дальше по общему пути
        return False

VLM_MODEL = os.getenv("VLM_MODEL", "gemma4:12b")
ENCODER = os.getenv("ENCODER", "siglip2")

_store: CatalogStore | None = None


_official: dict[str, dict] | None = None


def get_media_index() -> dict[str, dict]:
    """Медиа-индекс каталога (C-034): картинки, градиенты, сорта и блюда."""
    return load_media()


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
        "endpoints": ["/v1/health", "/v1/warmup", "/v1/eval/predict", "/v1/scan",
                      "/v1/similar/{slug}", "/v1/sommelier", "/v1/enotourism/{slug}"],
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


def _dish_image(name: str, dish_images: list) -> str:
    """Картинка блюда по названию; пустая строка, если соответствия нет."""
    from rapidfuzz import fuzz  # noqa: PLC0415

    best, best_score = "", 0.0
    for item in dish_images:
        if not isinstance(item, dict):
            continue
        score = fuzz.token_set_ratio(str(item.get("name") or "").lower(), str(name or "").lower()) / 100.0
        if score > best_score:
            best, best_score = str(item.get("url") or ""), score
    return best if best_score >= 0.6 else ""


def _sensory_note(record: dict, pairing: list[str]) -> str:
    """Короткая детерминированная заметка «почему подходит» без обращения к LLM."""
    parts = []
    category = str(record.get("category_official") or record.get("category") or "").strip()
    grapes = record.get("grapes") or ([record.get("grape")] if record.get("grape") else [])
    if category:
        parts.append(category.lower())
    if grapes:
        parts.append("сорт " + ", ".join(str(item) for item in grapes[:3]))
    description = str(record.get("description") or "").strip()
    if description:
        parts.append(description.split(".")[0][:140])
    if pairing:
        parts.append("подходит к: " + ", ".join(pairing[:3]).lower())
    return ". ".join(parts) + ("." if parts else "")


def _enrich_region(payload: dict, store: CatalogStore, slug: str) -> None:
    """Дополняет регион медиа и счётчиками (аддитивно, на месте)."""
    region = payload.get("region") or {}
    name = str(region.get("name") or "")
    entry = media_for(slug)
    region["image_url"] = str(entry.get("region_image_url") or "")
    region["geojson_url"] = str(entry.get("region_geojson_url") or "")
    regions_path = Path("data/api_cache/regions.json")
    if regions_path.is_file():
        items = (json.loads(regions_path.read_text(encoding="utf-8")).get("items") or [])
        for item in items:
            if str(item.get("name") or "").lower() == name.lower():
                region["manufacturers_count"] = int(item.get("manufacturersCount") or 0)
                region["events_count"] = int(item.get("eventsCount") or 0)
                break
    region["wines_count"] = sum(
        1 for record in store.slug_to_record().values()
        if str(record.get("region") or "").lower() == name.lower())
    region["tours_count"] = sum(
        1 for tour in (load_collections().get("tours") or []) if name.lower() in str(tour).lower())
    payload["region"] = region


@app.get("/v1/wine/{slug}")
def wine(slug: str, with_enotourism: bool = False) -> dict:
    """Полная карточка вина по slug — для перехода из каталога.

    Те же поля, что в /v1/scan.card, включая медиа портала; при with_enotourism добавляется
    блок энотуризма. Неизвестный slug — 404.
    """
    store = get_store()
    if not store.get(slug):
        raise HTTPException(status_code=404, detail=f"unknown slug: {slug}")
    card = build_card(store, slug, get_official().get(slug))
    entry = media_for(slug)
    card.category_gradient = str(entry.get("category_gradient") or "")
    card.region_image_url = str(entry.get("region_image_url") or "")
    card.region_geojson_url = str(entry.get("region_geojson_url") or "")
    card.grape_images = list(entry.get("grape_images") or [])
    card.dish_images = list(entry.get("dish_images") or [])
    payload: dict = {"slug": slug, "card": card.model_dump()}
    if with_enotourism:
        record = store.get(slug) or {}
        data = enotourism_for(slug, official=get_official().get(slug),
                              fallback_region=str(record.get("region") or ""),
                              fallback_winery=str(record.get("winery") or ""))
        _enrich_region(data, store, slug)
        payload["enotourism"] = data
    log_marker("M-GATEWAY", "wine", "DONE", slug=slug, enotourism=with_enotourism)
    return payload


@app.get("/v1/demo/metrics")
def demo() -> dict:
    """Метрики для дашборда: прокси, реальные фото, состояние гейта."""
    return demo_metrics()


@app.get("/v1/catalog")
def catalog(q: str = "", region: str = "", category: str = "", grape: str = "",
            winery: str = "", page: int = 1, per_page: int = 20, sort: str = "name") -> dict:
    """Браузер каталога: поиск, фильтры, пагинация и сортировка без тяжёлых моделей."""
    return catalog_search(get_store().slug_to_record(), get_media_index(), query=q, region=region,
                          category=category, grape=grape, winery=winery, page=page,
                          per_page=per_page, sort=sort)


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
    _enrich_region(payload, get_store(), slug)
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
    # C-037: slug необязателен — без него подбор идёт по всему каталогу по предпочтениям
    record = store.get(request.slug) if request.slug else {}
    catalog_record = (get_official().get(request.slug) or {}) if request.slug else {}
    if request.slug and not record:
        raise HTTPException(status_code=404, detail=f"unknown slug: {request.slug}")

    fallback = catalog_recommendations(store, request.slug, request.answers, likes=request.likes)
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
    entry = media_for(request.slug)
    dish_images = entry.get("dish_images") or []
    pairing_cards = [
        {"name": dish, "image_url": _dish_image(dish, dish_images)} for dish in pairing
    ]
    if request.slug:
        alternatives = [
            {"slug": item["slug"], "name": item.get("name", ""), "winery": item.get("winery", ""),
             "image_url": media_for(item["slug"]).get("image_url", "")}
            for item in find_similar_official(request.slug, limit=3)
        ]
    else:
        # подбор с нуля: альтернативы — следующие варианты каталога под те же условия
        extra = catalog_recommendations(store, "", request.answers, limit=8, likes=request.likes)[5:8]
        alternatives = [
            {"slug": item["slug"], "name": item.get("name", ""), "winery": item.get("winery", ""),
             "image_url": media_for(item["slug"]).get("image_url", "")}
            for item in extra
        ]
    enriched: list[SommelierItem] = []
    for item in fallback:
        slug_item = str(item.get("slug") or "")
        official_item = get_official().get(slug_item) or {}
        article = media_for(slug_item)
        dishes_for_item = article.get("dish_images") or []
        catalog_item = store.get(slug_item) or {}
        enriched.append(SommelierItem(
            slug=slug_item,
            name=item.get("name", "") or catalog_item.get("name", ""),
            winery=item.get("winery", "") or catalog_item.get("winery", ""),
            why=item.get("why", ""),
            image_url=str(article.get("image_url") or ""),
            sensory_note=_sensory_note(official_item or catalog_item,
                                       [dish for dish in (official_item.get("dishes") or [])][:3]),
            grape=str(catalog_item.get("grape") or ""),
            region=str(official_item.get("region_name") or catalog_item.get("region") or ""),
            category=str(official_item.get("category_official") or catalog_item.get("category") or ""),
            year=catalog_item.get("year"),
            pairing_cards=[{"name": dish, "image_url": _dish_image(dish, dishes_for_item)}
                           for dish in (official_item.get("dishes") or [])[:3]],
        ))
    return SommelierResponse(
        slug=request.slug,
        recommendations=enriched,
        pairing=pairing,
        official_dishes=dishes,
        source="official-dishes" if dishes else source,
        note=note,
        pairing_cards=pairing_cards,
        alternatives=alternatives,
        sensory_note=_sensory_note(catalog_record, pairing),
    )


_MAX_IMAGE_BYTES = 12 * 1024 * 1024
_IMAGE_SIGNATURES = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"RIFF", b"GIF87a", b"GIF89a", b"BM")


def _looks_like_image(data: bytes) -> bool:
    """Проверяет, что файл похож на изображение: по сигнатуре, при неизвестной — разбором."""
    if any(data.startswith(signature) for signature in _IMAGE_SIGNATURES):
        return True
    try:
        probe = Image.open(io.BytesIO(data))
        probe.verify()
        return True
    except Exception:  # noqa: BLE001 - неизвестный формат считаем не изображением
        return False


async def _read_image_upload(upload: UploadFile) -> bytes:
    """Читает файл и отклоняет пустой, слишком большой и не-изображение (C-043)."""
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty image")
    if len(data) > _MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="image too large")
    if not _looks_like_image(data):
        raise HTTPException(status_code=400, detail="file does not look like an image")
    return data


_TELEMETRY_PATH = Path("data") / "telemetry" / "ui.jsonl"


def _telemetry_path() -> str:
    """Путь файла обезличенных событий интерфейса."""
    return str(_TELEMETRY_PATH)


@app.post("/v1/telemetry")
async def _telemetry_ingest(payload: dict) -> dict:
    """Принимает обезличенный батч событий интерфейса.

    Приватность по умолчанию: поля вне белого списка отбрасываются, строки обрезаются, число
    событий ограничено. Ответ всегда успешный — телеметрия не имеет права ломать интерфейс.
    """
    try:
        events = payload.get("events") if isinstance(payload, dict) else None
        if not isinstance(events, list):
            return {"accepted": 0, "dropped": 0}
        session = str(payload.get("session_id") or "unknown")[:40]
        batch = list(events[:MAX_EVENTS_PER_BATCH])
        accepted = await run_in_threadpool(append_events, batch, session, _telemetry_path())
        return {"accepted": accepted, "dropped": max(0, len(events) - MAX_EVENTS_PER_BATCH)}
    except Exception as error:  # noqa: BLE001 - приём телеметрии best-effort
        log_marker("M-GATEWAY", "telemetry", "INGEST_FAILED", error=type(error).__name__)
        return {"accepted": 0, "dropped": 0}


@app.get("/v1/telemetry/summary")
async def _telemetry_aggregate() -> dict:
    """Сводка по обезличенным событиям: сессии, действия, ошибки, сценарии и задержки."""
    try:
        summary = await run_in_threadpool(telemetry_summary, _telemetry_path())
        # C-052: исходы валидации структурированных ответов моделей — аддитивное поле.
        summary["llm_validation"] = validation_stats()
        return summary
    except Exception as error:  # noqa: BLE001 - сводка best-effort
        log_marker("M-GATEWAY", "telemetry", "SUMMARY_FAILED", error=type(error).__name__)
        return {"events": 0, "sessions": 0, "note": "telemetry unavailable"}


async def _warm_catalog_on_startup() -> None:
    """Прогревает индекс форм каталога при старте gateway (C-047).

    Прогрев не должен мешать запуску: любая ошибка только логируется, а время построения
    записывается в маркер, чтобы было видно стоимость на реальном каталоге.
    """
    started = time.perf_counter()
    try:
        records = get_store().slug_to_record()
        count = await run_in_threadpool(warm_catalog_index, records)
        log_marker("M-GATEWAY", "startup", "CATALOG_WARMED", slugs=count,
                   ms=round((time.perf_counter() - started) * 1000.0, 1))
    except Exception as error:  # noqa: BLE001 - прогрев не обязателен для работы gateway
        log_marker("M-GATEWAY", "startup", "CATALOG_WARM_FAILED", error=type(error).__name__,
                   ms=round((time.perf_counter() - started) * 1000.0, 1))


@asynccontextmanager
async def _lifespan(application):
    """Жизненный цикл приложения: прогревает индекс форм каталога до приёма запросов (C-047).

    Прогрев обёрнут в защиту, поэтому сбой каталога не мешает сервису подняться.
    """
    await _warm_catalog_on_startup()
    yield


app.router.lifespan_context = _lifespan

_FEEDBACK_PATH = Path("data") / "flywheel" / "feedback.jsonl"


def _feedback_path() -> str:
    """Путь файла голосов пользователей."""
    return str(_FEEDBACK_PATH)


def _agent_consensus(slug: str) -> dict | None:
    """Консенсус агентов по вину: голоса агентов, а при их отсутствии — сам каталог.

    Каталог здесь выступает источником истины агентов: если вино есть в каталоге, агенты «за»,
    если slug неизвестен — «против». Так голоса подтверждающих пользователей могут стать
    доверенными, а голоса за несуществующее вино — нет.
    """
    if not slug:
        return None
    try:
        known = slug in get_store().slug_to_record()
    except Exception:  # noqa: BLE001 - без каталога доверие не подтверждается
        return None
    return {"vote": "up" if known else "down", "source": "catalog"}


@app.post("/v1/feedback")
async def _feedback_submit(payload: dict,
                           x_session_id: str = Header(default="", alias="X-Session-Id")) -> dict:
    """Принимает голос пользователя и сообщает, доверенный он или ушёл модератору.

    Анти-накрутка: один голос на сессию и изображение, ограничение частоты, влияние только при
    согласии нескольких независимых сессий и консенсуса агентов. Одиночный голос влияния не даёт.
    """
    vote = sanitize_feedback(payload, session_id=x_session_id)
    if vote is None:
        raise HTTPException(status_code=400, detail="vote must be up or down")
    path = _feedback_path()
    stored = load_feedback(path)
    if rate_limit_exceeded(stored, vote["session_id"]):
        log_marker("M-GATEWAY", "feedback", "RATE_LIMITED", session=vote["session_id"])
        return {"accepted": False, "trusted": False, "applied": False, "sessions": 0,
                "reason": "rate_limited", "detail": "too many votes per minute"}
    duplicate = any(vote_key(item) == vote_key(vote) for item in stored)
    if not duplicate:
        await run_in_threadpool(append_feedback, [vote], path)
        stored = load_feedback(path)
    verdict = evaluate_trust(stored, consensus=_agent_consensus(vote["slug"]),
                             image_sha256=vote["image_sha256"], slug=vote["slug"])
    await run_in_threadpool(append_events, [{"type": "action", "action": "feedback", "ok": True,
                                             "value": vote["vote"],
                                             "detail": "trusted" if verdict["trusted"] else "queued"}],
                            vote["session_id"], _telemetry_path())
    log_marker("M-GATEWAY", "feedback", "ACCEPTED", vote=vote["vote"], trusted=str(verdict["trusted"]),
               sessions=verdict["sessions"])
    return {"accepted": True, "duplicate": duplicate, "trusted": verdict["trusted"],
            "applied": verdict["applied"], "sessions": verdict["sessions"], "up": verdict["up"],
            "down": verdict["down"], "agreement": verdict["agreement"],
            "agent_vote": verdict["agent_vote"], "reasons": verdict["reasons"]}


@app.get("/v1/learning/summary")
async def _learning_summary() -> dict:
    """Агрегаты обратной связи: голоса, причины, доверенные и очередь."""
    try:
        votes = await run_in_threadpool(load_feedback, _feedback_path())
        return feedback_summary(votes, consensus_resolver=_agent_consensus)
    except Exception as error:  # noqa: BLE001 - сводка best-effort
        log_marker("M-GATEWAY", "learning", "SUMMARY_FAILED", error=type(error).__name__)
        return {"votes": 0, "unique_votes": 0, "sessions": 0, "note": "feedback unavailable"}


@app.get("/v1/learning/queue")
async def _learning_queue(limit: int = 50) -> dict:
    """Очередь на ручную проверку: всё, что не получило доверия, с провенансом."""
    try:
        votes = await run_in_threadpool(load_feedback, _feedback_path())
        items = moderation_queue(votes, limit=max(1, min(int(limit), 200)),
                                 consensus_resolver=_agent_consensus)
        return {"items": items, "total": len(items),
                "note": "модератор условный: раздел помечен визуально, реальная аутентификация не требуется"}
    except Exception as error:  # noqa: BLE001
        log_marker("M-GATEWAY", "learning", "QUEUE_FAILED", error=type(error).__name__)
        return {"items": [], "total": 0}


@app.post("/v1/eval/predict")
async def eval_predict(image: UploadFile = File(...),
                       x_session_id: str = Header(default="", alias="X-Session-Id")) -> EvalPredictResponse:
    """Контракт организатора: image → плоский slug.

    C-043: обещание держим жёстко — на плохой файл скрипт организатора не должен получать 5xx.
    Пустой файл, слишком большой или не-изображение отклоняются кодом 400 с понятной деталью,
    а внутренние сбои дают плоский ответ со значением null вместо ошибки.
    """
    data = await _read_image_upload(image)
    if _image_too_small(data):
        log_marker("M-GATEWAY", "eval_predict", "IMAGE_TOO_SMALL")
        return EvalPredictResponse(slug=None)
    try:
        result = await run_scan(data, get_store(), filename=image.filename or "query.jpg",
                                session_id=x_session_id[:40])
    except RuntimeError as error:
        log_marker("M-GATEWAY", "eval_predict", "PIPELINE_UNAVAILABLE", error=str(error))
        raise HTTPException(status_code=503, detail=str(error)) from error
    except HTTPException:
        raise
    except Exception as error:  # noqa: BLE001 - чужой скрипт не должен падать из-за нашего сбоя
        log_marker("M-GATEWAY", "eval_predict", "DEGRADED", error=type(error).__name__)
        return EvalPredictResponse(slug=None)
    return EvalPredictResponse(slug=result.slug)


@app.post("/v1/scan")
async def scan(image: UploadFile = File(...),
               x_session_id: str = Header(default="", alias="X-Session-Id")) -> ScanResponse:
    """Rich-ответ основного сценария сканирования."""
    data = await _read_image_upload(image)
    if _image_too_small(data):
        log_marker("M-GATEWAY", "scan", "IMAGE_TOO_SMALL")
        small_response = ScanResponse(slug=None, not_found=True, degraded=True, path="degraded",
                                      reason="image_too_small", top5=[], card=None)
        small_response.trace = build_trace(small_response)
        return small_response
    try:
        store = get_store()
        response = await run_scan(data, store, filename=image.filename or "query.jpg",
                                  official_index=get_official(), session_id=x_session_id[:40])
        if not response.trace:  # C-038/C-040: витрина уже собрана в пайплайне
            response.trace = build_trace(response)
        if response.card is not None and response.slug:
            entry = media_for(response.slug)
            response.card.category_gradient = str(entry.get("category_gradient") or "")
            response.card.region_image_url = str(entry.get("region_image_url") or "")
            response.card.region_geojson_url = str(entry.get("region_geojson_url") or "")
            response.card.grape_images = list(entry.get("grape_images") or [])
            response.card.dish_images = list(entry.get("dish_images") or [])
        return response
    except RuntimeError as error:
        log_marker("M-GATEWAY", "scan", "PIPELINE_UNAVAILABLE", error=str(error))
        raise HTTPException(status_code=503, detail=str(error)) from error
    except HTTPException:
        raise
    except Exception as error:  # noqa: BLE001 - C-043: сбой не должен превращаться в 500
        log_marker("M-GATEWAY", "scan", "DEGRADED", error=type(error).__name__)
        degraded_response = ScanResponse(slug=None, not_found=True, degraded=True, path="degraded",
                                         reason="internal_error", top5=[], card=None)
        degraded_response.trace = build_trace(degraded_response)
        return degraded_response
