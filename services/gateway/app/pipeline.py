# START_MODULE_CONTRACT
# PURPOSE: Оркестрация пайплайна normalize → retrieval и сборка ScanResponse для /v1/scan и /v1/eval/predict
# SCOPE: calibrated_confidence, run_scan, build_card
# DEPENDS: httpx, packages.common.catalog, packages.contracts.schemas, services.gateway.app.clients
# LINKS: M-GATEWAY
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# calibrated_confidence - softmax-уверенность по top-K с калиброванной температурой
# run_scan - полный проход пайплайна и сборка ScanResponse
# build_card - карточка вина из метаданных каталога
# END_MODULE_MAP
"""Оркестрация пайплайна.

M1-путь: normalize → retrieval → решение по Top-1. Ветвление на VLM (C-004)
и геоверификация (C-003) добавляются отдельными changes.
Деградация: если normalize недоступен, изображение уходит в retrieval как есть.
"""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field

import numpy as np

from fastapi.concurrency import run_in_threadpool

from packages.common.catalog import CatalogStore
from packages.common.config import settings
from packages.common.logging import log_marker
from packages.contracts.schemas import Card, ScanResponse, TopKItem
from services.gateway.app.clients import ServiceClients
from services.gateway.app.flywheel_log import append_query_log, build_log_record

__all__ = ["calibrated_confidence", "run_scan", "build_card"]


@dataclass
class PipelineTrace:
    """Диагностика прохождения пайплайна (для тестов и логов)."""

    normalize_method: str = "skipped"
    normalize_ok: bool = False
    retrieval_latency_ms: float = 0.0
    top1_score: float = 0.0
    second_score: float = 0.0
    notes: list[str] = field(default_factory=list)


def calibrated_confidence(scores: list[float], temperature: float | None = None) -> float:
    """Считает уверенность top-1 как softmax-вероятность по top-K.

    Раньше confidence был сырым косинусным скором: он не калиброван и плохо
    отражает отрыв. Softmax с температурой, подобранной на proxy-наборе (C-003),
    дает вероятность, которую можно трактовать как уверенность решения.
    """
    if not scores:
        return 0.0
    temp = temperature if temperature is not None else settings.confidence_temperature
    scaled = np.asarray(scores, dtype=np.float64) / max(temp, 1e-6)
    scaled -= scaled.max()
    weights = np.exp(scaled)
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0
    return float(weights[0] / total)


def build_card(store: CatalogStore, slug: str, official: dict | None = None) -> Card:
    """Собирает карточку вина: данные CSV плюс официальные поля API (C-018).

    Args:
        store: read-only каталог CSV.
        slug: slug вина.
        official: запись catalog_enriched_v3 (если slug совпал с API). Пустая запись —
            карточка остаётся на данных CSV, как раньше.
    """
    data = store.card(slug)
    if not data:
        return Card()
    extra = official or {}
    winery_slug = str(extra.get("manufacturer_slug") or "")
    grapes = [str(item) for item in (extra.get("grapes") or []) if item]
    alcohol = extra.get("alcohol_pct")
    return Card(
        name=data.get("name") or "",
        winery=extra.get("manufacturer_name") or data.get("winery") or "",
        winery_slug=winery_slug,
        winery_url=f"/wineries/{winery_slug}" if winery_slug else "",
        region=extra.get("region_name") or data.get("region") or "",
        category=data.get("category") or extra.get("category_official") or "",
        color=str(extra.get("color_official") or data.get("color") or ""),
        grape=data.get("grape") or (grapes[0] if grapes else ""),
        grapes=grapes,
        description=str(extra.get("description") or data.get("description") or ""),
        year=data.get("year"),
        sweetness=data.get("sweetness"),
        alcohol_pct=float(alcohol) if isinstance(alcohol, (int, float)) else None,
        temperature=str(extra.get("temperature") or ""),
        dishes=[str(dish) for dish in (extra.get("dishes") or []) if dish],
        image=data.get("image"),
        image_url=str(extra.get("image_url") or "") or None,
        api_match=bool(extra.get("api_match")),
    )


async def run_scan(
    image_bytes: bytes,
    store: CatalogStore,
    clients: ServiceClients | None = None,
    filename: str = "query.jpg",
    official_index: dict[str, dict] | None = None,
    trace: PipelineTrace | None = None,
) -> ScanResponse:
    """Прогоняет изображение по пайплайну Tier-1.

    Args:
        image_bytes: сырые байты изображения.
        store: read-only каталог для карточки.
        clients: клиенты сервисов (по умолчанию создаются из env).
        filename: имя файла для multipart.
        trace: необязательный объект диагностики.

    Returns:
        ScanResponse: Top-1 slug, confidence, margin, top5, card.

    Raises:
        RuntimeError: если retrieval недоступен (без него ответ невозможен).
    """
    started = time.perf_counter()
    diagnostics = trace if trace is not None else PipelineTrace()
    active_clients = clients or ServiceClients()

    normalized = await active_clients.normalize(image_bytes, filename)
    payload = image_bytes
    if normalized and normalized.get("crops"):
        diagnostics.normalize_ok = True
        diagnostics.normalize_method = str(normalized.get("method", "unknown"))
        first = normalized["crops"][0]
        try:
            payload = base64.b64decode(first["png_base64"])
        except Exception:  # noqa: BLE001 - поврежденный ответ нормализации не критичен
            diagnostics.notes.append("normalize payload decode failed; using original")
    else:
        diagnostics.notes.append("normalize unavailable; using original image")

    retrieval_started = time.perf_counter()
    found = await active_clients.search(payload, "query.png")
    diagnostics.retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000.0
    if found is None:
        log_marker("M-GATEWAY", "run_scan", "RETRIEVAL_UNAVAILABLE")
        raise RuntimeError("retrieval service unavailable")

    items = found.get("top_k") or []
    reranked_items: list[dict] | None = None
    vlm_used = False
    vlm_latency_ms = 0.0
    ocr_used = False
    ocr_latency_ms = 0.0
    degraded = False

    top5 = [
        TopKItem(
            slug=item.get("slug", ""),
            score=float(item.get("score", 0.0)),
            name=item.get("name"),
            winery=item.get("winery"),
            geo_score=float(item.get("geo_score", 0.0)),
            inliers=int(item.get("inliers", 0)),
        )
        for item in items[:5]
        if item.get("slug")
    ]

    latency_ms = (time.perf_counter() - started) * 1000.0
    if not top5:
        log_marker("M-GATEWAY", "run_scan", "NO_CANDIDATES", latency_ms=round(latency_ms, 1))
        return ScanResponse(slug=None, not_found=True, path="retrieval",
                            latency_ms=round(latency_ms, 1), top5=[], card=None)

    best = top5[0]
    diagnostics.top1_score = best.score
    diagnostics.second_score = top5[1].score if len(top5) > 1 else 0.0
    margin = round(best.score - diagnostics.second_score, 6)
    confident = (best.score >= settings.score_tau) and (margin >= settings.margin_delta)

    # Ветвление C-010: сначала дешёвый OCR (сотни мс), затем VLM (C-004) при явном включении.
    if not confident and settings.ocr_enabled:
        ocr_started = time.perf_counter()
        recognized = await active_clients.ocr(payload)
        ocr_latency_ms = (time.perf_counter() - ocr_started) * 1000.0
        fields = (recognized or {}).get("fields") or {}
        if recognized and recognized.get("ok") and (fields.get("vintage") or fields.get("wine_type")):
            reranked = await active_clients.rerank(
                [{"slug": item.get("slug"), "score": float(item.get("score", 0.0))}
                 for item in items if item.get("slug")],
                fields,
            )
            if reranked and reranked.get("items"):
                reranked_items = reranked["items"]
                ocr_used = True
            else:
                degraded = True
                diagnostics.notes.append("matcher unavailable; Tier-1 kept")
        else:
            reason = (recognized or {}).get("reason", "ocr gave no usable fields")
            diagnostics.notes.append(f"ocr no signal: {reason}")
            log_marker("M-GATEWAY", "run_scan", "OCR_NO_SIGNAL", reason=str(reason)[:120])

    # Ветвление C-004: неуверенный Tier-1 → VLM-поля → matcher → переранжирование.
    if not confident and reranked_items is None and settings.vlm_fallback_enabled:
        vlm_started = time.perf_counter()
        extracted = await active_clients.extract(payload, "query.png")
        vlm_latency_ms = (time.perf_counter() - vlm_started) * 1000.0
        if extracted and extracted.get("ok"):
            fields = extracted.get("fields") or {}
            reranked = await active_clients.rerank(
                [{"slug": item.get("slug"), "score": float(item.get("score", 0.0))}
                 for item in items if item.get("slug")],
                fields,
            )
            if reranked and reranked.get("items"):
                reranked_items = reranked["items"]
                vlm_used = True
            else:
                degraded = True
                diagnostics.notes.append("matcher unavailable; Tier-1 kept")
        else:
            degraded = True
            reason = (extracted or {}).get("reason", "vlm unavailable")
            diagnostics.notes.append(f"vlm degraded: {reason}")
            log_marker("M-GATEWAY", "run_scan", "VLM_DEGRADED", reason=str(reason)[:120])
    elif not confident and reranked_items is None:
        degraded = True
        diagnostics.notes.append("no fallback produced a signal")

    if reranked_items:
        by_slug = {item.get("slug"): item for item in items}
        rebuilt: list[TopKItem] = []
        for entry in reranked_items[:5]:
            slug = str(entry.get("slug", ""))
            source = by_slug.get(slug, {})
            rebuilt.append(TopKItem(
                slug=slug,
                score=float(entry.get("score", 0.0)),
                name=entry.get("name") or source.get("name"),
                winery=entry.get("winery") or source.get("winery"),
                geo_score=float(source.get("geo_score", 0.0)),
                inliers=int(source.get("inliers", 0)),
            ))
        if rebuilt:
            top5 = rebuilt
            best = top5[0]
            diagnostics.top1_score = best.score
            diagnostics.second_score = top5[1].score if len(top5) > 1 else 0.0
            margin = round(best.score - diagnostics.second_score, 6)

    confidence = round(calibrated_confidence([item.score for item in top5]), 6)
    not_found = best.score < settings.score_tau
    confident = (best.score >= settings.score_tau) and (margin >= settings.margin_delta)
    path = "vlm_fallback" if vlm_used else ("ocr_fallback" if ocr_used else "retrieval")

    log_marker("M-GATEWAY", "run_scan", "DECISION", slug=best.slug if not not_found else "none",
               score=round(best.score, 4), margin=margin, confidence=confidence,
               confident=str(confident), path=path, vlm_latency_ms=round(vlm_latency_ms, 1),
               ocr_latency_ms=round(ocr_latency_ms, 1),
               latency_ms=round(latency_ms, 1), normalize=diagnostics.normalize_method)
    response = ScanResponse(
        slug=best.slug,
        confidence=confidence,
        # f1_top1 — уверенность решения по первому кандидату, f1_top5 — средний скор пятёрки:
        # это разные величины (раньше стояла копипаста), и обе — рантайм-прокси, а не измеренный F1
        f1_top1=round(confidence, 6),
        f1_top5=round(sum(item.score for item in top5[:5]) / max(len(top5[:5]), 1), 6),
        margin=margin,
        path=path,
        latency_ms=round(latency_ms, 1),
        top5=top5,
        not_found=not_found,
        confident=confident,
        vlm_used=vlm_used,
        vlm_latency_ms=round(vlm_latency_ms, 1),
        ocr_used=ocr_used,
        ocr_latency_ms=round(ocr_latency_ms, 1),
        degraded=degraded,
        card=build_card(store, best.slug, (official_index or {}).get(best.slug)),
    )

    # Flywheel (C-006): лог запроса без PII. Best-effort — сбой не влияет на ответ.
    if settings.flywheel_enabled:
        record = build_log_record(image_bytes, response, margin_threshold=settings.margin_delta,
                                 normalize_method=diagnostics.normalize_method)
        logged = await run_in_threadpool(append_query_log, record, settings.flywheel_log_path)
        if not logged:
            log_marker("M-GATEWAY", "run_scan", "FLYWHEEL_LOG_FAILED")
    return response
