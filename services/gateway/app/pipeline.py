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

import json
from pathlib import Path

import base64
import time
from dataclasses import dataclass, field

import numpy as np

from fastapi.concurrency import run_in_threadpool

from packages.common.catalog import CatalogStore
from packages.common.text_rank import fields_from_ocr, rrf, text_index, text_search
from packages.common.config import settings
from packages.common.logging import log_marker
from packages.common.near_dup import build_groups, disambiguate
from services.gateway.app.recommend import find_similar
from packages.common.observer import (escalation_trigger, merge_candidates,
                                      observation_fields, observer_candidates,
                                      parse_observation)
from packages.contracts.schemas import Card, ScanResponse, TopKItem
from services.gateway.app.clients import ServiceClients
from services.gateway.app.flywheel_log import append_query_log, build_log_record

__all__ = ["calibrated_confidence", "run_scan", "build_card"]


@dataclass
class PipelineTrace:
    """Диагностика прохождения пайплайна (для тестов и логов)."""

    normalize_method: str = "skipped"
    normalize_ok: bool = False
    normalize_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    ocr_ms: float = 0.0
    vlm_ms: float = 0.0
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


_text_first_state: dict = {}


def text_first_ranking(store: CatalogStore, fields: dict, visual: list[str], *,
                       limit: int = 100) -> list[str] | None:
    """Ранжирование text-first с RRF; None, если флаг выключен или поиск не уложился в бюджет.

    Текст служит отдельным источником кандидатов по **всему** каталогу: при 32% фото,
    где правильного вина нет в визуальном top-100, реранк поверх визуальной выдачи помочь
    не может. Вес текста в RRF ниже визуального, чтобы не вытеснять уверенный визуальный top-1.
    """
    if not settings.text_first_enabled:
        return None
    if "index" not in _text_first_state:
        _text_first_state["index"] = text_index(store.slug_to_record())
    started = time.perf_counter()
    ranked = [slug for slug, _ in text_search(fields, store.slug_to_record(), limit=limit,
                                              index=_text_first_state["index"])]
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if elapsed_ms > settings.text_first_max_latency_ms:
        log_marker("M-TEXT", "text_first", "BUDGET_EXCEEDED", latency_ms=round(elapsed_ms, 1),
                   budget_ms=settings.text_first_max_latency_ms)
        return None
    log_marker("M-TEXT", "text_first", "DONE", text_candidates=len(ranked),
               latency_ms=round(elapsed_ms, 1))
    return rrf([visual, ranked], weights=[1.0, 0.5])


def build_trace(response, diagnostics=None) -> dict:
    """Собирает витрину пути решения: поэтапные тайминги и параметры.

    Этапы берутся из реально измеренной диагностики (normalize, retrieval, ocr, vlm)
    и из самого ответа (общая задержка решения). Неприменимые этапы пропускаются: например,
    VLM при выключенном флаге в списке не появляется. На решение трейс не влияет.
    """
    if not settings.explain_enabled:
        return {}
    meta_views = 1
    meta_path = Path(settings.index_dir) / "meta.json"
    if meta_path.is_file():
        try:
            meta_views = int(json.loads(meta_path.read_text(encoding="utf-8")).get("views") or 1)
        except (json.JSONDecodeError, TypeError, ValueError):
            meta_views = 1

    total_ms = float(getattr(response, "latency_ms", 0.0) or 0.0)
    stages: list[dict] = []
    if diagnostics is not None:
        measured = [
            ("normalize", float(getattr(diagnostics, "normalize_ms", 0.0) or 0.0)),
            ("retrieval+geo", float(getattr(diagnostics, "retrieval_latency_ms", 0.0) or 0.0)),
            ("ocr", float(getattr(diagnostics, "ocr_ms", 0.0) or 0.0)),
            ("vlm", float(getattr(diagnostics, "vlm_ms", 0.0) or 0.0)),
        ]
        stages = [{"name": name, "latency_ms": round(value, 1)} for name, value in measured if value > 0]
    stages.append({"name": "decision", "latency_ms": round(total_ms, 1)})
    return {
        "stages": stages,
        "methods": {
            "encoder": getattr(settings, "encoder", "siglip2"),
            "geo": "orb",
            "ocr": settings.ocr_engine or "rapidocr",
            "normalization": "letterbox+pas",
            "detection": str(getattr(diagnostics, "normalize_method", "") or "none")
            if diagnostics is not None else "none",
        },
        "flags": {
            "TEXT_FUSION_MODE": settings.text_fusion_mode,
            "TEXT_FIRST_ENABLED": settings.text_first_enabled,
            "TEXT_CE_ENABLED": settings.text_ce_enabled,
            "COLOR_TIEBREAK_ENABLED": settings.color_tiebreak_enabled,
            "CYLINDER_UNWRAP_ENABLED": settings.cylinder_unwrap_enabled,
            "VLM_FALLBACK_ENABLED": settings.vlm_fallback_enabled,
            "OCR_ENGINE": settings.ocr_engine or "rapidocr",
            "NORMALIZE_CROP": settings.normalize_crop,
            "EXPLAIN_ENABLED": settings.explain_enabled,
            "VIEWS": meta_views,
        },
        "top_k": [{"slug": item.slug, "score": round(float(item.score), 6),
                   "geo_score": round(float(getattr(item, "geo_score", 0.0)), 4),
                   "inliers": int(getattr(item, "inliers", 0))} for item in (response.top5 or [])[:5]],
        "confidence": float(getattr(response, "confidence", 0.0) or 0.0),
        "margin": float(getattr(response, "margin", 0.0) or 0.0),
        "path": str(getattr(response, "path", "") or ""),
        "notes": list(getattr(diagnostics, "notes", []) or []) if diagnostics is not None else [],
    }


from packages.common.official_api import MEDIA_SIZE_HERO, image_proxy_url


from packages.common.calibration import apply as apply_calibration  # noqa: E402
from packages.common.calibration import load as load_calibration  # noqa: E402

_CALIBRATION_CACHE: dict[str, object] = {}


def _calibrated_probability(raw: float) -> float | None:
    """Калиброванная вероятность правильного ответа (C-061).

    По умолчанию калибровка выключена и поле остаётся пустым: решение и поведение не меняются.
    Параметры читаются один раз и кэшируются по пути.
    """
    if not settings.calibration_enabled:
        return None
    cached = _CALIBRATION_CACHE.get("params")
    if cached is None:
        cached = load_calibration(settings.calibration_path)
        _CALIBRATION_CACHE["params"] = cached or False
    if not cached:
        return None
    return apply_calibration(cached, raw)  # type: ignore[arg-type]


_NEAR_DUP_CACHE: dict[str, object] = {}


def _near_dup_groups(store: CatalogStore) -> dict:
    """Группы near-дубликатов каталога: строятся один раз и кэшируются по объекту каталога."""
    key = str(id(store))
    cached = _NEAR_DUP_CACHE.get(key)
    if cached is None:
        cached = build_groups(store.slug_to_record(), settings.near_dup_threshold)
        _NEAR_DUP_CACHE[key] = cached
    return cached  # type: ignore[return-value]


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
        # C-058: основное фото вина — крупное изображение, поэтому размер hero.
        image_url=image_proxy_url(str(extra.get("image_url") or ""), MEDIA_SIZE_HERO) or None,
        api_match=bool(extra.get("api_match")),
    )


RECOGNIZED_KEYS = ("winery", "wine_name", "vintage", "grape", "sweetness", "region")


def _collect_recognized_fields(ocr_result: dict | None, vlm_result: dict | None) -> dict:
    """Собирает распознанные с этикетки поля, уточняя OCR данными модели.

    Пустые значения не попадают в результат, год приводится к числу: UI показывает только то,
    что действительно прочитано, и не создаёт пустых строк.
    """
    def _fields(payload: dict | None) -> dict:
        if not payload or not payload.get("ok"):
            return {}
        raw = payload.get("fields") or {}
        merged: dict = {}
        for key, value in raw.items():
            target = {"name": "wine_name", "wine_type": "grape"}.get(key, key)
            if target in RECOGNIZED_KEYS and value not in (None, "") and str(value).strip():
                merged[target] = value
        return merged

    result: dict = {}
    for source in (_fields(ocr_result), _fields(vlm_result)):
        result.update(source)
    if "vintage" in result:
        digits = str(result["vintage"])
        if digits.isdigit():
            result["vintage"] = int(digits)
        else:
            result.pop("vintage")
    for key in list(result):
        if isinstance(result[key], str):
            result[key] = result[key].strip()
            if not result[key]:
                result.pop(key)
    return result


def _resolve_scan_reason(not_found: bool, confident: bool, recognized_fields: dict,
                         candidates: int) -> str:
    """Объясняет отказ: нет кандидатов, нет совпадения в каталоге или низкая уверенность."""
    if not_found:
        return "no_catalog_match" if candidates else "no_candidates"
    if not confident:
        return "low_confidence"
    return ""


async def run_scan(
    image_bytes: bytes,
    store: CatalogStore,
    clients: ServiceClients | None = None,
    filename: str = "query.jpg",
    official_index: dict[str, dict] | None = None,
    trace: PipelineTrace | None = None,
    session_id: str = "",
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

    normalize_started = time.perf_counter()
    normalized = await active_clients.normalize(image_bytes, filename)
    diagnostics.normalize_ms = (time.perf_counter() - normalize_started) * 1000.0
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
    text_first_used = False
    ocr_latency_ms = 0.0
    degraded = False
    recognized_fields: dict = {}
    scan_reason = ""
    ocr_result: dict | None = None

    retrieval_scores: list[float] = []
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
    retrieval_scores = [item.score for item in top5]  # шкала для уверенности — до реранка
    # C-065: расширенная выдача до десяти кандидатов. Прежнее поле top5 сохранено для совместимости,
    # а кандидаты с шестого по десятый берутся в порядке поиска (реранк переупорядочивает только пятёрку).
    top10 = list(top5) + [
        TopKItem(slug=item.get("slug", ""), score=float(item.get("score", 0.0)), name=item.get("name"),
                 winery=item.get("winery"), geo_score=float(item.get("geo_score", 0.0)),
                 inliers=int(item.get("inliers", 0)))
        for item in items[len(top5): max(5, int(settings.scan_top_k))]
        if item.get("slug")
    ]

    latency_ms = (time.perf_counter() - started) * 1000.0
    if not top5:
        log_marker("M-GATEWAY", "run_scan", "NO_CANDIDATES", latency_ms=round(latency_ms, 1))
        return ScanResponse(slug=None, not_found=True, path="retrieval",
                            latency_ms=round(latency_ms, 1), top5=[], card=None,
                            reason="no_candidates")

    best = top5[0]
    diagnostics.top1_score = best.score
    diagnostics.second_score = top5[1].score if len(top5) > 1 else 0.0
    margin = round(best.score - diagnostics.second_score, 6)
    confident = (best.score >= settings.score_tau) and (margin >= settings.margin_delta)

    # Ветвление C-010: сначала дешёвый OCR (сотни мс), затем VLM (C-004) при явном включении.
    if not confident and settings.ocr_enabled:
        ocr_started = time.perf_counter()
        recognized = await active_clients.ocr(payload)
        ocr_result = recognized
        ocr_latency_ms = (time.perf_counter() - ocr_started) * 1000.0
        fields = (recognized or {}).get("fields") or {}
        recognized_fields = _collect_recognized_fields(recognized, None)
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
            if recognized is None:
                scan_reason = "ocr_unavailable"
                diagnostics.notes.append("ocr unavailable; recognized fields skipped")
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
            recognized_fields = _collect_recognized_fields(ocr_result, extracted)
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

    # Уверенность считается на ОДНОЙ шкале: базовые retrieval-скоры фиксируются до реранка.
    # Иначе после matcher-фьюжна в calibrated_confidence попадали fused-скоры, посчитанные
    # по другой шкале (base + текстовые бонусы), и пороги SCORE_TAU/MARGIN_DELTA теряли смысл.
    confidence = round(calibrated_confidence(retrieval_scores), 6)
    not_found = best.score < settings.score_tau
    confident = (best.score >= settings.score_tau) and (margin >= settings.margin_delta)
    path = ("text_first" if text_first_used else
            ("vlm_fallback" if vlm_used else ("ocr_fallback" if ocr_used else "retrieval")))

    # C-062: режим наблюдателя. Включается флагом и срабатывает только при недостаточной уверенности:
    # модель описывает этикетку, поиск идёт по ВСЕМУ каталогу, а гибрид с визуальным топом может дать
    # кандидата, которого нет в галерее эталонов (новые и перерисованные этикетки). Каталог остаётся
    # источником истины: наблюдатель не выбирает товар, он лишь описывает видимое.
    observer_hits: list[tuple[str, float]] = []
    observation: dict = {}
    observer_used = False
    # C-074: LLM — ядро на каждом запросе. При observer_always наблюдение запускается независимо
    # от уверенности retrieval: мы не знаем, какое фото придёт от клиента.
    if settings.observer_enabled and (settings.observer_always or escalation_trigger(
            _calibrated_probability(best.score), confident, settings.observer_threshold)):
        observed = await active_clients.observe(payload, "query.png")
        if observed and observed.get("ok"):
            observation = parse_observation(observed.get("raw_text") or "")
            observer_used = bool(observation_fields(observation))
            observer_hits = observer_candidates(observation, store.slug_to_record(),
                                                limit=settings.observer_top_k)
            image_slugs = {item.slug for item in top5}
            if observer_hits:
                merged = merge_candidates(observer_hits, [(item.slug, item.score) for item in top5])
                if merged and merged[0][0] not in image_slugs:
                    # Наблюдатель нашёл то, чего нет в визуальном топе: он получает право победить.
                    winner = merged[0][0]
                    summary = store.summary(winner)
                    top5 = [TopKItem(slug=winner, score=float(merged[0][1]),
                                     name=summary.get("name"), winery=summary.get("winery"))] + top5[:4]
                    best = top5[0]
                    margin = round(best.score - (top5[1].score if len(top5) > 1 else 0.0), 6)
                    path = "observer"
                    not_found = False
                    confident = True
        else:
            diagnostics.notes.append("observer unavailable; Tier-1 kept")

    # C-074: LLM-матчер. Пул кандидатов — находки наблюдения по всему каталогу плюс визуальный топ;
    # LLM выбирает из них лучшее совпадение или честно отказывается. Slug вне списка не принимается.
    if settings.llm_matcher_enabled and observation:
        records = store.slug_to_record()
        pool: list[str] = []
        for slug, _score in observer_hits:
            if slug not in pool:
                pool.append(slug)
        for item in top10:
            if item.slug and item.slug not in pool:
                pool.append(item.slug)
        candidates = [
            {"slug": slug, "name": (records.get(slug) or {}).get("name"),
             "winery": (records.get(slug) or {}).get("winery"),
             "year": (records.get(slug) or {}).get("year"),
             "grape": (records.get(slug) or {}).get("grape"),
             "sweetness": (records.get(slug) or {}).get("sweetness")}
            for slug in pool[:20]
        ]
        matched = await active_clients.match(observation, candidates)
        if matched and matched.get("slug"):
            winner = str(matched["slug"])
            summary = store.summary(winner)
            rest = [item for item in top10 if item.slug != winner]
            top5 = [TopKItem(slug=winner, score=float(matched.get("p_correct") or 0.9),
                             name=summary.get("name"), winery=summary.get("winery"))] + rest[:4]
            top10 = [top5[0]] + rest
            best = top5[0]
            margin = round(best.score - (top5[1].score if len(top5) > 1 else 0.0), 6)
            path = "llm_match"
            not_found = False
            confident = True
            diagnostics.notes.append(f"llm match: {winner}")
        elif matched and matched.get("source") == "llm" and matched.get("in_catalog") is False:
            # LLM явно сказала: среди кандидатов совпадения нет → честный отказ, а не случайное вино.
            not_found = True
            confident = False
            diagnostics.notes.append("llm match: no catalog match")
        elif matched is None:
            diagnostics.notes.append("llm matcher unavailable; retrieval kept")

    # C-065: дизамбигуация near-дубликатов и аналоги при отказе. Флаг по умолчанию выключен, поэтому
    # метрики гейта не меняются; при включении внутри одной серии выбираем по подтверждённым сигналам
    # различия, а если различить нельзя — честно помечаем ответ неоднозначным и отдаём кандидатов.
    ambiguous = False
    alternatives: list[TopKItem] = []
    if settings.near_dup_enabled and top10:
        groups = _near_dup_groups(store)
        decision = disambiguate([item.slug for item in top10], recognized_fields,
                                store.slug_to_record(), groups)
        if decision["ambiguous"]:
            ambiguous = True
            path = "ambiguous"
            not_found = False
            diagnostics.notes.append(f"near-dup: {decision['reason']}")
        elif decision["slug"] and decision["slug"] != best.slug:
            matched = next((item for item in top10 if item.slug == decision["slug"]), None)
            if matched is not None:
                top10 = [matched] + [item for item in top10 if item.slug != decision["slug"]]
                top5 = top10[:5]
                best = top5[0]
                path = "near_dup"
                not_found = False
                confident = True
                diagnostics.notes.append(f"near-dup: {decision['reason']}")
    if not_found and best.slug:
        # Аналоги из других виноделен — именно этого ждёт организатор при отказе.
        try:
            alternatives = [TopKItem(slug=str(item.get("slug") or ""),
                                     score=float(item.get("score") or 0.0),
                                     name=item.get("name"), winery=item.get("winery"))
                            for item in find_similar(store, best.slug, limit=6)
                            if item.get("slug")]
        except Exception:  # noqa: BLE001 - аналоги не обязательны для ответа
            alternatives = []

    log_marker("M-GATEWAY", "run_scan", "DECISION", slug=best.slug if not not_found else "none",
               score=round(best.score, 4), margin=margin, confidence=confidence,
               confident=str(confident), path=path, vlm_latency_ms=round(vlm_latency_ms, 1),
               ocr_latency_ms=round(ocr_latency_ms, 1),
               latency_ms=round(latency_ms, 1), normalize=diagnostics.normalize_method)
    response = ScanResponse(
        slug=best.slug,
        confidence=confidence,
        # Оценка на шкале поиска остаётся; вероятность заполняется только при включённой калибровке.
        p_correct=_calibrated_probability(best.score),
        # f1_top1 — уверенность решения по первому кандидату, f1_top5 — средний скор пятёрки:
        # это разные величины (раньше стояла копипаста), и обе — рантайм-прокси, а не измеренный F1
        f1_top1=round(confidence, 6),
        f1_top5=round(sum(item.score for item in top5[:5]) / max(len(top5[:5]), 1), 6),
        margin=margin,
        path=path,
        latency_ms=round(latency_ms, 1),
        top5=top5,
        top10=top10,
        ambiguous=ambiguous,
        alternatives=alternatives,
        not_found=not_found,
        confident=confident,
        vlm_used=vlm_used,
        vlm_latency_ms=round(vlm_latency_ms, 1),
        ocr_used=ocr_used,
        ocr_latency_ms=round(ocr_latency_ms, 1),
        degraded=degraded,
        observation=observation,
        recognized_fields=recognized_fields,
        reason=scan_reason or _resolve_scan_reason(not_found, confident, recognized_fields, len(top5)),
        card=build_card(store, best.slug, (official_index or {}).get(best.slug)),
    )

    # Flywheel (C-006): лог запроса без PII. Best-effort — сбой не влияет на ответ.
    if settings.flywheel_enabled:
        record = build_log_record(image_bytes, response, margin_threshold=settings.margin_delta,
                                 normalize_method=diagnostics.normalize_method)
        # C-044: сквозная связь «действие в интерфейсе ↔ вызов API». Только обезличенный идентификатор
        # сессии: ни изображений, ни пользовательских данных в лог не добавляется.
        if isinstance(record, dict) and session_id:
            record["session_id"] = session_id[:40]
        logged = await run_in_threadpool(append_query_log, record, settings.flywheel_log_path)
        if not logged:
            log_marker("M-GATEWAY", "run_scan", "FLYWHEEL_LOG_FAILED")

    # C-040: витрина пути решения с поэтапными таймингами (normalize/retrieval/ocr/vlm).
    # Собирается из измеренной диагностики; на решение не влияет.
    diagnostics.ocr_ms = ocr_latency_ms
    diagnostics.vlm_ms = vlm_latency_ms
    response.trace = build_trace(response, diagnostics)
    return response
