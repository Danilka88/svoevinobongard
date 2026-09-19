# START_MODULE_CONTRACT
# PURPOSE: Поиск Top-K с геоверификацией: kNN по эмбеддингам, RANSAC-верификация топ-кандидатов и фьюжн скоров
# SCOPE: SearchHit, aggregate_variants, dedupe_by_slug, search_images
# DEPENDS: numpy, packages.common.catalog, packages.common.logging, services.retrieval.app.encoder,
#          services.retrieval.app.geo, services.retrieval.app.head, services.retrieval.app.index
# LINKS: M-RETRIEVAL
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# SearchHit - результат поиска: slug, скор, geo-скор, инлайеры и метаданные каталога
# POOL_FACTOR - во сколько раз пул строк больше k (галерея даёт несколько видов на slug)
# aggregate_variants - слияние выдач TTA-вариантов по максимальному скору
# dedupe_by_slug - схлопывание строк-видов в один slug по максимальному скору
# search_images - кодирование, Top-K и опциональная геоверификация с фьюжном
# END_MODULE_MAP
"""Поиск Tier-1 с геоверификацией.

Двухстадийная выдача: kNN по эмбеддингам дает Top-K, затем топ-кандидаты
переранжируются по геометрии локальных признаков (RANSAC-инлайеры). Это
различает near-duplicates, которые эмбеддинг видит как почти одинаковые.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from packages.common.catalog import CatalogStore
from packages.common.logging import log_marker
from services.retrieval.app.encoder import ImageEncoder
from services.retrieval.app.geo import GeoIndex, extract_features, fuse_scores
from services.retrieval.app.head import apply_head
from services.retrieval.app.index import VectorIndex

__all__ = ["POOL_FACTOR", "SearchHit", "aggregate_variants", "dedupe_by_slug", "search_images"]


@dataclass(frozen=True)
class SearchHit:
    """Один результат поиска."""

    slug: str
    score: float
    name: str = ""
    winery: str = ""
    region: str = ""
    year: int | None = None
    geo_score: float = 0.0
    inliers: int = 0


POOL_FACTOR = 10


def dedupe_by_slug(rows: list[tuple[str, float]], limit: int) -> list[tuple[str, float]]:
    """Схлопывает строки галереи в один slug, сохраняя максимальный скор.

    Мультивью-галерея хранит несколько векторов на позицию, поэтому k ближайших строк
    могут содержать один и тот же slug несколько раз. Без схлопывания в top-K остаётся
    мало разных вин, а гео-ветка ещё и фьюзит дубликаты как разные кандидаты.
    """
    best: dict[str, float] = {}
    for slug, score in rows:
        value = float(score)
        if slug not in best or value > best[slug]:
            best[slug] = value
    ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)
    return ranked[:limit]


def aggregate_variants(variant_hits: list[list[tuple[str, float]]], k: int) -> list[tuple[str, float]]:
    """Сливает выдачи TTA-вариантов, оставляя по каждому slug максимальный скор."""
    best: dict[str, float] = {}
    for hits in variant_hits:
        for slug, score in hits:
            if slug not in best or score > best[slug]:
                best[slug] = score
    ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)
    return ranked[:k]


def _to_hits(ranked: list[tuple[str, float]], store: CatalogStore,
             geo_scores: dict[str, float] | None = None,
             inliers: dict[str, int] | None = None) -> list[SearchHit]:
    """Превращает пары (slug, скор) в SearchHit с метаданными каталога."""
    geo_scores = geo_scores or {}
    inliers = inliers or {}
    hits: list[SearchHit] = []
    for slug, score in ranked:
        summary = store.summary(slug)
        hits.append(SearchHit(
            slug=slug,
            score=float(score),
            name=summary.get("name", ""),
            winery=summary.get("winery", ""),
            region=summary.get("region", ""),
            year=summary.get("year"),
            geo_score=float(geo_scores.get(slug, 0.0)),
            inliers=int(inliers.get(slug, 0)),
        ))
    return hits


def search_images(
    encoder: ImageEncoder,
    index: VectorIndex,
    store: CatalogStore,
    images: list[np.ndarray],
    k: int = 30,
    head: object | None = None,
    geo_index: GeoIndex | None = None,
    geo_weight: float = 0.0,
    geo_topk: int = 10,
    geo_min_matches: int = 8,
) -> list[SearchHit]:
    """Ищет Top-K с опциональной геоверификацией.

    Args:
        encoder: загруженный image-энкодер.
        index: векторный индекс эталонных фото.
        store: read-only каталог для метаданных.
        images: список BGR-изображений (TTA-варианты); первый используется для геоверификации.
        k: сколько кандидатов вернуть.
        head: проекционная голова домена (должна совпадать с индексом).
        geo_index: предвычисленные локальные признаки эталонов.
        geo_weight: вклад геоверификации в итоговый скор (0 отключает стадию).
        geo_topk: сколько верхних кандидатов верифицировать.
        geo_min_matches: минимум хороших матчей для RANSAC.

    Returns:
        Список SearchHit, отсортированный по убыванию итогового скора.
    """
    if not images:
        return []

    if geo_index is None or geo_weight <= 0.0:
        vectors = apply_head(head, encoder.encode_arrays(images))
        # пул строк шире k: иначе после схлопывания видов до разных slug остаётся слишком мало позиций
        pool = max(k * POOL_FACTOR, 200)
        merged = aggregate_variants([index.search(vector, k=pool) for vector in vectors], pool)
        hits = _to_hits(dedupe_by_slug(merged, k), store)
        log_marker("M-RETRIEVAL", "search", "TOPK", k=len(hits), geo="off",
                   top1=hits[0].slug if hits else "none")
        return hits

    primary = images[0]
    vector = apply_head(head, encoder.encode_arrays([primary]))[0]
    pool = max(k * POOL_FACTOR, 200)
    # сначала схлопываем виды в позиции, и только потом верифицируем и фьюзим:
    # иначе дубликаты одного вина занимают весь top-K и гео считает их независимо
    candidates = dedupe_by_slug(index.search(vector, k=pool), k)
    query_features = extract_features(primary, method=geo_index.method)

    fused: list[tuple[str, float]] = []
    geo_scores: dict[str, float] = {}
    inliers: dict[str, int] = {}
    verified = 0
    for rank, (slug, retrieval_score) in enumerate(candidates):
        if rank < geo_topk:
            result = geo_index.verify_slug(slug, query_features, min_matches=geo_min_matches)
            geo_scores[slug] = result.score
            inliers[slug] = result.inliers
            verified += 1
            fused.append((slug, fuse_scores(retrieval_score, result.score, geo_weight)))
        else:
            fused.append((slug, float(retrieval_score)))

    fused.sort(key=lambda item: item[1], reverse=True)
    hits = _to_hits(fused, store, geo_scores, inliers)
    log_marker("M-RETRIEVAL", "search", "TOPK", k=len(hits), geo=f"on({verified})",
               weight=geo_weight, top1=hits[0].slug if hits else "none",
               top1_geo=round(hits[0].geo_score, 4) if hits else 0.0)
    return hits
