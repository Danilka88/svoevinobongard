# START_MODULE_CONTRACT
# PURPOSE: Поиск Top-K по индексу: агрегация TTA-вариантов по максимуму скора и обогащение метаданными каталога
# SCOPE: SearchHit, aggregate_variants, search_images
# DEPENDS: numpy, packages.common.catalog, services.retrieval.app.encoder, services.retrieval.app.index
# LINKS: M-RETRIEVAL
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# SearchHit - результат поиска: slug, score, метаданные каталога
# aggregate_variants - слияние выдач TTA-вариантов по максимальному скору
# search_images - кодирование изображений и Top-K по индексу
# END_MODULE_MAP
"""Поиск Tier-1 по каталогу.

TTA-варианты одного запроса дают несколько выдач; итоговый скор slug — максимум
по вариантам (устойчивее к неудачному кропу, чем усреднение).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from packages.common.catalog import CatalogStore
from packages.common.logging import log_marker
from services.retrieval.app.encoder import ImageEncoder
from services.retrieval.app.index import VectorIndex

__all__ = ["SearchHit", "aggregate_variants", "search_images"]


@dataclass(frozen=True)
class SearchHit:
    """Один результат поиска."""

    slug: str
    score: float
    name: str = ""
    winery: str = ""
    region: str = ""
    year: int | None = None


def aggregate_variants(variant_hits: list[list[tuple[str, float]]], k: int) -> list[tuple[str, float]]:
    """Сливает выдачи TTA-вариантов, оставляя по каждому slug максимальный скор."""
    best: dict[str, float] = {}
    for hits in variant_hits:
        for slug, score in hits:
            if slug not in best or score > best[slug]:
                best[slug] = score
    ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)
    return ranked[:k]


def search_images(
    encoder: ImageEncoder,
    index: VectorIndex,
    store: CatalogStore,
    images: list[np.ndarray],
    k: int = 30,
) -> list[SearchHit]:
    """Кодирует изображения и возвращает Top-K результатов с метаданными.

    Args:
        encoder: загруженный image-энкодер.
        index: векторный индекс эталонных фото.
        store: read-only каталог для метаданных.
        images: список BGR-изображений (TTA-варианты).
        k: сколько кандидатов вернуть.

    Returns:
        Список SearchHit, отсортированный по убыванию скора.
    """
    if not images:
        return []
    vectors = encoder.encode_arrays(images)
    variant_hits = [index.search(vector, k=k) for vector in vectors]
    merged = aggregate_variants(variant_hits, k)

    hits: list[SearchHit] = []
    for slug, score in merged:
        summary = store.summary(slug)
        hits.append(SearchHit(
            slug=slug,
            score=float(score),
            name=summary.get("name", ""),
            winery=summary.get("winery", ""),
            region=summary.get("region", ""),
            year=summary.get("year"),
        ))

    margin = 0.0
    if len(hits) >= 2:
        margin = round(hits[0].score - hits[1].score, 6)
    log_marker("M-RETRIEVAL", "search", "TOPK", k=len(hits), top1=hits[0].slug if hits else "none",
               margin=margin)
    return hits
