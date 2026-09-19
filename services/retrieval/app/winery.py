# START_MODULE_CONTRACT
# PURPOSE: Прототипы виноделен: средний нормированный эмбеддинг позиций как первый уровень поиска
# SCOPE: WineryPrototypes, build_prototypes, load_manufacturers, boost
# DEPENDS: numpy, json, pathlib, packages.common.logging
# LINKS: M-RETRIEVAL, M-CATALOG
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# WineryPrototypes - прототипы виноделен: матрица и отображение slug позиции в индекс винодельни
# build_prototypes - строит прототипы из векторов индекса и официальных производителей
# load_manufacturers - читает manufacturer_slug из обогащённого каталога v3
# boost - добавляет к скорам кандидатов близость их винодельни к запросу
# END_MODULE_MAP
"""Двухуровневый поиск по винодельням без обучения.

Идея: этикетки одной винодельни визуально ближе друг к другу, чем к остальному каталогу.
Прототип винодельни — средний L2-нормированный эмбеддинг её позиций. Запрос сначала
сравнивается с прототипами (первый уровень), и кандидаты из близких виноделен получают
небольшой бонус (второй уровень остаётся прежним kNN).

Метод включается флагом WINERY_PROTO_ENABLED и по умолчанию выключен: пока нет
подтверждённого выигрыша на proxy, поведение поиска не меняется.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from packages.common.logging import log_marker

DEFAULT_V3 = "data/catalog/catalog_enriched_v3.jsonl"

__all__ = ["WineryPrototypes", "boost", "build_prototypes", "load_manufacturers"]


class WineryPrototypes:
    """Прототипы виноделен и отображение позиций каталога на них."""

    def __init__(self, names: list[str], matrix: np.ndarray, position: np.ndarray) -> None:
        self.names = list(names)
        self.matrix = matrix
        self.position = position
        self._norm = np.linalg.norm(matrix, axis=1, keepdims=True)
        self._norm[self._norm == 0] = 1.0

    def __len__(self) -> int:
        return len(self.names)

    def scores(self, vector: np.ndarray) -> np.ndarray:
        """Косинусная близость запроса к каждому прототипу."""
        if self.matrix.size == 0:
            return np.zeros(0, dtype=np.float32)
        norm = float(np.linalg.norm(vector)) or 1.0
        return (self.matrix / self._norm) @ (vector / norm)

    def winery_of(self, index: int) -> int:
        """Индекс прототипа для позиции индекса; -1, если винодельня неизвестна."""
        if 0 <= index < len(self.position):
            return int(self.position[index])
        return -1

    @classmethod
    def empty(cls) -> "WineryPrototypes":
        """Пустой набор: используется, когда данных о производителях нет."""
        return cls([], np.zeros((0, 0), dtype=np.float32), np.full(0, -1, dtype=np.int32))


def load_manufacturers(path: str | Path = DEFAULT_V3) -> dict[str, str]:
    """Читает соответствие slug позиции -> manufacturer_slug из обогащения v3."""
    target = Path(path)
    if not target.is_file():
        log_marker("M-RETRIEVAL", "winery", "SKIP", reason="v3-missing", path=str(target))
        return {}
    mapping: dict[str, str] = {}
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            slug = str(record.get("slug") or "")
            maker = str(record.get("manufacturer_slug") or "")
            if slug and maker:
                mapping[slug] = maker
    log_marker("M-RETRIEVAL", "winery", "MANUFACTURERS", slugs=len(mapping))
    return mapping


def build_prototypes(slugs: list[str], vectors: np.ndarray,
                     manufacturers: dict[str, str] | None = None) -> WineryPrototypes:
    """Строит прототипы виноделен из векторов индекса.

    Args:
        slugs: slug позиций индекса (по одному на вектор).
        vectors: матрица эмбеддингов индекса.
        manufacturers: отображение slug -> manufacturer_slug; при None читается из v3.

    Returns:
        WineryPrototypes; пустой набор, если данных о производителях нет.
    """
    mapping = manufacturers if manufacturers is not None else load_manufacturers()
    if not mapping or not slugs or vectors.size == 0:
        return WineryPrototypes.empty()
    names: list[str] = []
    index_of: dict[str, int] = {}
    buckets: list[list[int]] = []
    position = np.full(len(slugs), -1, dtype=np.int32)
    for index, slug in enumerate(slugs):
        maker = mapping.get(slug)
        if not maker:
            continue
        if maker not in index_of:
            index_of[maker] = len(names)
            names.append(maker)
            buckets.append([])
        bucket = index_of[maker]
        buckets[bucket].append(index)
        position[index] = bucket
    matrix = np.zeros((len(names), vectors.shape[1]), dtype=np.float32)
    for bucket, rows in enumerate(buckets):
        block = vectors[rows]
        mean = block.mean(axis=0)
        norm = float(np.linalg.norm(mean)) or 1.0
        matrix[bucket] = mean / norm
    log_marker("M-RETRIEVAL", "winery", "PROTOTYPES", makers=len(names),
               indexed=sum(len(rows) for rows in buckets))
    return WineryPrototypes(names, matrix, position)


def boost(candidates: list[dict], prototypes: WineryPrototypes,
          query_vector: np.ndarray, weight: float = 0.15,
          top_wineries: int = 5) -> list[dict]:
    """Добавляет бонус кандидатам из виноделен, близких к запросу.

    Первый уровень — top_wineries прототипов по близости к запросу; их позиции получают
    прибавку weight * близость. Возвращает новый список, отсортированный по итоговому скору.
    """
    if not len(prototypes) or weight <= 0.0 or not candidates:
        return candidates
    scores = prototypes.scores(np.asarray(query_vector, dtype=np.float32))
    order = np.argsort(-scores)[:top_wineries]
    bonus: dict[int, float] = {int(index): float(scores[index]) * weight for index in order}
    boosted: list[dict] = []
    for item in candidates:
        index = int(item.get("index", -1))
        maker = prototypes.winery_of(index) if index >= 0 else -1
        extra = bonus.get(maker, 0.0) if maker >= 0 else 0.0
        boosted.append({**item, "score": float(item.get("score", 0.0)) + extra,
                        "winery_bonus": round(extra, 4)})
    boosted.sort(key=lambda entry: entry["score"], reverse=True)
    return boosted
