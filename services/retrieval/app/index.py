# START_MODULE_CONTRACT
# PURPOSE: Векторный индекс каталога: FAISS (prod-dev) или numpy-exact, сохранение и поиск по косинусу
# SCOPE: VectorIndex, save_index, load_index
# DEPENDS: numpy, json, faiss (опционально)
# LINKS: M-RETRIEVAL, M-CATALOG
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# VectorIndex - индекс векторов и slugs с поиском Top-K по косинусу
# save_index - запись vectors.npy, slugs.json и meta.json
# load_index - чтение индекса из каталога
# END_MODULE_MAP
"""Векторный индекс Tier-1.

Backend выбирается через ``VECTOR_BACKEND``:
- ``numpy`` — точный перебор через матричное умножение (значение по умолчанию);
- ``faiss`` — FAISS IndexFlatIP (только там, где libomp от faiss не конфликтует с torch;
  на macOS arm64 это приводит к OMP Error #15, поэтому нужен numpy).

Векторы L2-нормированы, поэтому внутреннее произведение равно косинусной близости.
PostgreSQL + pgvector — prod-цель; при недоступном Docker в M1 используется этот backend.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from packages.common.logging import log_marker

__all__ = ["VectorIndex", "save_index", "load_index"]

VECTORS_FILE = "vectors.npy"
SLUGS_FILE = "slugs.json"
META_FILE = "meta.json"


def _import_faiss():
    """Импортирует faiss, возвращая None, если он недоступен."""
    try:
        import faiss  # type: ignore

        return faiss
    except Exception:  # noqa: BLE001 - отсутствие faiss является штатным fallback
        return None


class VectorIndex:
    """Индекс эмбеддингов эталонных фото."""

    def __init__(self, backend: str = "numpy") -> None:
        self.backend = backend
        self.effective_backend = backend
        self.slugs: list[str] = []
        self.dim: int = 0
        self._matrix: np.ndarray | None = None
        self._faiss_index = None

    def build(self, slugs: list[str], vectors: np.ndarray) -> None:
        """Строит индекс из slug и L2-нормированных векторов.

        Raises:
            ValueError: если число slug не совпадает с числом векторов.
        """
        if len(slugs) != vectors.shape[0]:
            raise ValueError(f"slugs ({len(slugs)}) != vectors ({vectors.shape[0]})")
        self.slugs = list(slugs)
        self.dim = int(vectors.shape[1]) if vectors.size else 0
        self._matrix = np.ascontiguousarray(vectors.astype(np.float32))

        # faiss импортируется только когда он реально запрошен: на macOS arm64
        # libomp от faiss конфликтует с libomp от torch (OMP Error #15) и роняет процесс.
        faiss = _import_faiss() if self.backend == "faiss" else None
        if faiss is not None and self.dim:
            index = faiss.IndexFlatIP(self.dim)
            index.add(self._matrix)
            self._faiss_index = index
            self.effective_backend = "faiss"
            log_marker("M-RETRIEVAL", "build_index", "FAISS", vectors=len(self.slugs), dim=self.dim)
        else:
            self._faiss_index = None
            self.effective_backend = "numpy"
            log_marker("M-RETRIEVAL", "build_index", "NUMPY", vectors=len(self.slugs),
                       dim=self.dim, requested=self.backend)

    def search(self, query: np.ndarray, k: int = 30) -> list[tuple[str, float]]:
        """Возвращает Top-K пар (slug, косинусный скор), отсортированных по убыванию."""
        if self._matrix is None or not self.slugs:
            return []
        vector = np.asarray(query, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return []
        vector = (vector / norm).reshape(1, -1)
        limit = min(k, len(self.slugs))

        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(vector, limit)
            pairs = [
                (self.slugs[int(idx)], float(score))
                for score, idx in zip(scores[0], indices[0])
                if idx >= 0
            ]
        else:
            similarities = (self._matrix @ vector.reshape(-1)).astype(np.float32)
            top = np.argpartition(-similarities, limit - 1)[:limit] if limit < len(self.slugs) else np.arange(len(self.slugs))
            ordered = sorted(top.tolist(), key=lambda idx: -float(similarities[idx]))
            pairs = [(self.slugs[idx], float(similarities[idx])) for idx in ordered]

        pairs.sort(key=lambda item: item[1], reverse=True)
        return pairs

    def score(self, slug: str, query: np.ndarray) -> float:
        """Косинусный скор конкретного slug (для self-retrieval проверок)."""
        if self._matrix is None or slug not in self.slugs:
            return 0.0
        vector = np.asarray(query, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return 0.0
        return float(self._matrix[self.slugs.index(slug)] @ (vector / norm))


def save_index(directory: str | Path, index: VectorIndex, meta: dict | None = None) -> None:
    """Сохраняет индекс в каталог: vectors.npy, slugs.json, meta.json."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    if index._matrix is None:
        raise ValueError("index is empty; call build() first")
    np.save(path / VECTORS_FILE, index._matrix)
    (path / SLUGS_FILE).write_text(json.dumps(index.slugs, ensure_ascii=False), encoding="utf-8")
    payload = dict(meta or {})
    payload.update({"backend": index.effective_backend, "count": len(index.slugs), "dim": index.dim})
    (path / META_FILE).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log_marker("M-RETRIEVAL", "save_index", "DONE", dir=str(path), vectors=len(index.slugs))


def load_index(directory: str | Path, backend: str = "numpy") -> VectorIndex:
    """Загружает индекс из каталога.

    Raises:
        FileNotFoundError: если отсутствуют файлы индекса.
    """
    path = Path(directory)
    vectors_file = path / VECTORS_FILE
    slugs_file = path / SLUGS_FILE
    if not vectors_file.is_file() or not slugs_file.is_file():
        raise FileNotFoundError(f"index not found in {path}")
    matrix = np.load(vectors_file).astype(np.float32)
    slugs = json.loads(slugs_file.read_text(encoding="utf-8"))
    index = VectorIndex(backend=backend)
    index.build(slugs, matrix)
    return index
