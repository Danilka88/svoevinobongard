# START_MODULE_CONTRACT
# PURPOSE: Верификация векторного индекса: косинусный Top-K, лимит k, round-trip сохранения, numpy-бэкенд
# SCOPE: test_search_orders_by_cosine, test_k_limit, test_save_load_roundtrip, test_numpy_backend_skips_faiss
# DEPENDS: numpy, services.retrieval.app.index
# LINKS: M-RETRIEVAL
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# test_search_orders_by_cosine - Top-K отсортирован по убыванию косинусного скора
# test_k_limit - k ограничивает длину выдачи
# test_save_load_roundtrip - индекс сохраняется и восстанавливается эквивалентно
# test_numpy_backend_skips_faiss - numpy-бэкенд не поднимает faiss (libomp-конфликт)
# END_MODULE_MAP
"""V-M-RETRIEVAL: индекс и поиск."""
from __future__ import annotations

import numpy as np

from services.retrieval.app.index import VectorIndex, load_index, save_index

__all__ = [
    "test_search_orders_by_cosine",
    "test_k_limit",
    "test_save_load_roundtrip",
    "test_numpy_backend_skips_faiss",
]


def _fixture() -> tuple[list[str], np.ndarray]:
    """Три ортонормированных вектора размерности 4."""
    slugs = ["alpha", "beta", "gamma"]
    vectors = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)
    return slugs, vectors


def test_search_orders_by_cosine() -> None:
    slugs, vectors = _fixture()
    index = VectorIndex(backend="numpy")
    index.build(slugs, vectors)
    hits = index.search(np.array([1.0, 0.1, 0.0, 0.0], dtype=np.float32), k=3)
    assert [slug for slug, _ in hits] == ["alpha", "beta", "gamma"]
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True)

    exact = index.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), k=1)
    assert exact[0][0] == "alpha"
    assert abs(exact[0][1] - 1.0) < 1e-6


def test_k_limit() -> None:
    slugs, vectors = _fixture()
    index = VectorIndex(backend="numpy")
    index.build(slugs, vectors)
    assert len(index.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), k=2)) == 2
    assert len(index.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), k=99)) == 3


def test_save_load_roundtrip(tmp_path) -> None:
    slugs, vectors = _fixture()
    index = VectorIndex(backend="numpy")
    index.build(slugs, vectors)
    save_index(tmp_path, index, {"encoder": "test"})
    restored = load_index(tmp_path, backend="numpy")
    assert restored.slugs == slugs
    assert restored.dim == 4
    top = restored.search(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32), k=1)
    assert top[0][0] == "beta"
    assert abs(top[0][1] - 1.0) < 1e-6


def test_numpy_backend_skips_faiss() -> None:
    slugs, vectors = _fixture()
    index = VectorIndex(backend="numpy")
    index.build(slugs, vectors)
    assert index.effective_backend == "numpy"
    assert index._faiss_index is None
