# START_MODULE_CONTRACT
# PURPOSE: Верификация проекционной головы домен-адаптации: нормировка выхода, round-trip весов, применение к векторам
# SCOPE: test_head_output_is_normalized, test_head_save_load_roundtrip, test_apply_head_matches_module, test_head_meta_description
# DEPENDS: torch, numpy, services.retrieval.app.head
# LINKS: M-RETRIEVAL
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# test_head_output_is_normalized - выход головы L2-нормирован и имеет нужную размерность
# test_head_save_load_roundtrip - веса сохраняются и восстанавливаются без потери результата
# test_apply_head_matches_module - apply_head совпадает с прямым вызовом модуля
# test_head_meta_description - метаданные и описание головы читаются из meta.json
# END_MODULE_MAP
"""V-M-RETRIEVAL: проекционная голова домен-адаптации."""
from __future__ import annotations

import numpy as np
import torch

from services.retrieval.app.head import (
    ProjectionHead,
    apply_head,
    describe_head,
    head_meta,
    load_head,
    read_head_meta,
    save_head,
)

__all__ = [
    "test_head_output_is_normalized",
    "test_head_save_load_roundtrip",
    "test_apply_head_matches_module",
    "test_head_meta_description",
]


def _sample_vectors(count: int = 8, dim: int = 32, seed: int = 0) -> np.ndarray:
    """L2-нормированные случайные векторы."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(count, dim)).astype(np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def test_head_output_is_normalized() -> None:
    head = ProjectionHead(in_dim=32, hidden=16, out_dim=8)
    output = head(torch.from_numpy(_sample_vectors()))
    assert tuple(output.shape) == (8, 8)
    norms = output.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_head_save_load_roundtrip(tmp_path) -> None:
    head = ProjectionHead(in_dim=32, hidden=16, out_dim=8)
    vectors = _sample_vectors()
    expected = head(torch.from_numpy(vectors)).detach().numpy()
    save_head(tmp_path / "head.pt", head, {"pairs": 3})

    restored = load_head(tmp_path / "head.pt")
    actual = restored(torch.from_numpy(vectors)).detach().numpy()
    assert np.allclose(expected, actual, atol=1e-6)


def test_apply_head_matches_module(tmp_path) -> None:
    head = ProjectionHead(in_dim=32, hidden=16, out_dim=8)
    vectors = _sample_vectors()
    via_helper = apply_head(head, vectors)
    via_module = head(torch.from_numpy(vectors)).detach().numpy()
    assert np.allclose(via_helper, via_module, atol=1e-6)
    assert np.allclose(apply_head(None, vectors), vectors)


def test_head_meta_description() -> None:
    head = ProjectionHead(in_dim=32, hidden=16, out_dim=8)
    meta = {"head": head_meta(head, {"pairs": 3})}
    assert read_head_meta(meta)["out_dim"] == 8
    assert describe_head(meta) == "32->16->8"
    assert describe_head({}) == "none"
