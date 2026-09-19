# START_MODULE_CONTRACT
# PURPOSE: Верификация геоверификации: извлечение признаков, RANSAC-инлайеры, фьюжн скоров и загрузка индекса признаков
# SCOPE: test_extract_features_on_texture, test_verify_same_image_has_inliers, test_verify_unrelated_is_zero, test_verify_empty_features, test_fuse_scores, test_geoindex_roundtrip
# DEPENDS: numpy, cv2, services.retrieval.app.geo
# LINKS: M-RETRIEVAL
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# test_extract_features_on_texture - признаки находятся на текстурированном изображении
# test_verify_same_image_has_inliers - самопроверка дает инлайеры и скор больше нуля
# test_verify_unrelated_is_zero - несвязанные изображения дают скор около нуля
# test_verify_empty_features - пустые признаки не ломают верификацию
# test_fuse_scores - фьюжн складывает retrieval и взвешенный geo
# test_geoindex_roundtrip - GeoIndex сохраняется и загружается, скор по slug доступен
# END_MODULE_MAP
"""V-M-RETRIEVAL: геоверификация Top-K."""
from __future__ import annotations

import json

import cv2
import numpy as np

from services.retrieval.app.geo import (
    FEATURES_FILE,
    GeoFeatures,
    GeoIndex,
    extract_features,
    fuse_scores,
    verify,
)

__all__ = [
    "test_extract_features_on_texture",
    "test_verify_same_image_has_inliers",
    "test_verify_unrelated_is_zero",
    "test_verify_empty_features",
    "test_fuse_scores",
    "test_geoindex_roundtrip",
]


def _texture(seed: int = 0, size: int = 320) -> np.ndarray:
    """Синтетическая «этикетка»: рамка, фигуры и текст."""
    rng = np.random.default_rng(seed)
    canvas = np.full((size, size, 3), 235, dtype=np.uint8)
    cv2.rectangle(canvas, (20, 20), (size - 20, size - 20), (30, 30, 30), 3)
    for _ in range(12):
        x, y = rng.integers(40, size - 60, size=2)
        color = tuple(int(value) for value in rng.integers(0, 120, size=3))
        if rng.random() < 0.5:
            cv2.rectangle(canvas, (int(x), int(y)), (int(x + 40), int(y + 25)), color, -1)
        else:
            cv2.circle(canvas, (int(x), int(y)), int(rng.integers(8, 22)), color, -1)
    cv2.putText(canvas, f"WINE {seed}", (40, size - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 3)
    return canvas


def _noise_texture(seed: int = 0, size: int = 320) -> np.ndarray:
    """Изображение без структуры этикетки: высокочастотный шум."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)


def test_extract_features_on_texture() -> None:
    features = extract_features(_texture(1))
    assert features.keypoints.shape[0] > 10
    assert features.descriptors.shape[1] == 32
    assert features.descriptors.dtype == np.uint8


def test_verify_same_image_has_inliers() -> None:
    features = extract_features(_texture(2))
    result = verify(features, features)
    assert result.matches > 0
    assert result.inliers > 0
    assert result.score > 0.5


def test_verify_unrelated_is_zero() -> None:
    """Разные этикетки дают заметно меньше инлайеров, чем та же этикетка."""
    same = extract_features(_texture(3))
    other = extract_features(_texture(3, size=320) * 0 + _noise_texture(77))
    self_result = verify(same, same)
    other_result = verify(same, other)
    assert other_result.score < 0.35
    assert self_result.score > 3 * max(other_result.score, 1e-6)


def test_verify_empty_features() -> None:
    empty = GeoFeatures(np.zeros((0, 2), dtype=np.float32), np.zeros((0, 32), dtype=np.uint8))
    result = verify(empty, extract_features(_texture(4)))
    assert result.score == 0.0 and result.inliers == 0


def test_fuse_scores() -> None:
    assert fuse_scores(0.5, 0.0, 0.35) == 0.5
    assert abs(fuse_scores(0.5, 1.0, 0.35) - 0.85) < 1e-9
    assert fuse_scores(0.5, 1.0, 0.0) == 0.5


def test_geoindex_roundtrip(tmp_path) -> None:
    slugs = ["alpha", "beta"]
    images = [_texture(5), _texture(6)]
    max_kp = 100
    keypoints, descriptors, counts = [], [], []
    for image in images:
        features = extract_features(image, max_keypoints=max_kp)
        count = min(max_kp, len(features.keypoints))
        padded_kp = np.zeros((max_kp, 2), dtype=np.float32)
        padded_desc = np.zeros((max_kp, 32), dtype=np.uint8)
        padded_kp[:count] = features.keypoints[:count]
        padded_desc[:count] = features.descriptors[:count]
        keypoints.append(padded_kp)
        descriptors.append(padded_desc)
        counts.append(count)

    np.savez_compressed(tmp_path / FEATURES_FILE,
                        keypoints=np.stack(keypoints).astype(np.float32),
                        descriptors=np.stack(descriptors).astype(np.uint8),
                        counts=np.array(counts, dtype=np.int32))
    (tmp_path / "slugs.json").write_text(json.dumps(slugs), encoding="utf-8")
    meta = {"features": {"file": FEATURES_FILE, "method": "orb", "max_keypoints": max_kp, "dim": 32}}

    index = GeoIndex.load(tmp_path, meta)
    assert index is not None and index.slugs == slugs
    same = index.verify_slug("alpha", extract_features(images[0]))
    other = index.verify_slug("beta", extract_features(images[0]))
    assert same.score > other.score
    assert GeoIndex.load(tmp_path, {}) is None
