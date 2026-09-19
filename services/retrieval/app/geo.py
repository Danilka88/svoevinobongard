# START_MODULE_CONTRACT
# PURPOSE: Геоверификация Top-K локальными признаками: OpenCV ORB/SIFT, BFMatcher с ratio-test, RANSAC-гомография
# SCOPE: GeoFeatures, GeoResult, extract_features, verify, fuse_scores, GeoIndex, resolve_geo_method
# DEPENDS: numpy, cv2, json, packages.common.logging
# LINKS: M-RETRIEVAL
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# GeoFeatures - ключевые точки и дескрипторы изображения
# GeoResult - результат верификации: matches, inliers, inlier_ratio, score
# resolve_geo_method - выбор метода признаков (orb | sift | auto)
# extract_features - извлечение локальных признаков изображения
# verify - верификация пары изображений через RANSAC-гомографию
# fuse_scores - фьюжн retrieval-скора и geo-скора
# GeoIndex - предвычисленные признаки эталонов с поиском скора по кандидатам
# END_MODULE_MAP
"""Геоверификация second-stage.

Эмбеддинг различает «похожие этикетки» плохо: near-duplicates (год, сладость,
серия) дают близкие векторы. Локальные признаки и геометрия этикетки различают
их точнее, поэтому Top-K переранжируется по числу RANSAC-инлайеров.

Метод по умолчанию — ORB (компактные бинарные дескрипторы, десятки МБ на 2037
эталонов). SIFT доступен через GEO_METHOD=sift. Kornia SuperPoint+LightGlue
рассматривался как опция, но в окружении отсутствует (нет пакета и весов),
поэтому основной безопасный путь — OpenCV (Apache-2.0).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from packages.common.logging import log_marker

__all__ = ["GeoFeatures", "GeoResult", "resolve_geo_method", "extract_features",
           "verify", "fuse_scores", "GeoIndex"]

FEATURES_FILE = "features.npz"
DEFAULT_MAX_KEYPOINTS = 300
RATIO_TEST = 0.75
RANSAC_REPROJ = 5.0


@dataclass(frozen=True)
class GeoFeatures:
    """Локальные признаки изображения."""

    keypoints: np.ndarray
    descriptors: np.ndarray


@dataclass(frozen=True)
class GeoResult:
    """Результат геоверификации пары изображений."""

    matches: int
    inliers: int
    inlier_ratio: float
    score: float


def resolve_geo_method(requested: str = "auto") -> str:
    """Определяет метод локальных признаков.

    Args:
        requested: orb | sift | auto.

    Returns:
        'orb' или 'sift'.
    """
    if requested in {"orb", "sift"}:
        return requested
    return "orb"


def _make_detector(method: str, max_keypoints: int):
    """Создает детектор OpenCV для выбранного метода."""
    if method == "sift":
        return cv2.SIFT_create(nfeatures=max_keypoints)
    return cv2.ORB_create(nfeatures=max_keypoints)


def extract_features(
    image: np.ndarray,
    method: str = "orb",
    max_keypoints: int = DEFAULT_MAX_KEYPOINTS,
) -> GeoFeatures:
    """Извлекает ключевые точки и дескрипторы.

    Args:
        image: BGR-изображение.
        method: orb | sift.
        max_keypoints: ограничение числа точек (компактность артефакта).

    Returns:
        GeoFeatures; при отсутствии точек массивы пустые.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    detector = _make_detector(method, max_keypoints)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) == 0:
        dim = 128 if method == "sift" else 32
        return GeoFeatures(np.zeros((0, 2), dtype=np.float32),
                           np.zeros((0, dim), dtype=np.uint8 if method == "orb" else np.float32))
    points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    return GeoFeatures(points, descriptors)


def verify(
    query: GeoFeatures,
    reference: GeoFeatures,
    method: str = "orb",
    min_matches: int = 8,
) -> GeoResult:
    """Верифицирует пару изображений через RANSAC-гомографию.

    Args:
        query: признаки запроса.
        reference: признаки эталона.
        method: orb | sift.
        min_matches: минимум хороших матчей для попытки RANSAC.

    Returns:
        GeoResult; score в диапазоне [0, 1] — доля инлайеров.
    """
    if len(query.keypoints) < 4 or len(reference.keypoints) < 4:
        return GeoResult(0, 0, 0.0, 0.0)

    norm = cv2.NORM_HAMMING if method == "orb" else cv2.NORM_L2
    query_desc = query.descriptors.astype(np.float32) if method == "sift" else query.descriptors
    ref_desc = reference.descriptors.astype(np.float32) if method == "sift" else reference.descriptors
    matcher = cv2.BFMatcher(norm)
    try:
        knn = matcher.knnMatch(query_desc, ref_desc, k=2)
    except cv2.error:
        return GeoResult(0, 0, 0.0, 0.0)

    good = [pair[0] for pair in knn if len(pair) == 2 and pair[0].distance < RATIO_TEST * pair[1].distance]
    if len(good) < min_matches:
        return GeoResult(len(good), 0, 0.0, 0.0)

    source = np.float32([query.keypoints[m.queryIdx] for m in good]).reshape(-1, 1, 2)
    target = np.float32([reference.keypoints[m.trainIdx] for m in good]).reshape(-1, 1, 2)
    try:
        _, mask = cv2.findHomography(source, target, cv2.RANSAC, RANSAC_REPROJ)
    except cv2.error:
        return GeoResult(len(good), 0, 0.0, 0.0)
    if mask is None:
        return GeoResult(len(good), 0, 0.0, 0.0)

    inliers = int(mask.sum())
    denominator = float(max(1, min(len(query.keypoints), len(reference.keypoints))))
    ratio = min(1.0, inliers / denominator)
    return GeoResult(len(good), inliers, ratio, ratio)


def fuse_scores(retrieval_score: float, geo_score: float, weight: float) -> float:
    """Линейный фьюжн retrieval-скора и geo-скора.

    Args:
        retrieval_score: косинусный скор эмбеддинга.
        geo_score: доля RANSAC-инлайеров.
        weight: вклад геоверификации.

    Returns:
        Итоговый скор.
    """
    return float(retrieval_score + weight * geo_score)


class GeoIndex:
    """Предвычисленные локальные признаки эталонов."""

    def __init__(
        self,
        slugs: list[str],
        keypoints: np.ndarray,
        descriptors: np.ndarray,
        counts: np.ndarray,
        method: str = "orb",
    ) -> None:
        self.slugs = list(slugs)
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.counts = counts
        self.method = method
        self._positions: dict[str, list[int]] = {}
        for index, slug in enumerate(self.slugs):
            self._positions.setdefault(slug, []).append(index)
        self._position = {slug: positions[0] for slug, positions in self._positions.items()}

    @classmethod
    def load(cls, directory: str | Path, meta: dict) -> "GeoIndex | None":
        """Загружает признаки из features.npz, если они есть и выровнены по индексу."""
        payload = (meta or {}).get("features") or {}
        if not payload:
            return None
        path = Path(directory) / payload.get("file", FEATURES_FILE)
        slugs_path = Path(directory) / "slugs.json"
        if not path.is_file() or not slugs_path.is_file():
            return None
        try:
            data = np.load(path)
            slugs = json.loads(slugs_path.read_text(encoding="utf-8"))
            if len(slugs) != int(data["counts"].shape[0]):
                log_marker("M-RETRIEVAL", "geo_load", "SIZE_MISMATCH",
                           slugs=len(slugs), features=int(data["counts"].shape[0]))
                return None
            index = cls(slugs, data["keypoints"], data["descriptors"], data["counts"],
                        str(payload.get("method", "orb")))
        except Exception as error:  # noqa: BLE001 - поврежденный артефакт не должен валить сервис
            log_marker("M-RETRIEVAL", "geo_load", "LOAD_FAILED", error=type(error).__name__)
            return None
        log_marker("M-RETRIEVAL", "geo_load", "READY", features=len(index.slugs), method=index.method)
        return index

    def features_for(self, slug: str) -> GeoFeatures | None:
        """Возвращает признаки эталона по slug."""
        position = self._position.get(slug)
        if position is None:
            return None
        count = int(self.counts[position])
        return GeoFeatures(self.keypoints[position][:count], self.descriptors[position][:count])

    def verify_all_views(self, slug: str, query: GeoFeatures,
                         min_matches: int = 8) -> GeoResult:
        """Проверяет все виды slug и берёт лучший результат.

        Мультивью-галерея хранит несколько наборов признаков на позицию; проверка одного
        произвольного вида (последнего в массиве) теряла остальные ракурсы.
        """
        best: GeoResult | None = None
        for position in self._positions.get(slug, []):
            reference = GeoFeatures(self.keypoints[position], self.descriptors[position],
                                    int(self.counts[position]))
            result = verify(query, reference, method=self.method, min_matches=min_matches)
            if best is None or result.score > best.score:
                best = result
        if best is None:
            return GeoResult(score=0.0, inliers=0, homography=None)
        return best

    def verify_slug(self, slug: str, query: GeoFeatures, min_matches: int = 8) -> GeoResult:
        """Верифицирует запрос против эталона по slug."""
        reference = self.features_for(slug)
        if reference is None:
            return GeoResult(0, 0, 0.0, 0.0)
        return verify(query, reference, method=self.method, min_matches=min_matches)
