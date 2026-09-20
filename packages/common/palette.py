# START_MODULE_CONTRACT
# PURPOSE: Палитры этикеток в CIELAB и цветовой tie-breaker для близких визуальных скоров
# SCOPE: DEFAULT_K, dominant_palette, palette_delta_e, palette_similarity, load_palettes,
#        save_palettes, build_palettes, tiebreak_order
# DEPENDS: numpy, cv2, json, pathlib, packages.common.logging
# LINKS: M-RETRIEVAL, M-EVAL
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# DEFAULT_K - число доминирующих цветов в палитре
# MAX_SIDE - длинная сторона при вычислении палитры
# TIEBREAK_MARGIN - ширина зоны близких викторов, где включается цвет
# TIEBREAK_WEIGHT - вес цветового бонуса
# dominant_palette - доминирующие цвета кадра в CIELAB через k-means
# palette_delta_e - минимальное ΔE по парам доминирующих цветов двух палитр
# palette_similarity - близость палитр в диапазоне 0..1 (1 — совпадение)
# load_palettes - чтение артефакта палитр
# save_palettes - запись артефакта палитр
# build_palettes - офлайн-сборка палитр эталонов с возобновлением
# tiebreak_order - перестановка кандидатов в tie-зоне по близости палитр
# END_MODULE_MAP
"""Палитры этикеток и цветовой tie-breaker.

Цвет не является самостоятельным сигналом: у 31% красных купажей палитры визуально
неотличимы (ΔE ≤ 3.5), поэтому сигнал применяется только как разрыв ties — когда
визуальные скоры top-1 и top-2 близки и эмбеддинг не даёт уверенного ответа.

Все вычисления идут в CIELAB: расстояние в нём ближе к восприятию, чем в BGR.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from packages.common.logging import log_marker

DEFAULT_K = 5
MAX_SIDE = 256
TIEBREAK_MARGIN = 0.05
TIEBREAK_WEIGHT = 0.02

__all__ = ["DEFAULT_K", "MAX_SIDE", "TIEBREAK_MARGIN", "TIEBREAK_WEIGHT", "build_palettes",
           "dominant_palette", "load_palettes", "palette_delta_e", "palette_similarity",
           "save_palettes", "tiebreak_order"]


def dominant_palette(image_bgr: np.ndarray, k: int = DEFAULT_K) -> np.ndarray | None:
    """Считает доминирующие цвета кадра в CIELAB.

    Args:
        image_bgr: кадр или кроп в BGR.
        k: число доминирующих цветов.

    Returns:
        Массив (k, 3) float32 в CIELAB или None, если кадр непригоден.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None
    height, width = image_bgr.shape[:2]
    scale = MAX_SIDE / max(height, width)
    prepared = cv2.resize(image_bgr, (max(1, int(width * scale)), max(1, int(height * scale))),
                          interpolation=cv2.INTER_AREA) if scale < 1.0 else image_bgr
    lab = cv2.cvtColor(prepared, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    if len(lab) < k:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 0.5)
    _, labels, centers = cv2.kmeans(lab, k, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    # OpenCV хранит L в 0..255: приводим к 0..100, чтобы ΔE был сопоставим с LAB-шкалой
    centers = centers.copy()
    centers[:, 0] = centers[:, 0] * (100.0 / 255.0)
    centers[:, 1] = centers[:, 1] - 128.0
    centers[:, 2] = centers[:, 2] - 128.0
    order = np.argsort(-np.bincount(labels.flatten(), minlength=k))
    return centers[order].astype(np.float32)


def palette_delta_e(left: np.ndarray | None, right: np.ndarray | None) -> float:
    """Минимальное ΔE76 по парам доминирующих цветов двух палитр.

    Минимум, а не среднее: этикетка может занимать малую часть кадра, и достаточно
    одного совпадающего доминирующего цвета, чтобы палитры считались близкими.
    """
    if left is None or right is None or len(left) == 0 or len(right) == 0:
        return 100.0
    diff = left[:, None, :] - right[None, :, :]
    return float(np.sqrt((diff ** 2).sum(axis=2)).min())


def palette_similarity(left: np.ndarray | None, right: np.ndarray | None,
                       scale: float = 40.0) -> float:
    """Близость палитр 0..1: 1 при совпадении, 0 при ΔE больше scale."""
    delta = palette_delta_e(left, right)
    return max(0.0, 1.0 - delta / scale)


def load_palettes(path: str | Path) -> dict[str, np.ndarray]:
    """Читает артефакт палитр (JSON со списками цветов)."""
    target = Path(path)
    if not target.is_file():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {slug: np.asarray(colors, dtype=np.float32) for slug, colors in payload.items()}


def save_palettes(path: str | Path, palettes: dict[str, np.ndarray]) -> None:
    """Сохраняет палитры в JSON (компактно: округление до двух знаков)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {slug: np.round(np.asarray(colors, dtype=np.float32), 2).tolist()
               for slug, colors in palettes.items()}
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def build_palettes(slug_to_image: dict[str, str], uploads: str | Path,
                   out: str | Path, k: int = DEFAULT_K, resume: bool = True,
                   limit: int = 0) -> dict[str, np.ndarray]:
    """Считает палитры эталонов офлайн, продолжая с места остановки.

    Args:
        slug_to_image: соответствие slug -> имя файла эталона.
        uploads: каталог эталонных изображений.
        out: путь артефакта (data/, gitignored).
        resume: не пересчитывать уже сохранённые позиции.
        limit: ограничить число позиций (0 — все).

    Returns:
        Словарь ``slug -> палитра``.
    """
    palettes = load_palettes(out) if resume else {}
    upload_dir = Path(uploads)
    pending = [(slug, name) for slug, name in sorted(slug_to_image.items()) if slug not in palettes]
    if limit:
        pending = pending[:limit]
    processed = 0
    for slug, name in pending:
        try:
            with open(upload_dir / name, "rb") as handle:
                buffer = np.frombuffer(handle.read(), dtype=np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        except Exception:  # noqa: BLE001 - битый файл не должен валить сборку
            image = None
        palette = dominant_palette(image, k) if image is not None else None
        if palette is not None:
            palettes[slug] = palette
        processed += 1
        if processed % 250 == 0:
            print(f"  палитры: {processed}/{len(pending)}", flush=True)
    save_palettes(out, palettes)
    print(f"палитр: {len(palettes)} из {len(slug_to_image)} (посчитано сейчас {processed})")
    log_marker("M-RETRIEVAL", "palettes", "DONE", palettes=len(palettes), processed=processed)
    return palettes


def tiebreak_order(candidates: list[tuple[str, float]], query_palette: np.ndarray | None,
                   palettes: dict[str, np.ndarray], *, margin: float = TIEBREAK_MARGIN,
                   weight: float = TIEBREAK_WEIGHT) -> list[str]:
    """Переставляет кандидатов в tie-зоне по близости палитр.

    Если разрыв между лучшими двумя скорами больше ``margin``, порядок возвращается без
    изменений: вне зоны близких скоров цвет не влияет. Внутри зоны к скору добавляется
    ``weight * palette_similarity(запрос, кандидат)``, что может поменять местами соседей.
    """
    ordered = sorted(candidates, key=lambda item: item[1], reverse=True)
    if len(ordered) < 2 or query_palette is None:
        return [slug for slug, _ in ordered]
    if ordered[0][1] - ordered[1][1] > margin:
        return [slug for slug, _ in ordered]
    boosted: list[tuple[str, float]] = []
    for slug, score in ordered:
        palette = palettes.get(slug)
        bonus = weight * palette_similarity(query_palette, palette) if palette is not None else 0.0
        boosted.append((slug, score + bonus))
    boosted.sort(key=lambda item: item[1], reverse=True)
    return [slug for slug, _ in boosted]
