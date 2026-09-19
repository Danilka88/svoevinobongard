#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Зонд реальных фото: распределение попаданий по глубине top-K и проверка гипотез улучшения
# SCOPE: load_cases, encode_query, rank_of_identity, containment, hypothesis_run, main
# DEPENDS: numpy, PIL, cv2, packages.common.catalog, packages.common.config, packages.common.images,
#          packages.common.ocr, packages.common.preprocess, scripts.eval.real_field_eval,
#          services.retrieval.app.encoder, services.retrieval.app.head, services.retrieval.app.index,
#          services.retrieval.app.search, services.retrieval.app.winery
# LINKS: M-EVAL, M-RETRIEVAL, M-NORMALIZE
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# DEPTHS - глубины top-K, на которых проверяется наличие правильной идентичности
# DEFAULT_DIR - каталог реальных фото
# EXTRA_DIR - каталог официальных фото, участвующих в том же прогоне
# IDENTITY_THRESHOLD - порог fuzzy-совпадения для слабой GT
# DEFAULT_DIR - каталог реальных фото
# load_cases - загрузка фото с разобранной слабой GT и кандидатами каталога
# encode_query - кодирование запроса с головой
# rank_of_identity - ранг первой правильной идентичности в выдаче
# containment - распределение попаданий по глубине
# hypothesis_run - метрики identity@1/@5 и latency для набора вариантов запроса
# main - режимы containment и hypotheses
# END_MODULE_MAP
"""Зонд реальных полевых фото.

Отвечает на два вопроса. Первый: на какой глубине выдачи вообще находится правильная
идентичность — если её нет и в top-100, проблема в представлении, а не в ранжировании.
Второй: что из дешёвых приёмов реально помогает на магазинных фото (кроп этикетки,
удаление фона, прототипы виноделен, разрешение, CLAHE/WB) — каждый вариант измеряется
теми же метриками, что и базовый.

Все замеры идут в процессе, без HTTP: так доступны глубины top-K, недостижимые через
контракт /v1/scan, и не нужен запуск сервисов.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from packages.common.catalog import CatalogStore, load_catalog  # noqa: E402
from packages.common.config import settings  # noqa: E402
from packages.common.images import flatten_alpha  # noqa: E402
from packages.common.logging import log_marker  # noqa: E402
from packages.common.ocr import run_ocr  # noqa: E402
from packages.common.preprocess import PreprocessConfig, enhance_clahe_wb  # noqa: E402
from scripts.eval.real_field_eval import identity_candidates, parse_label  # noqa: E402
from services.normalize.app.tta import build_variants  # noqa: E402
from services.retrieval.app.encoder import load_encoder  # noqa: E402
from services.retrieval.app.head import apply_head, load_head  # noqa: E402
from services.retrieval.app.index import load_index  # noqa: E402
from services.retrieval.app.search import dedupe_by_slug, search_images  # noqa: E402
from services.retrieval.app.winery import boost, build_prototypes  # noqa: E402

DEPTHS = (1, 5, 10, 30, 50, 100)
DEFAULT_DIR = "data/samples/real_field"
EXTRA_DIR = "data/samples/eval_queries"
IDENTITY_THRESHOLD = 72.0

__all__ = ["DEPTHS", "DEFAULT_DIR", "containment", "encode_query", "hypothesis_run",
           "load_cases", "main", "rank_of_identity"]


def load_cases(directory: Path, catalog: dict) -> list[dict]:
    """Загружает фото с разобранной слабой GT и набором кандидатов каталога."""
    manifest = directory / "manifest.tsv"
    labels: dict[str, str] = {}
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                name, _, text = line.partition("\t")
                labels[name] = text
    cases: list[dict] = []
    for path in sorted(directory.glob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        labels.setdefault(path.name, path.stem.replace("_", " ").strip())
    extra = _ROOT / EXTRA_DIR
    if extra.is_dir():
        for path in sorted(extra.glob("*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                labels.setdefault(path.name, path.stem)
    for name, text in labels.items():
        path = directory / name if (directory / name).is_file() else extra / name
        if not path.is_file():
            continue
        try:
            with Image.open(path) as raw:
                image = flatten_alpha(np.asarray(raw.convert("RGBA"))[:, :, [2, 1, 0, 3]])
        except Exception:  # noqa: BLE001
            continue
        label = parse_label(text)
        label["candidates"] = identity_candidates(label, catalog, IDENTITY_THRESHOLD)
        cases.append({"file": name, "image": image, "label": label})
    return cases


def encode_query(encoder, head, image: np.ndarray) -> np.ndarray:
    """Кодирует один кадр с применением проекционной головы."""
    vector = encoder.encode_arrays([image])
    if head is not None:
        vector = apply_head(head, vector, device="cpu")
    return np.asarray(vector, dtype=np.float32)[0]


def rank_of_identity(ranked: list[str], candidates: list[str]) -> int:
    """Ранг первой правильной идентичности; 0, если её нет в выдаче."""
    pool = set(candidates)
    for position, slug in enumerate(ranked, 1):
        if slug in pool:
            return position
    return 0


def _search_ranked(encoder, head, index, image: np.ndarray, limit: int) -> list[str]:
    """Ранжирует каталог по одному кадру: пул строк, схлопывание по slug, обрезка."""
    vector = encode_query(encoder, head, image)
    pool = max(limit * 10, 200)
    return [slug for slug, _ in dedupe_by_slug(index.search(vector, k=pool), limit)]


def containment(cases: list[dict], encoder, head, index, limit: int = 100) -> dict:
    """Считает, на какой глубине находится правильная идентичность."""
    rows: list[dict] = []
    for case in cases:
        candidates = case["label"]["candidates"]
        if not candidates:
            rows.append({"file": case["file"], "rank": 0, "out_of_catalog": case["label"]["non_russian"]})
            continue
        ranked = _search_ranked(encoder, head, index, case["image"], limit)
        rows.append({"file": case["file"], "rank": rank_of_identity(ranked, candidates),
                     "out_of_catalog": False, "top1": ranked[0] if ranked else None})
    with_identity = [row for row in rows if not row["out_of_catalog"]]
    distribution = {f"at_{depth}": round(sum(1 for row in with_identity if 0 < row["rank"] <= depth)
                                         / len(with_identity), 4) if with_identity else 0.0
                    for depth in DEPTHS}
    missed_100 = [row["file"] for row in with_identity if row["rank"] == 0 or row["rank"] > 100]
    return {"cases": len(rows), "with_identity": len(with_identity),
            "out_of_catalog": len(rows) - len(with_identity),
            "distribution": distribution, "not_in_top_100": missed_100, "rows": rows}


def _label_crop(image: np.ndarray) -> np.ndarray | None:
    """Кроп этикетки по объединению OCR-боксов (эвристика без детектора).

    Элемент выдачи RapidOCR — ``[box, text, score]``, где box это четыре точки [x, y].
    """
    try:
        import rapidocr_onnxruntime  # noqa: PLC0415

        engine = _rapid_engine.setdefault("engine", rapidocr_onnxruntime.RapidOCR())
        boxes, _ = _rapid_engine["engine"](image)
    except Exception:  # noqa: BLE001
        return None
    if not boxes:
        return None
    xs: list[float] = []
    ys: list[float] = []
    for item in boxes:
        box = item[0] if isinstance(item, (list, tuple)) and len(item) >= 1 else item
        try:
            for point in box:
                xs.append(float(point[0]))
                ys.append(float(point[1]))
        except (TypeError, ValueError, IndexError):
            continue
    if len(xs) < 8:
        return None
    x0, x1 = int(max(0, min(xs))), int(min(image.shape[1], max(xs)))
    y0, y1 = int(max(0, min(ys))), int(min(image.shape[0], max(ys)))
    if x1 - x0 < 24 or y1 - y0 < 24:
        return None
    pad_x, pad_y = int(0.06 * (x1 - x0)), int(0.06 * (y1 - y0))
    return image[max(0, y0 - pad_y):min(image.shape[0], y1 + pad_y),
                 max(0, x0 - pad_x):min(image.shape[1], x1 + pad_x)]


_rapid_engine: dict = {}


def _remove_background(image: np.ndarray) -> np.ndarray | None:
    """Удаляет фон GrabCut-ом и возвращает объект на белом фоне."""
    mask = np.zeros(image.shape[:2], np.uint8)
    rect = (int(image.shape[1] * 0.12), int(image.shape[0] * 0.12),
            int(image.shape[1] * 0.76), int(image.shape[0] * 0.76))
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
    except Exception:  # noqa: BLE001
        return None
    foreground = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    if foreground.sum() < 0.02 * 255 * foreground.size:
        return None
    white = np.full_like(image, 245)
    return np.where(foreground[:, :, None] > 0, image, white)


def hypothesis_run(cases: list[dict], encoder, head, index, catalog: dict,
                   store: CatalogStore, prototypes) -> dict:
    """Считает identity@1/@5 и latency для набора вариантов запроса."""
    variants: dict[str, dict] = {}

    def score(name: str, builder, extra=None) -> None:
        latencies: list[float] = []
        at1 = at5 = total = 0
        for case in cases:
            candidates = case["label"]["candidates"]
            if not candidates:
                continue
            started = time.perf_counter()
            image = builder(case)
            if image is None:
                continue
            ranked = _search_ranked(encoder, head, index, image, 5)
            latencies.append((time.perf_counter() - started) * 1000.0)
            total += 1
            rank = rank_of_identity(ranked, candidates)
            at1 += int(rank == 1)
            at5 += int(0 < rank <= 5)
        variants[name] = {
            "identity_at_1": round(at1 / total, 4) if total else 0.0,
            "identity_at_5": round(at5 / total, 4) if total else 0.0,
            "cases": total,
            "latency_p50": round(sorted(latencies)[len(latencies) // 2], 1) if latencies else 0.0,
            **(extra or {}),
        }
        print(f"  {name:<28} identity@1 {variants[name]['identity_at_1']:<7} @5 {variants[name]['identity_at_5']:<7} "
              f"p50 {variants[name]['latency_p50']:>7} мс", flush=True)

    def or_self(transform):
        """Возвращает исходный кадр, если преобразование не сработало (None)."""
        def inner(case):
            result = transform(case["image"])
            return case["image"] if result is None else result
        return inner

    score("baseline_dedup", lambda case: case["image"])
    score("h_label_crop", or_self(_label_crop))
    score("h_bg_grabcut", or_self(_remove_background))
    score("h_norm_clahe_wb", lambda case: enhance_clahe_wb(case["image"]))
    score("h_res_768", lambda case: _resize(case["image"], 768))
    score("h_res_1024", lambda case: _resize(case["image"], 1024))
    score("h_variants_lastframe", lambda case: build_variants(case["image"], None)[-1])
    if len(prototypes):
        # H-brand: буст кандидатов из виноделен, близких к запросу (флаг по умолчанию выключен)
        position = {slug: index for index, slug in enumerate(index.slugs)}
        latencies, at1, at5, total = [], 0, 0, 0
        for case in cases:
            candidates_gt = case["label"]["candidates"]
            if not candidates_gt:
                continue
            started = time.perf_counter()
            vector = encode_query(encoder, head, case["image"])
            rows = dedupe_by_slug(index.search(vector, k=200), 100)
            ordered = boost([{"slug": slug, "score": score_value, "index": position.get(slug, -1)}
                             for slug, score_value in rows], prototypes, vector, weight=0.15, top_wineries=5)
            latencies.append((time.perf_counter() - started) * 1000.0)
            total += 1
            rank = rank_of_identity([entry["slug"] for entry in ordered][:5], candidates_gt)
            at1 += int(rank == 1)
            at5 += int(0 < rank <= 5)
        variants["h_brand_proto"] = {
            "identity_at_1": round(at1 / total, 4) if total else 0.0,
            "identity_at_5": round(at5 / total, 4) if total else 0.0,
            "cases": total,
            "latency_p50": round(sorted(latencies)[len(latencies) // 2], 1) if latencies else 0.0,
            "note": f"прототипов виноделен: {len(prototypes)}",
        }
        print(f"  {'h_brand_proto':<28} identity@1 {variants['h_brand_proto']['identity_at_1']:<7} "
              f"@5 {variants['h_brand_proto']['identity_at_5']:<7} p50 {variants['h_brand_proto']['latency_p50']:>7} мс", flush=True)
    return variants


def _resize(image: np.ndarray, max_side: int) -> np.ndarray:
    """Ресайз кадра по длинной стороне."""
    height, width = image.shape[:2]
    scale = max_side / max(height, width)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))))


def main(argv: list[str] | None = None) -> int:
    """Запускает containment или проверку гипотез."""
    parser = argparse.ArgumentParser(description="Зонд реальных полевых фото")
    parser.add_argument("mode", choices=["containment", "hypotheses", "all"])
    parser.add_argument("--dir", default=str(_ROOT / DEFAULT_DIR))
    parser.add_argument("--out", default=str(_ROOT / "reports" / "real_field_probe.json"))
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args(argv)

    catalog = load_catalog(_ROOT / settings.catalog_jsonl)
    store = CatalogStore(catalog)
    cases = load_cases(Path(args.dir), catalog)
    meta = json.loads((_ROOT / settings.index_dir / "meta.json").read_text(encoding="utf-8"))
    encoder = load_encoder(encoder=meta.get("encoder"), model_id=meta.get("model"), device=meta.get("device"))
    index = load_index(_ROOT / settings.index_dir, backend=settings.vector_backend)
    head_path = _ROOT / settings.index_dir / "head.pt"
    head = load_head(head_path) if head_path.is_file() else None
    print(f"фото: {len(cases)} · фото с идентичностью в GT: {sum(1 for c in cases if c['label']['candidates'])}")

    out_path = Path(args.out)
    report = json.loads(out_path.read_text(encoding="utf-8")) if out_path.is_file() else {}
    if args.mode in {"containment", "all"}:
        result = containment(cases, encoder, head, index, args.limit)
        report["containment"] = result
        print("=== CONTAINMENT (доля фото с правильной идентичностью) ===")
        for depth in DEPTHS:
            print(f"  top-{depth:<4} {result['distribution'][f'at_{depth}']}")
        print(f"  нет в top-100: {len(result['not_in_top_100'])} из {result['with_identity']}")
    if args.mode in {"hypotheses", "all"}:
        matrix = np.load(_ROOT / settings.index_dir / "vectors.npy").astype(np.float32)
        prototypes = build_prototypes(index.slugs, matrix)
        report["prototypes"] = {"count": len(prototypes)}
        print("=== ГИПОТЕЗЫ ===")
        report["hypotheses"] = hypothesis_run(cases, encoder, head, index, catalog, store, prototypes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"отчёт: {out_path}")
    log_marker("M-EVAL", "real_field_probe", "DONE", mode=args.mode, cases=len(cases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
