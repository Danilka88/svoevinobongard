#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Сборка векторного индекса эталонных фото и локальных признаков для геоверификации
# SCOPE: parse_args, main
# DEPENDS: numpy, PIL, packages.common.catalog, packages.common.images, packages.common.logging,
#          services.retrieval.app.encoder, services.retrieval.app.geo, services.retrieval.app.head, services.retrieval.app.index
# LINKS: M-CATALOG, M-RETRIEVAL
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# parse_args - разбор аргументов CLI
# main - сборка индекса, признаков геоверификации и запись artifacts
# END_MODULE_MAP
"""Сборка индекса эталонных фото.

Читает ``catalog.jsonl`` + эталонные фото из uploads, считает эмбеддинги
(alpha схлопывается на белый фон, опционально применяется проекционная голова)
и локальные признаки для геоверификации, затем пишет
``data/index/{vectors.npy,slugs.json,meta.json,head.pt,features.npz}``.

Пример:
    python3 scripts/catalog/build_index.py --encoder siglip2 --head models/head.pt --features
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from packages.common.catalog import load_catalog  # noqa: E402
from packages.common.images import flatten_alpha  # noqa: E402
from packages.common.logging import log_marker  # noqa: E402
from services.retrieval.app.encoder import load_encoder  # noqa: E402
from services.retrieval.app.geo import (  # noqa: E402
    DEFAULT_MAX_KEYPOINTS,
    FEATURES_FILE,
    extract_features,
    resolve_geo_method,
)
from services.retrieval.app.head import apply_head, head_meta, load_head  # noqa: E402
from services.retrieval.app.index import VectorIndex, save_index  # noqa: E402

_DEFAULT_MATERIALS = (
    _ROOT.parent / "материалы для конкурса" / "Датасет"
    / "prod-svoe-vino-strapi" / "prod-svoe-vino" / "strapi" / "uploads"
)
_MAX_SIDE = 512
_BATCH = 32

__all__ = ["parse_args", "main"]


def _descriptor_dim(method: str) -> int:
    """Размерность дескриптора выбранного метода."""
    return 128 if method == "sift" else 32


def _descriptor_dtype(method: str):
    """Тип хранения дескрипторов: ORB компактнее в uint8, SIFT — в float16."""
    return np.float16 if method == "sift" else np.uint8


def _pad(array: np.ndarray, rows: int, columns: int, dtype) -> np.ndarray:
    """Дополняет матрицу нулями до фиксированной формы."""
    padded = np.zeros((rows, columns), dtype=dtype)
    if array.size:
        count = min(rows, array.shape[0])
        padded[:count, :array.shape[1]] = array[:count]
    return padded


def _load_reference(path: Path) -> np.ndarray | None:
    """Читает эталонное фото и схлопывает прозрачность на белый фон."""
    try:
        with Image.open(path) as raw:
            rgba = raw.convert("RGBA")
    except Exception:  # noqa: BLE001 - битый файл не должен ломать сборку
        return None
    array = np.asarray(rgba)
    return flatten_alpha(array[:, :, [2, 1, 0, 3]])


def _resize(array: np.ndarray, max_side: int = _MAX_SIDE) -> np.ndarray:
    """Пропорционально уменьшает изображение по длинной стороне."""
    import cv2

    height, width = array.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return array
    scale = max_side / float(longest)
    return cv2.resize(array, (max(1, int(width * scale)), max(1, int(height * scale))),
                      interpolation=cv2.INTER_AREA)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    import os

    parser = argparse.ArgumentParser(description="Сборка векторного индекса каталога")
    parser.add_argument("--catalog", default=str(_ROOT / "data" / "catalog" / "catalog.jsonl"))
    parser.add_argument("--uploads", default=str(_DEFAULT_MATERIALS))
    parser.add_argument("--out", default=str(_ROOT / "data" / "index"))
    parser.add_argument("--encoder", default=os.getenv("ENCODER", "siglip2"),
                        choices=["siglip2", "dinov2", "clip"])
    parser.add_argument("--model", default="")
    parser.add_argument("--device", default=os.getenv("ENCODER_DEVICE", "auto"))
    parser.add_argument("--head", default="", help="путь к обученной проекционной голове (models/head.pt)")
    parser.add_argument("--features", action="store_true",
                        help="предвычислить локальные признаки эталонов для геоверификации")
    parser.add_argument("--geo-method", default=os.getenv("GEO_METHOD", "orb"), choices=["orb", "sift", "auto"])
    parser.add_argument("--geo-max-keypoints", type=int, default=DEFAULT_MAX_KEYPOINTS)
    parser.add_argument("--limit", type=int, default=0, help="ограничить число позиций (0 = все)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Собирает индекс (и признаки) и пишет artifacts в --out."""
    args = parse_args(argv)
    uploads = Path(args.uploads)
    records = load_catalog(args.catalog)
    geo_method = resolve_geo_method(args.geo_method)
    log_marker("M-CATALOG", "build_index", "START", catalog=len(records), uploads=str(uploads),
               features=args.features, geo_method=geo_method)

    entries: list[tuple[str, Path]] = []
    missing = 0
    for slug, record in sorted(records.items()):
        image_name = record.get("image")
        if not image_name:
            continue
        path = uploads / image_name
        if not path.is_file():
            missing += 1
            continue
        entries.append((slug, path))
    if args.limit:
        entries = entries[: args.limit]
    if not entries:
        print(f"ERROR: no reference images found under {uploads}", file=sys.stderr)
        return 2

    encoder = load_encoder(encoder=args.encoder, model_id=args.model or None, device=args.device)
    slugs: list[str] = []
    vectors: list[np.ndarray] = []
    feat_keypoints: list[np.ndarray] = []
    feat_descriptors: list[np.ndarray] = []
    feat_counts: list[int] = []
    descriptor_dim = _descriptor_dim(geo_method)
    descriptor_dtype = _descriptor_dtype(geo_method)
    started = time.perf_counter()

    for start in range(0, len(entries), _BATCH):
        chunk = entries[start:start + _BATCH]
        images = []
        chunk_slugs = []
        for slug, path in chunk:
            array = _load_reference(path)
            if array is None:
                missing += 1
                continue
            resized = _resize(array)
            images.append(resized)
            chunk_slugs.append(slug)
            if args.features:
                features = extract_features(resized, method=geo_method,
                                            max_keypoints=args.geo_max_keypoints)
                feat_keypoints.append(_pad(features.keypoints, args.geo_max_keypoints, 2, np.float32))
                feat_descriptors.append(_pad(features.descriptors, args.geo_max_keypoints,
                                             descriptor_dim, descriptor_dtype))
                feat_counts.append(int(len(features.keypoints)))
        if not images:
            continue
        encoded = encoder.encode_arrays(images)
        vectors.append(encoded)
        slugs.extend(chunk_slugs)
        if (start // _BATCH) % 10 == 0:
            print(f"  encoded {len(slugs)}/{len(entries)}", file=sys.stderr)

    matrix = np.concatenate(vectors, axis=0) if vectors else np.zeros((0, 0), dtype=np.float32)
    head_payload: dict | None = None
    if args.head:
        head = load_head(args.head, device="cpu")
        matrix = apply_head(head, matrix, device="cpu")
        head_payload = head_meta(head)
        log_marker("M-RETRIEVAL", "build_index", "HEAD_APPLIED",
                   in_dim=head.in_dim, out_dim=head.out_dim, vectors=len(slugs))

    index = VectorIndex(backend="numpy")
    index.build(slugs, matrix)

    meta = {
        "encoder": args.encoder,
        "model": encoder.model_id,
        "device": encoder.device,
        "dim": index.dim,
        "count": len(slugs),
        "missing_images": missing,
        "max_side": _MAX_SIDE,
        "catalog": str(args.catalog),
        "backend": index.effective_backend,
    }
    if head_payload is not None:
        meta["head"] = head_payload
    output = Path(args.out)
    save_index(output, index, meta)

    if head_payload is not None:
        target_head = output / head_payload["file"]
        target_head.parent.mkdir(parents=True, exist_ok=True)
        target_head.write_bytes(Path(args.head).read_bytes())
        log_marker("M-RETRIEVAL", "build_index", "HEAD_SAVED", path=str(target_head))

    if args.features and feat_counts:
        np.savez_compressed(
            output / FEATURES_FILE,
            keypoints=np.stack(feat_keypoints).astype(np.float32),
            descriptors=np.stack(feat_descriptors).astype(descriptor_dtype),
            counts=np.array(feat_counts, dtype=np.int32),
        )
        meta["features"] = {
            "file": FEATURES_FILE,
            "method": geo_method,
            "max_keypoints": args.geo_max_keypoints,
            "dim": descriptor_dim,
        }
        (output / "meta.json").write_text(
            __import__("json").dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log_marker("M-RETRIEVAL", "build_index", "FEATURES_SAVED", method=geo_method,
                   images=len(feat_counts), mean_keypoints=round(float(np.mean(feat_counts)), 1))

    elapsed = time.perf_counter() - started
    print("=== INDEX REPORT ===")
    print(f"encoder:        {args.encoder} ({encoder.model_id})")
    print(f"device:         {encoder.device}")
    print(f"indexed slugs:  {len(slugs)}")
    print(f"missing images: {missing}")
    print(f"dim:            {index.dim}")
    print(f"head:           {head_payload['in_dim']}->{head_payload['out_dim']}" if head_payload else "head:           none")
    if args.features and feat_counts:
        print(f"geo features:   {geo_method}, images {len(feat_counts)}, "
              f"mean keypoints {float(np.mean(feat_counts)):.1f}, dim {descriptor_dim}")
    else:
        print("geo features:   none")
    print(f"elapsed:        {elapsed:.1f}s")
    print(f"out:            {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
