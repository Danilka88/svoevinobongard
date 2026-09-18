#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Сборка векторного индекса эталонных фото каталога через image-энкодер retrieval
# SCOPE: parse_args, main
# DEPENDS: numpy, PIL, packages.common.catalog, services.retrieval.app.encoder, services.retrieval.app.index
# LINKS: M-CATALOG, M-RETRIEVAL
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# parse_args - разбор аргументов CLI
# main - сборка индекса и запись artifacts
# END_MODULE_MAP
"""Сборка индекса эталонных фото.

Читает ``catalog.jsonl`` + эталонные фото из uploads, считает эмбеддинги
(alpha схлопывается на белый фон) и пишет ``data/index/{vectors.npy,slugs.json,meta.json}``.

Пример:
    python3 scripts/catalog/build_index.py --encoder dinov2 --device auto
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

from packages.common.config import settings  # noqa: E402
from packages.common.catalog import load_catalog  # noqa: E402
from packages.common.images import flatten_alpha  # noqa: E402
from packages.common.logging import log_marker  # noqa: E402
from services.retrieval.app.encoder import load_encoder  # noqa: E402
from services.retrieval.app.index import VectorIndex, save_index  # noqa: E402

__all__ = [
    "parse_args",
    "main",
]

_DEFAULT_MATERIALS = (
    _ROOT.parent / "материалы для конкурса" / "Датасет"
    / "prod-svoe-vino-strapi" / "prod-svoe-vino" / "strapi" / "uploads"
)
_MAX_SIDE = 512
_BATCH = 32


def _load_reference(path: Path) -> np.ndarray | None:
    """Читает эталонное фото и схлопывает прозрачность на белый фон."""
    try:
        with Image.open(path) as raw:
            rgba = raw.convert("RGBA")
    except Exception:  # noqa: BLE001 - битый файл не должен ломать сборку
        return None
    array = np.asarray(rgba)
    bgr = flatten_alpha(array[:, :, [2, 1, 0, 3]])
    return bgr


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
    parser = argparse.ArgumentParser(description="Сборка векторного индекса каталога")
    parser.add_argument("--catalog", default=str(_ROOT / settings.catalog_jsonl))
    parser.add_argument("--uploads", default=str(_DEFAULT_MATERIALS))
    parser.add_argument("--out", default=str(_ROOT / settings.index_dir))
    parser.add_argument("--encoder", default=settings.encoder, choices=["siglip2", "dinov2", "clip"])
    parser.add_argument("--model", default="")
    parser.add_argument("--device", default=settings.encoder_device)
    parser.add_argument("--limit", type=int, default=0, help="ограничить число позиций (0 = все)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Собирает индекс и пишет artifacts в --out."""
    args = parse_args(argv)
    uploads = Path(args.uploads)
    records = load_catalog(args.catalog)
    log_marker("M-CATALOG", "build_index", "START", catalog=len(records), uploads=str(uploads))

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
            images.append(_resize(array))
            chunk_slugs.append(slug)
        if not images:
            continue
        encoded = encoder.encode_arrays(images)
        vectors.append(encoded)
        slugs.extend(chunk_slugs)
        if (start // _BATCH) % 10 == 0:
            print(f"  encoded {len(slugs)}/{len(entries)}", file=sys.stderr)

    matrix = np.concatenate(vectors, axis=0) if vectors else np.zeros((0, 0), dtype=np.float32)
    index = VectorIndex(backend=settings.vector_backend)
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
    }
    save_index(args.out, index, meta)
    elapsed = time.perf_counter() - started
    print("=== INDEX REPORT ===")
    print(f"encoder:        {args.encoder} ({encoder.model_id})")
    print(f"device:         {encoder.device}")
    print(f"indexed slugs:  {len(slugs)}")
    print(f"missing images: {missing}")
    print(f"dim:            {index.dim}")
    print(f"elapsed:        {elapsed:.1f}s")
    print(f"out:            {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
