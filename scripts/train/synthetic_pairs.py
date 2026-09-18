#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Генерация синтетических «полевых» пар (anchor/positive) и hard negatives для домен-адаптации
# SCOPE: parse_args, main
# DEPENDS: numpy, cv2, PIL, packages.common.catalog, packages.common.images, scripts.eval.augment, services.retrieval.app.encoder
# LINKS: M-EVAL, M-RETRIEVAL
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# parse_args - разбор аргументов CLI
# main - генерация embeddings.npy и pairs.jsonl для обучения головы
# END_MODULE_MAP
"""Синтетические пары для контрастивной домен-адаптации.

Для каждого slug строится:
- **positive** — чистый эталон (кодируется один раз);
- **anchor** — тот же эталон, аугментированный в «полевое» фото (per-slug штук);
- **hard negatives** — индексы чужих эталонов той же винодельни или того же сорта
  (near-duplicates драйвят обучение сильнее случайных негативов).

Результат: ``embeddings.npy`` (N, D) L2-нормированные и ``pairs.jsonl`` с индексами.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from packages.common.catalog import load_catalog  # noqa: E402
from packages.common.logging import log_marker  # noqa: E402
from scripts.eval.augment import field_augment  # noqa: E402
from services.retrieval.app.encoder import ImageEncoder  # noqa: E402

_DEFAULT_MATERIALS = (
    _ROOT.parent / "материалы для конкурса" / "Датасет"
    / "prod-svoe-vino-strapi" / "prod-svoe-vino" / "strapi" / "uploads"
)
_BATCH = 16

__all__ = ["parse_args", "main"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы CLI."""
    parser = argparse.ArgumentParser(description="Синтетические пары для обучения головы")
    parser.add_argument("--catalog", default=str(_ROOT / "data" / "catalog" / "catalog.jsonl"))
    parser.add_argument("--uploads", default=str(_DEFAULT_MATERIALS))
    parser.add_argument("--encoder-model", default="google/siglip2-so400m-patch16-naflex")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=700, help="сколько slug брать (0 = все)")
    parser.add_argument("--per-slug", type=int, default=2, help="сколько аугментаций на slug")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(_ROOT / "data" / "train"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Строит пары и сохраняет эмбеддинги с индексами пар."""
    args = parse_args(argv)
    uploads = Path(args.uploads)
    catalog = load_catalog(args.catalog)

    entries = [
        (slug, uploads / record["image"], record)
        for slug, record in sorted(catalog.items())
        if record.get("image") and (uploads / record["image"]).is_file()
    ]
    if args.limit:
        entries = entries[: args.limit]
    log_marker("M-EVAL", "synthetic_pairs", "START", slugs=len(entries), per_slug=args.per_slug,
               model=args.encoder_model)

    encoder = ImageEncoder(args.encoder_model, device=args.device)
    rng = np.random.default_rng(args.seed)

    slugs: list[str] = []
    vectors: list[np.ndarray] = []
    records: list[dict] = []
    started = time.perf_counter()

    # positives: чистые эталоны
    positives: dict[str, int] = {}
    pending_images: list[Image.Image] = []
    pending_slugs: list[str] = []
    for position, (slug, path, _record) in enumerate(entries, 1):
        try:
            with Image.open(path) as raw:
                rgba = np.asarray(raw.convert("RGBA"))
        except Exception:  # noqa: BLE001
            continue
        bgr = rgba[:, :, [2, 1, 0, 3]]
        pending_images.append(Image.fromarray(bgr[:, :, [2, 1, 0]]))
        pending_slugs.append(slug)
        if len(pending_images) >= _BATCH:
            encoded = encoder.encode_images(pending_images)
            for offset, item_slug in enumerate(pending_slugs):
                positives[item_slug] = len(vectors)
                vectors.append(encoded[offset])
                slugs.append(item_slug)
                records.append({"slug": item_slug, "role": "positive"})
            pending_images, pending_slugs = [], []
        if position % 200 == 0:
            print(f"  positives {position}/{len(entries)}", file=sys.stderr)
    if pending_images:
        encoded = encoder.encode_images(pending_images)
        for offset, item_slug in enumerate(pending_slugs):
            positives[item_slug] = len(vectors)
            vectors.append(encoded[offset])
            slugs.append(item_slug)
            records.append({"slug": item_slug, "role": "positive"})

    # anchors: аугментированные эталоны
    anchor_indices: dict[str, list[int]] = {}
    for position, (slug, path, _record) in enumerate(entries, 1):
        if slug not in positives:
            continue
        try:
            with Image.open(path) as raw:
                rgba = np.asarray(raw.convert("RGBA"))
        except Exception:  # noqa: BLE001
            continue
        for _ in range(max(1, args.per_slug)):
            augmented = field_augment(rgba, rng)
            rgb = Image.fromarray(augmented[:, :, ::-1])
            vector = encoder.encode_images([rgb])[0]
            anchor_indices.setdefault(slug, []).append(len(vectors))
            vectors.append(vector)
            slugs.append(slug)
            records.append({"slug": slug, "role": "anchor"})
        if position % 200 == 0:
            print(f"  anchors {position}/{len(entries)}", file=sys.stderr)

    # hard negatives: та же винодельня или тот же сорт
    by_winery: dict[str, list[str]] = {}
    by_grape: dict[str, list[str]] = {}
    for slug, _path, record in entries:
        if slug not in positives:
            continue
        winery = (record.get("winery") or "").strip()
        grape = (record.get("grape") or "").strip()
        if winery:
            by_winery.setdefault(winery, []).append(slug)
        if grape:
            by_grape.setdefault(grape, []).append(slug)

    pairs: list[dict] = []
    for slug, anchors in anchor_indices.items():
        record = catalog.get(slug) or {}
        hard: list[int] = []
        hard_slugs: set[str] = set()
        for key, index in (("winery", by_winery), ("grape", by_grape)):
            value = (record.get(key) or "").strip()
            if not value:
                continue
            for other in index.get(value, []):
                if other == slug or other not in positives or other in hard_slugs:
                    continue
                hard.append(positives[other])
                hard_slugs.add(other)
                if len(hard) >= 8:
                    break
            if len(hard) >= 8:
                break
        pairs.append({
            "slug": slug,
            "positive": positives[slug],
            "anchors": anchors,
            "hard_negatives": hard,
        })

    matrix = np.stack(vectors).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    matrix = matrix / norms

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", matrix)
    (out / "slugs.json").write_text(json.dumps(slugs, ensure_ascii=False), encoding="utf-8")
    (out / "records.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records), encoding="utf-8"
    )
    (out / "pairs.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in pairs), encoding="utf-8"
    )
    meta = {
        "encoder_model": args.encoder_model,
        "device": encoder.device,
        "dim": int(matrix.shape[1]),
        "embeddings": int(matrix.shape[0]),
        "slugs": len(positives),
        "pairs": len(pairs),
        "per_slug": args.per_slug,
        "seed": args.seed,
        "hard_negatives_total": int(sum(len(p["hard_negatives"]) for p in pairs)),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log_marker("M-EVAL", "synthetic_pairs", "DONE", embeddings=meta["embeddings"],
               pairs=meta["pairs"], hard_negatives=meta["hard_negatives_total"],
               elapsed_s=round(time.perf_counter() - started, 1))

    print("=== PAIRS REPORT ===")
    print(f"encoder:        {args.encoder_model} on {encoder.device}")
    print(f"embeddings:     {meta['embeddings']} (dim {meta['dim']})")
    print(f"slugs:          {meta['slugs']}")
    print(f"pairs:          {meta['pairs']} (per-slug {args.per_slug}, seed {args.seed})")
    print(f"hard negatives: {meta['hard_negatives_total']}")
    print(f"elapsed:        {time.perf_counter() - started:.1f}s")
    print(f"out:            {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
