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
import re
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
    parser.add_argument("--same-series", action="store_true",
                        help="приоритизировать hard negatives из той же серии (винодельня + название без года)")
    parser.add_argument("--pairs-only", action="store_true",
                        help="пересобрать только pairs.jsonl из закэшированных эмбеддингов (быстрый перебор негативов)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(_ROOT / "data" / "train"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Строит пары и сохраняет эмбеддинги с индексами пар."""
    args = parse_args(argv)
    uploads = Path(args.uploads)
    catalog = load_catalog(args.catalog)
    out = Path(args.out)

    entries = [
        (slug, uploads / record["image"], record)
        for slug, record in sorted(catalog.items())
        if record.get("image") and (uploads / record["image"]).is_file()
    ]
    if args.limit:
        entries = entries[: args.limit]
    log_marker("M-EVAL", "synthetic_pairs", "START", slugs=len(entries), per_slug=args.per_slug,
               model=args.encoder_model, pairs_only=args.pairs_only)

    if not args.pairs_only:
        return _build_all(args, entries, catalog, out)

    # Быстрый путь: эмбеддинги уже посчитаны, пересобираем только пары
    slugs = json.loads((out / "slugs.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (out / "records.jsonl").read_text(encoding="utf-8").splitlines() if line]
    positives: dict[str, int] = {}
    anchor_indices: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        slug = record["slug"]
        if record.get("role") == "positive":
            positives[slug] = index
        else:
            anchor_indices.setdefault(slug, []).append(index)
    pairs = _select_pairs(args, entries, catalog, positives, anchor_indices)
    (out / "pairs.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in pairs), encoding="utf-8"
    )
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    meta.update({"pairs": len(pairs), "same_series": bool(args.same_series),
                 "hard_negatives_total": int(sum(len(p["hard_negatives"]) for p in pairs))})
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log_marker("M-EVAL", "synthetic_pairs", "PAIRS_ONLY", pairs=len(pairs),
               hard_negatives=meta["hard_negatives_total"], same_series=bool(args.same_series))
    print("=== PAIRS REPORT (pairs-only) ===")
    print(f"pairs:          {meta['pairs']}")
    print(f"hard negatives: {meta['hard_negatives_total']}")
    print(f"same-series:    {bool(args.same_series)}")
    print(f"out:            {out}")
    return 0


def _select_pairs(
    args: argparse.Namespace,
    entries: list[tuple[str, Path, dict]],
    catalog: dict[str, dict],
    positives: dict[str, int],
    anchor_indices: dict[str, list[int]],
) -> list[dict]:
    """Выбирает hard negatives (серия, винодельня, сорт) для каждой пары."""
    by_winery: dict[str, list[str]] = {}
    by_grape: dict[str, list[str]] = {}
    by_series: dict[str, list[str]] = {}
    for slug, _path, record in entries:
        if slug not in positives:
            continue
        winery = (record.get("winery") or "").strip()
        grape = (record.get("grape") or "").strip()
        series = _series_key(record)
        if winery:
            by_winery.setdefault(winery, []).append(slug)
        if grape:
            by_grape.setdefault(grape, []).append(slug)
        by_series.setdefault(series, []).append(slug)

    pairs: list[dict] = []
    for slug, anchors in anchor_indices.items():
        record = catalog.get(slug) or {}
        hard: list[int] = []
        hard_slugs: set[str] = set()
        sources: list[tuple[str, dict[str, list[str]]]] = []
        if args.same_series:
            sources.append((_series_key(record), by_series))
        sources.append(((record.get("winery") or "").strip(), by_winery))
        sources.append(((record.get("grape") or "").strip(), by_grape))
        for value, index in sources:
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
        pairs.append({"slug": slug, "positive": positives[slug],
                      "anchors": anchors, "hard_negatives": hard})
    return pairs


def _series_key(record: dict) -> str:
    """Ключ серии near-duplicates: винодельня + название без цифр."""
    winery = (record.get("winery") or "").strip().lower()
    name = " ".join(re.sub(r"[0-9]+", " ", (record.get("name") or "").lower()).split())[:48]
    return f"{winery}|{name}"


def _build_all(
    args: argparse.Namespace,
    entries: list[tuple[str, Path, dict]],
    catalog: dict[str, dict],
    out: Path,
) -> int:
    """Полный путь: кодирование эталонов и аугментаций плюс выбор пар."""
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

    pairs = _select_pairs(args, entries, catalog, positives, anchor_indices)

    matrix = np.stack(vectors).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    matrix = matrix / norms

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
        "same_series": bool(args.same_series),
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
