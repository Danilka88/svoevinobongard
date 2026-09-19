#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Baseline-отчёт Tier-1: self-retrieval, proxy F1 на синтетике, групповые метрики near-duplicates и статистика геоверификации
# SCOPE: parse_args, main
# DEPENDS: numpy, cv2, PIL, packages.common.catalog, packages.common.config, packages.common.logging,
#          scripts.eval.augment, services.normalize.app.detect, services.normalize.app.tta,
#          services.retrieval.app.encoder, services.retrieval.app.geo, services.retrieval.app.head,
#          services.retrieval.app.index, services.retrieval.app.search
# LINKS: M-EVAL, M-RETRIEVAL
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# parse_args - разбор аргументов CLI
# main - прогон self-retrieval и proxy-метрик, запись отчёта
# END_MODULE_MAP
"""Baseline-отчёт качества Tier-1.

Что измеряется:
1. **self-retrieval** — контроль исправности индекса.
2. **synthetic proxy** — эталон аугментируется в «полевое» фото и ищется в индексе.
3. **Групповые метрики** — вместо macro-F1 по 200 singleton-классам считаются метрики
   по группам near-duplicates (винодельня + название без года). Именно на этих парах
   проверяется достоверность по ТЗ.
4. **Статистика геоверификации** — средний geo-скор у верных и неверных ответов.

Полевой набор в комплект не входит: используются только официальные материалы
конкурса, поэтому реальные F1@1/@5 на полевых фото не считаются — метрики качества
берутся из synthetic proxy, а официальные фото дают демонстрацию и latency.
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

from packages.common.catalog import CatalogStore, load_catalog  # noqa: E402
from packages.common.config import settings  # noqa: E402
from packages.common.images import flatten_alpha  # noqa: E402
from packages.common.logging import log_marker  # noqa: E402
from scripts.eval.augment import field_augment, holdout_augment  # noqa: E402
from services.normalize.app.detect import detect_label  # noqa: E402
from services.normalize.app.tta import build_variants  # noqa: E402
from services.retrieval.app.encoder import ImageEncoder  # noqa: E402
from services.retrieval.app.geo import GeoIndex  # noqa: E402
from services.retrieval.app.head import load_head, read_head_meta  # noqa: E402
from services.retrieval.app.index import load_index  # noqa: E402
from services.retrieval.app.search import search_images  # noqa: E402

_DEFAULT_MATERIALS = (
    _ROOT.parent / "материалы для конкурса" / "Датасет"
    / "prod-svoe-vino-strapi" / "prod-svoe-vino" / "strapi" / "uploads"
)
_MAX_SIDE = 512
_TOP_K = 5

__all__ = ["parse_args", "main"]


def _resize_bgr(array: np.ndarray, max_side: int = _MAX_SIDE) -> np.ndarray:
    """Пропорционально уменьшает BGR-изображение по длинной стороне."""
    import cv2

    height, width = array.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return array
    scale = max_side / float(longest)
    return cv2.resize(array, (max(1, int(width * scale)), max(1, int(height * scale))),
                      interpolation=cv2.INTER_AREA)


def _load_reference_rgba(path: Path) -> np.ndarray | None:
    """Читает эталон как RGBA-массив."""
    try:
        with Image.open(path) as raw:
            return np.asarray(raw.convert("RGBA"))
    except Exception:  # noqa: BLE001 - битый файл пропускаем
        return None


def _group_key(record: dict) -> str:
    """Ключ группы near-duplicates: винодельня + название без года и цифр."""
    winery = (record.get("winery") or "").strip().lower()
    name = re.sub(r"\d+", " ", (record.get("name") or "").lower())
    name = " ".join(name.split())[:48]
    return f"{winery}|{name}"


def _macro_f1(pairs: list[tuple[str, str]]) -> float:
    """Macro-F1 по парам (gold, predicted)."""
    labels = {gold for gold, _ in pairs} | {pred for _, pred in pairs}
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        tp = sum(1 for gold, pred in pairs if gold == label and pred == label)
        fp = sum(1 for gold, pred in pairs if gold != label and pred == label)
        fn = sum(1 for gold, pred in pairs if gold == label and pred != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return sum(scores) / len(scores)


def _ratio(numerator: int, total: int) -> float:
    """Безопасная доля с округлением до 4 знаков."""
    return round(numerator / total, 4) if total else 0.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы CLI."""
    parser = argparse.ArgumentParser(description="Baseline-отчёт Tier-1 retrieval")
    parser.add_argument("--index", default=str(_ROOT / "data" / "index"))
    parser.add_argument("--catalog", default=str(_ROOT / "data" / "catalog" / "catalog.jsonl"))
    parser.add_argument("--uploads", default=str(_DEFAULT_MATERIALS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize", action="store_true",
                        help="прогнать синтетический запрос через normalize (детекция+кроп) до retrieval")
    parser.add_argument("--only-indexed", action="store_true",
                        help="сэмплировать только slug, присутствующие в индексе")
    parser.add_argument("--geo", action="store_true", help="включить геоверификацию Top-K")
    parser.add_argument("--geo-weight", type=float, default=None,
                        help="вес геоверификации (по умолчанию из env GEO_WEIGHT)")
    parser.add_argument("--augment", default="train", choices=["train", "holdout"],
                        help="семейство аугментаций запроса: train (галерея) или OOD holdout")
    parser.add_argument("--tta", type=int, default=1,
                        help="сколько вариантов запроса агрегировать (C-022: query-TTA)")
    parser.add_argument("--winery-proto", action="store_true",
                        help="включить бонус прототипов виноделен (C-022)")
    parser.add_argument("--winery-weight", type=float, default=None,
                        help="вес бонуса винодельни; по умолчанию из настроек")
    parser.add_argument("--out", default=str(_ROOT / "reports" / "baseline.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Считает self-retrieval и proxy-метрики, пишет JSON-отчёт."""
    args = parse_args(argv)
    index_dir = Path(args.index)
    meta_path = index_dir / "meta.json"
    if not meta_path.is_file():
        print(f"ERROR: index meta not found: {meta_path}", file=sys.stderr)
        return 2
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    index = load_index(index_dir, backend="numpy")
    encoder = ImageEncoder(meta["model"], device=args.device)
    head_payload = read_head_meta(meta)
    head = load_head(index_dir / head_payload["file"], device="cpu") if head_payload else None
    geo_index = GeoIndex.load(index_dir, meta) if args.geo else None
    geo_weight = args.geo_weight if args.geo_weight is not None else settings.geo_weight
    log_marker("M-EVAL", "baseline", "START", index=len(index.slugs), model=meta["model"],
               samples=args.samples, head="yes" if head else "no",
               geo="yes" if geo_index else "no", geo_weight=geo_weight)

    catalog = load_catalog(args.catalog)
    store = CatalogStore(catalog)
    group_sizes: dict[str, int] = {}
    gold_group: dict[str, str] = {}
    for slug, record in catalog.items():
        key = _group_key(record)
        gold_group[slug] = key
        group_sizes[key] = group_sizes.get(key, 0) + 1

    uploads = Path(args.uploads)
    indexed = set(index.slugs)
    available = [
        (slug, uploads / record["image"])
        for slug, record in sorted(catalog.items())
        if record.get("image") and (uploads / record["image"]).is_file()
        and (not args.only_indexed or slug in indexed)
    ]
    if args.samples and args.samples < len(available):
        picker = np.random.default_rng(args.seed)
        chosen = picker.choice(len(available), size=args.samples, replace=False)
        sample = [available[int(i)] for i in np.sort(chosen)]
    else:
        sample = available

    self_hits: list[tuple[str, str]] = []
    self_scores: list[float] = []
    gold: list[str] = []
    top1: list[str] = []
    top5: list[list[str]] = []
    top1_geo: list[float] = []
    latencies: list[float] = []
    geo_correct: list[float] = []
    geo_wrong: list[float] = []
    misses: list[dict] = []
    prototypes = None
    proto_weight = args.winery_weight if args.winery_weight is not None else settings.winery_proto_weight
    if args.winery_proto or settings.winery_proto_enabled:
        from services.retrieval.app.winery import build_prototypes  # noqa: PLC0415

        # vectors.npy уже в пространстве головы: применять её повторно нельзя (256 против 1152)
        matrix = np.load(Path(args.index) / "vectors.npy").astype(np.float32)
        prototypes = build_prototypes(index.slugs, matrix)
        print(f"winery prototypes: {len(prototypes)} виноделен, вес {proto_weight}")

    rng = np.random.default_rng(args.seed + 1)

    def _search(images: list[np.ndarray], k: int) -> list:
        return search_images(
            encoder, index, store, images, k=k, head=head,
            geo_index=geo_index, geo_weight=geo_weight,
            geo_topk=settings.geo_topk, geo_min_matches=settings.geo_min_matches,
        )

    for position, (slug, path) in enumerate(sample, 1):
        rgba = _load_reference_rgba(path)
        if rgba is None:
            continue

        plain = _resize_bgr(flatten_alpha(rgba[:, :, [2, 1, 0, 3]]))
        plain_hits = _search([plain], _TOP_K)
        self_hits.append((slug, plain_hits[0].slug if plain_hits else ""))
        self_scores.append(float(plain_hits[0].score) if plain_hits else 0.0)

        augment_fn = holdout_augment if args.augment == "holdout" else field_augment
        augmented = augment_fn(rgba, rng)
        query_image = augmented
        query_images: list[np.ndarray] = [augmented]
        if args.normalize:
            detection = detect_label(augmented)
            query_images = list(build_variants(augmented, detection.quad))[: max(1, args.tta)]
            query_image = query_images[0]

        # пул кандидатов шире top-5: геоверификация должна уметь поднимать ранги 6+
        # C-022: query-TTA — агрегируем несколько вариантов предобработки
        started = time.perf_counter()
        hits = _search(query_images, settings.top_k)
        latencies.append((time.perf_counter() - started) * 1000.0)

        if prototypes is not None and len(prototypes):
            # второй уровень: бонус кандидатам из близких к запросу виноделен
            from services.retrieval.app.head import apply_head  # noqa: PLC0415
            from services.retrieval.app.winery import boost  # noqa: PLC0415

            # голова лежит на CPU (как и в search_images), поэтому и применяется на CPU
            query_vector = apply_head(head, encoder.encode_arrays([query_images[0]]), device="cpu")[0]
            index_of = {slug: position for position, slug in enumerate(index.slugs)}
            candidates = [{"slug": hit.slug, "score": float(hit.score),
                           "index": index_of.get(hit.slug, -1), "hit": hit} for hit in hits]
            ordered = boost(candidates, prototypes, query_vector,
                            weight=proto_weight, top_wineries=settings.winery_proto_top_wineries
                            if hasattr(settings, "winery_proto_top_wineries") else 5)
            hits = [entry["hit"] for entry in ordered]

        ranked = [hit.slug for hit in hits]
        predicted = ranked[0] if ranked else ""
        gold.append(slug)
        top1.append(predicted)
        top5.append(ranked[:_TOP_K])
        top1_geo.append(float(hits[0].geo_score) if hits else 0.0)
        if predicted == slug:
            geo_correct.append(top1_geo[-1])
        else:
            geo_wrong.append(top1_geo[-1])
            misses.append({
                "slug": slug,
                "predicted": predicted or None,
                "same_group": gold_group.get(predicted) == gold_group.get(slug),
                "top5": ranked,
                "geo_score": round(top1_geo[-1], 4),
            })

        if position % 50 == 0:
            print(f"  evaluated {position}/{len(sample)}", file=sys.stderr)

    total = len(gold)
    pairs = list(zip(gold, top1))
    self_gold = [slug for slug, _ in sample][: len(self_hits)]
    self_acc1 = _ratio(sum(1 for g, p in zip(self_gold, [h for _, h in self_hits]) if g == p),
                       len(self_hits))

    group_pairs = [
        (gold_group.get(g, ""), gold_group.get(p, ""))
        for g, p in pairs
    ]
    near_dup_index = [i for i, g in enumerate(gold) if group_sizes.get(gold_group.get(g, ""), 0) > 1]
    nd_gold = [gold[i] for i in near_dup_index]
    nd_top1 = [top1[i] for i in near_dup_index]
    nd_top5 = [top5[i] for i in near_dup_index]

    report = {
        "encoder": meta.get("encoder"),
        "model": meta.get("model"),
        "device": meta.get("device"),
        "index_size": len(index.slugs),
        "index_dim": meta.get("dim"),
        "vector_backend": meta.get("backend"),
        "head": meta.get("head") or "none",
        "geo_weight": geo_weight if geo_index else 0.0,
        "geo_method": (meta.get("features") or {}).get("method"),
        "samples": total,
        "seed": args.seed,
        "normalize_applied": bool(args.normalize),
        "augment_family": args.augment,
        "self_retrieval": {
            "accuracy@1": self_acc1,
            "mean_top1_score": round(float(np.mean(self_scores)), 4) if self_scores else 0.0,
        },
        "synthetic_field": {
            "accuracy@1": _ratio(sum(1 for g, p in pairs if g == p), total),
            "accuracy@5": _ratio(sum(1 for g, r in zip(gold, top5) if g in r), total),
            "f1_macro_singleton@1": round(_macro_f1(pairs), 4),
            "group_accuracy@1": _ratio(sum(1 for g, p in group_pairs if g and g == p), total),
            "f1_macro_group@1": round(_macro_f1(group_pairs), 4),
            "latency_ms": {
                "p50": round(float(np.percentile(latencies, 50)), 1) if latencies else 0.0,
                "p95": round(float(np.percentile(latencies, 95)), 1) if latencies else 0.0,
            },
            "misses": len(misses),
            "miss_examples": misses[:10],
        },
        "near_duplicate": {
            "samples": len(nd_gold),
            "accuracy@1": _ratio(sum(1 for g, p in zip(nd_gold, nd_top1) if g == p), len(nd_gold)),
            "accuracy@5": _ratio(sum(1 for g, r in zip(nd_gold, nd_top5) if g in r), len(nd_gold)),
            "group_accuracy@1": _ratio(
                sum(1 for g, p in zip(nd_gold, nd_top1)
                    if gold_group.get(p, "") == gold_group.get(g, "")), len(nd_gold)),
        },
        "geo_stats": {
            "mean_top1_geo_correct": round(float(np.mean(geo_correct)), 4) if geo_correct else 0.0,
            "mean_top1_geo_wrong": round(float(np.mean(geo_wrong)), 4) if geo_wrong else 0.0,
        },
        "note": ("Полевой набор не входит в комплект (только официальные материалы конкурса), "
                 "не считаются; synthetic_field — proxy-оценка доменного разрыва. "
                 "Групповые метрики считаются по группам near-duplicates (винодельня + название без года)."),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log_marker("M-EVAL", "baseline", "DONE", self_acc1=self_acc1,
               acc1=report["synthetic_field"]["accuracy@1"],
               acc5=report["synthetic_field"]["accuracy@5"],
               nd_acc1=report["near_duplicate"]["accuracy@1"])

    print("=== BASELINE REPORT ===")
    print(f"encoder:            {report['encoder']} ({report['model']}) on {report['device']}")
    print(f"index:              {report['index_size']} vectors, dim {report['index_dim']}, {report['vector_backend']}")
    print(f"head / geo:         {report['head']} / {report['geo_method']} (weight {report['geo_weight']})")
    print(f"samples:            {report['samples']} (seed {args.seed}, normalize={report['normalize_applied']})")
    print(f"self acc@1:         {report['self_retrieval']['accuracy@1']}")
    print(f"synthetic acc@1:    {report['synthetic_field']['accuracy@1']}")
    print(f"synthetic acc@5:    {report['synthetic_field']['accuracy@5']}")
    print(f"f1_macro_singleton: {report['synthetic_field']['f1_macro_singleton@1']}")
    print(f"group acc@1:        {report['synthetic_field']['group_accuracy@1']}")
    print(f"f1_macro_group:     {report['synthetic_field']['f1_macro_group@1']}")
    print(f"near-dup samples:   {report['near_duplicate']['samples']}  acc@1 {report['near_duplicate']['accuracy@1']}  group acc@1 {report['near_duplicate']['group_accuracy@1']}")
    print(f"geo correct/wrong:  {report['geo_stats']['mean_top1_geo_correct']} / {report['geo_stats']['mean_top1_geo_wrong']}")
    print(f"latency:            p50 {report['synthetic_field']['latency_ms']['p50']}ms  p95 {report['synthetic_field']['latency_ms']['p95']}ms")
    print(f"out:                {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
