#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Офлайн-эвалюатор — multipart-POST изображений на endpoint, сбор slug, top5 и latency, accuracy@1/@5 и macro-F1
# SCOPE: build_multipart, post_image, extract_slug, extract_top5, load_ground_truth, percentile, compute_metrics, main
# DEPENDS: none (только stdlib)
# LINKS: M-EVAL
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# IMAGE_EXT - расширения файлов изображений
# build_multipart - сборка multipart-тела с полем image
# post_image - POST одного изображения с замером latency
# extract_slug - парсинг slug из объекта и из массива
# extract_top5 - парсинг top5 из dict- и str-элементов
# load_ground_truth - чтение TSV query_id и slug
# percentile - перцентиль latency p50 и p95
# compute_metrics - accuracy@1/@5 и macro-F1
# main - CLI: прогон, отчёт и predictions.jsonl
# END_MODULE_MAP
"""Оценка сервиса распознавания вин.

Отправляет изображения на endpoint (по одному, как скрипт организатора),
собирает предсказанный slug (+ top5, если endpoint /v1/scan), измеряет latency.
Если задан ground-truth — считает accuracy@1/@5 и macro/weighted F1.

Без внешних зависимостей (stdlib).

Примеры:
  python3 scripts/eval/evaluate.py --images data/samples/eval_queries \
      --endpoint http://127.0.0.1:8080/v1/eval/predict \
      --output predictions.jsonl

  python3 scripts/eval/evaluate.py --images data/samples/eval_queries \
      --ground-truth data/samples/eval_queries/ground_truth.tsv \
      --endpoint http://127.0.0.1:8080/v1/scan \
      --output predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter, defaultdict
from pathlib import Path

__all__ = [
    "IMAGE_EXT",
    "build_multipart",
    "post_image",
    "extract_slug",
    "extract_top5",
    "load_ground_truth",
    "percentile",
    "compute_metrics",
    "main",
]

# START_BLOCK_EVAL_MULTIPART
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".tif", ".tiff"}


def build_multipart(image_path: Path) -> tuple[bytes, str]:
    boundary = "----svoe" + uuid.uuid4().hex
    ctype = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    body = head + image_path.read_bytes() + tail
    return body, f"multipart/form-data; boundary={boundary}"
# END_BLOCK_EVAL_MULTIPART


# START_BLOCK_EVAL_POST
def post_image(endpoint: str, image_path: Path, timeout: float) -> tuple[float, dict | None, str]:
    body, ctype = build_multipart(image_path)
    req = urllib.request.Request(
        endpoint, data=body, method="POST", headers={"Content-Type": ctype}
    )
    print(f"[M-EVAL][POST_IMAGE][MULTIPART] endpoint={endpoint} bytes={len(body)}",
          file=sys.stderr)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            status = resp.status
    except urllib.error.HTTPError as e:
        latency = (time.perf_counter() - t0) * 1000
        return latency, None, f"HTTP {e.code}"
    except Exception as e:  # noqa: BLE001
        latency = (time.perf_counter() - t0) * 1000
        return latency, None, str(e)
    latency = (time.perf_counter() - t0) * 1000
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return latency, None, f"bad json (status {status})"
    print(f"[M-EVAL][POST_IMAGE][DONE] latency_ms={round(latency, 1)}", file=sys.stderr)
    return latency, data, ""
# END_BLOCK_EVAL_POST


# START_BLOCK_EVAL_PARSE_SLUG
def extract_slug(data) -> str | None:
    if isinstance(data, dict):
        slug = data.get("slug")
        return slug if isinstance(slug, str) and slug else None
    if isinstance(data, list) and data and isinstance(data[0], dict):
        slug = data[0].get("slug")
        return slug if isinstance(slug, str) and slug else None
    return None
# END_BLOCK_EVAL_PARSE_SLUG


# START_BLOCK_EVAL_PARSE_TOP5
def extract_top5(data) -> list[str]:
    if not isinstance(data, dict):
        return []
    top5 = data.get("top5")
    if not isinstance(top5, list):
        return []
    out = []
    for item in top5:
        if isinstance(item, dict) and isinstance(item.get("slug"), str):
            out.append(item["slug"])
        elif isinstance(item, str):
            out.append(item)
    return out
# END_BLOCK_EVAL_PARSE_TOP5


# START_BLOCK_EVAL_GT_LOAD
def load_ground_truth(path: Path) -> dict[str, str]:
    gt: dict[str, str] = {}
    with path.open(encoding="utf-8-sig") as fh:
        for i, line in enumerate(fh):
            line = line.rstrip("\n\r")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if i == 0 and parts[0].lower() in {"query_id", "id"}:
                continue
            if len(parts) >= 2:
                gt[parts[0]] = parts[1]
    return gt
# END_BLOCK_EVAL_GT_LOAD


# START_BLOCK_EVAL_PERCENTILE
def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)
# END_BLOCK_EVAL_PERCENTILE


# START_BLOCK_EVAL_METRICS
def compute_metrics(rows: list[dict], gt: dict[str, str]) -> dict:
    n = len(rows)
    acc1 = acc5 = 0
    tp = Counter()
    fp = Counter()
    fn = Counter()
    for r in rows:
        gold = gt.get(r["query_id"])
        pred = r["predicted_slug"]
        if gold is None:
            continue
        if pred == gold:
            acc1 += 1
        if r["top5"] and gold in r["top5"]:
            acc5 += 1
        elif pred == gold:
            acc5 += 1
        for label, correct in ((gold, pred == gold),):
            if correct:
                tp[label] += 1
            else:
                fn[label] += 1
                if pred:
                    fp[pred] += 1
    labeled = sum(1 for r in rows if gt.get(r["query_id"]) is not None)
    labels = set(tp) | set(fn) | set(fp)
    macro = 0.0
    if labels:
        f1s = []
        for lab in labels:
            p = tp[lab] / (tp[lab] + fp[lab]) if (tp[lab] + fp[lab]) else 0.0
            rc = tp[lab] / (tp[lab] + fn[lab]) if (tp[lab] + fn[lab]) else 0.0
            f1s.append(2 * p * rc / (p + rc) if (p + rc) else 0.0)
        macro = sum(f1s) / len(f1s)
    print(f"[M-EVAL][METRICS][F1] n={n} labeled={labeled} "
          f"acc1={round(acc1 / labeled, 4) if labeled else None} "
          f"acc5={round(acc5 / labeled, 4) if labeled else None} "
          f"macro={round(macro, 4) if labeled else None}", file=sys.stderr)
    return {
        "n": n,
        "labeled": labeled,
        "accuracy@1": round(acc1 / labeled, 4) if labeled else None,
        "accuracy@5": round(acc5 / labeled, 4) if labeled else None,
        "f1_macro@1": round(macro, 4) if labeled else None,
    }
# END_BLOCK_EVAL_METRICS


# START_BLOCK_EVAL_MAIN
def main() -> int:
    ap = argparse.ArgumentParser(description="Оценка сервиса распознавания вин")
    ap.add_argument("--images", required=True, help="папка с изображениями")
    ap.add_argument("--endpoint", required=True, help="URL сервиса")
    ap.add_argument("--manifest", help="TSV query_id<TAB>image_path (опционально)")
    ap.add_argument("--ground-truth", help="TSV query_id<TAB>slug (опционально)")
    ap.add_argument("--output", default="predictions.jsonl")
    ap.add_argument("--timeout", type=float, default=10.0)
    args = ap.parse_args()

    images_dir = Path(args.images)
    if not images_dir.is_dir():
        print(f"ERROR: images dir not found: {images_dir}", file=sys.stderr)
        return 2

    items: list[tuple[str, Path]] = []
    if args.manifest:
        with Path(args.manifest).open(encoding="utf-8-sig") as fh:
            for i, line in enumerate(fh):
                line = line.rstrip("\n\r")
                if not line:
                    continue
                parts = line.split("\t")
                if i == 0 and parts[0].lower() in {"query_id", "id"}:
                    continue
                if len(parts) < 2:
                    continue
                items.append((parts[0], images_dir / parts[1]))
    else:
        for p in sorted(images_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXT:
                items.append((p.stem, p))

    if not items:
        print("ERROR: no images to evaluate", file=sys.stderr)
        return 2

    gt = load_ground_truth(Path(args.ground_truth)) if args.ground_truth else {}
    rows: list[dict] = []
    latencies: list[float] = []

    out_path = Path(args.output)
    if out_path.exists():
        print(f"ERROR: output already exists: {out_path}", file=sys.stderr)
        return 2

    print(f"[i] endpoint: {args.endpoint}")
    print(f"[i] images:   {len(items)}")
    print(f"[i] gt:       {len(gt)} labels")
    print("-" * 60)

    with out_path.open("w", encoding="utf-8") as fh:
        for idx, (qid, img) in enumerate(items, 1):
            if not img.is_file():
                print(f"  [{idx}/{len(items)}] MISSING {img}")
                continue
            latency, data, err = post_image(args.endpoint, img, args.timeout)
            slug = extract_slug(data) if data is not None else None
            top5 = extract_top5(data) if data is not None else []
            latencies.append(latency)
            row = {
                "query_id": qid,
                "image_path": img.name,
                "predicted_slug": slug,
                "top5": top5,
                "latency_ms": round(latency, 1),
                "error": err,
            }
            rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            mark = ""
            if gt.get(qid):
                mark = "OK" if slug == gt[qid] else "MISS"
            print(f"  [{idx}/{len(items)}] {img.name:40s} -> {str(slug):45s} "
                  f"{latency:7.0f}ms {mark} {err}")

    report = {
        "endpoint": args.endpoint,
        "processed": len(rows),
        "errors": sum(1 for r in rows if r["error"]),
        "null_slug": sum(1 for r in rows if not r["predicted_slug"]),
        "latency_ms": {
            "p50": round(percentile(latencies, 0.50), 1),
            "p95": round(percentile(latencies, 0.95), 1),
            "max": round(max(latencies), 1) if latencies else 0.0,
        },
    }
    if gt:
        report.update(compute_metrics(rows, gt))

    report_path = out_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 60)
    print("=== REPORT ===")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"\nСохранено:\n  {out_path}\n  {report_path}")
    print(f"[M-EVAL][MAIN][REPORT] processed={report['processed']} errors={report['errors']} "
          f"null_slug={report['null_slug']}", file=sys.stderr)
    return 0
# END_BLOCK_EVAL_MAIN


if __name__ == "__main__":
    raise SystemExit(main())
