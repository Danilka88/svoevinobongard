#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Харнесс сравнения препроцессинга VLM на gemma4:12b: crop/squash против full-frame/letterbox/P&S
# SCOPE: parse_args, main
# DEPENDS: numpy, cv2, PIL, packages.common.catalog, packages.common.config, packages.common.images,
#          packages.common.logging, packages.common.llm.router, packages.common.preprocess,
#          services.normalize.app.detect, services.normalize.app.tta, services.vlm.app.extract, services.vlm.app.prompt
# LINKS: M-VLM, M-EVAL
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# Mode - режим препроцессинга: crop, squash, ctx, tiles, промпт
# MODES - матрица режимов A/B/C/D для сравнения
# LONG_PROMPT - длинная инструкция (legacy-вариант промпта)
# parse_args - разбор аргументов CLI
# main - прогон матрицы препроцессинга и отчёт по полям, candidate-selection и latency
# END_MODULE_MAP
"""Сравнение препроцессинга для VLM.

Гипотеза C-012: vision-энкодер Gemma работает на 896x896 и при squash неквадратного
кадра искажает текст; Ollama не делает pan-and-scan; дефолтный num_ctx 2048 режет
контекст. Проверяем на gemma4:12b четыре режима:

A — legacy:  кроп-варп этикетки, squash в 896, ctx 2048, без тайлов, длинный промпт
B — full:    letterbox полного кадра, ctx 8192, без тайлов, короткий JSON-промпт
C — full+PAS4: letterbox + 4 тайла всего кадра, ctx 8192
D — full+PAS6: letterbox + 6 тайлов (неуверенный режим), ctx 8192

KPI: доля заполненных полей, попадание года и винодельни, latency p50/p95.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from packages.common.catalog import load_catalog  # noqa: E402
from packages.common.images import flatten_alpha  # noqa: E402
from packages.common.llm.errors import LLMError  # noqa: E402
from packages.common.llm.router import LLMRouter  # noqa: E402
from packages.common.logging import log_marker  # noqa: E402
from packages.common.preprocess import PreprocessConfig, letterbox_square  # noqa: E402
from services.normalize.app.detect import detect_label  # noqa: E402
from services.normalize.app.homography import warp_quad  # noqa: E402
from services.normalize.app.tta import build_vlm_inputs  # noqa: E402
from services.vlm.app.extract import parse_fields  # noqa: E402
from services.vlm.app.prompt import INSTRUCTION, encode_image_data_url  # noqa: E402

_DEFAULT_MATERIALS = (
    _ROOT.parent / "материалы для конкурса" / "Датасет"
    / "prod-svoe-vino-strapi" / "prod-svoe-vino" / "strapi" / "uploads"
)
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")

LONG_PROMPT = (
    "Ты — эксперт по российским винам. Внимательно изучи фотографию этикетки.\n"
    "Опиши как можно подробнее: производителя, полное название вина, год урожая,\n"
    "сорт винограда, тип (белое/красное/розовое/оранжевое), сладость, регион,\n"
    "крепость, объём, а также все прочие надписи (медали, награды, адрес сайта).\n"
    "Если что-то не видно — оставь поле пустым. Ответ дай в формате JSON со\n"
    "следующими ключами: winery, wine_name, vintage, grape, sweetness, region,\n"
    "alcohol, volume, extra_text, confidence. Верни только JSON."
)

__all__ = ["parse_args", "main"]


@dataclass(frozen=True)
class Mode:
    """Режим препроцессинга."""

    key: str
    crop: bool
    squash: bool
    ctx: int
    tiles: int
    long_prompt: bool


MODES = {
    "A": Mode("A_legacy", crop=True, squash=True, ctx=2048, tiles=0, long_prompt=True),
    "B": Mode("B_full_letterbox", crop=False, squash=False, ctx=8192, tiles=0, long_prompt=False),
    "C": Mode("C_full_pas4", crop=False, squash=False, ctx=8192, tiles=4, long_prompt=False),
    "D": Mode("D_full_pas6", crop=False, squash=False, ctx=8192, tiles=6, long_prompt=False),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы CLI."""
    parser = argparse.ArgumentParser(description="Сравнение препроцессинга VLM")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--configs", default="A,B,C")
    parser.add_argument("--model", default="gemma4:12b")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--uploads", default=str(_DEFAULT_MATERIALS))
    parser.add_argument("--out", default=str(_ROOT / "reports" / "vlm_preprocess.json"))
    return parser.parse_args(argv)


def _prepare(image: np.ndarray, mode: Mode) -> list[np.ndarray]:
    """Готовит изображения для режима."""
    if mode.tiles:
        images, _ = build_vlm_inputs(image, PreprocessConfig(pas_tiles=mode.tiles))
        return images
    if mode.crop:
        detection = detect_label(image)
        source = warp_quad(image, detection.quad) if detection.quad is not None else image
        if mode.squash:
            import cv2

            return [cv2.resize(source, (896, 896), interpolation=cv2.INTER_AREA)]
        return [letterbox_square(source, size=896)]
    return [letterbox_square(image, size=896)]


def main(argv: list[str] | None = None) -> int:
    """Прогоняет матрицу и печатает отчёт."""
    args = parse_args(argv)
    keys = [key.strip() for key in args.configs.split(",") if key.strip()]
    modes = [MODES[key] for key in keys if key in MODES]
    if not modes:
        print("ERROR: no valid configs", file=sys.stderr)
        return 2

    uploads = Path(args.uploads)
    catalog = load_catalog(_ROOT / "data" / "catalog" / "catalog.jsonl")
    picks = [
        (slug, record) for slug, record in sorted(catalog.items())
        if record.get("image") and record.get("year") and record.get("winery")
        and (uploads / record["image"]).is_file()
    ][: args.samples]
    if not picks:
        print("ERROR: no reference images with known fields", file=sys.stderr)
        return 2

    router = LLMRouter()
    report: dict = {"model": args.model, "samples": len(picks), "modes": {}}

    for mode in modes:
        rows: list[dict] = []
        latencies: list[float] = []
        for slug, record in picks:
            rgba = np.asarray(Image.open(uploads / record["image"]).convert("RGBA"))
            frame = flatten_alpha(rgba[:, :, [2, 1, 0, 3]])
            images = _prepare(frame, mode)
            prompt = LONG_PROMPT if mode.long_prompt else INSTRUCTION
            content = [{"type": "text", "text": prompt}]
            for image in images:
                content.append({"type": "image_url",
                                "image_url": {"url": encode_image_data_url(image)}})
            started = time.perf_counter()
            try:
                result = router.chat(
                    [{"role": "user", "content": content}],
                    role="text_model",
                    model=args.model,
                    max_tokens=args.max_tokens,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    extra={"reasoning_effort": "none", "options": {"num_ctx": mode.ctx}},
                    timeout_s=args.timeout,
                )
                text, error = result.text, ""
            except LLMError as exc:
                text, error = "", f"{type(exc).__name__}"
            latency = (time.perf_counter() - started) * 1000.0
            latencies.append(latency)
            fields = parse_fields(text)
            truth_year = str(record.get("year") or "")
            filled = [key for key, value in fields.items() if value and key != "confidence"]
            rows.append({
                "slug": slug,
                "images": len(images),
                "ctx": mode.ctx,
                "ok": bool(fields),
                "filled": filled,
                "vintage": fields.get("vintage", ""),
                "truth_year": truth_year,
                "vintage_hit": bool(fields.get("vintage")) and truth_year and truth_year in str(fields.get("vintage")),
                "winery": fields.get("winery", ""),
                "truth_winery": record.get("winery", ""),
                "winery_hit": bool(fields.get("winery")) and record.get("winery", "").lower()[:6] in str(fields.get("winery")).lower(),
                "error": error,
                "latency_ms": round(latency, 1),
            })
            print(f"  [{mode.key}] {slug[:32]:<34} filled={len(filled)} "
                  f"vintage={fields.get('vintage', '-')!r}/{truth_year} "
                  f"winery={str(fields.get('winery', '-'))[:22]!r} {latency:.0f}ms {error}",
                  file=sys.stderr)

        total = len(rows)
        report["modes"][mode.key] = {
            "config": {"crop": mode.crop, "squash": mode.squash, "num_ctx": mode.ctx,
                       "tiles": mode.tiles, "long_prompt": mode.long_prompt},
            "answers_with_fields": sum(1 for row in rows if row["ok"]),
            "answers_with_fields_share": round(sum(1 for row in rows if row["ok"]) / total, 3) if total else 0.0,
            "mean_filled_fields": round(sum(len(row["filled"]) for row in rows) / total, 2) if total else 0.0,
            "vintage_accuracy": round(sum(1 for row in rows if row["vintage_hit"]) / total, 3) if total else 0.0,
            "winery_accuracy": round(sum(1 for row in rows if row["winery_hit"]) / total, 3) if total else 0.0,
            "latency_ms": {
                "p50": round(float(np.percentile(latencies, 50)), 1) if latencies else 0.0,
                "p95": round(float(np.percentile(latencies, 95)), 1) if latencies else 0.0,
            },
            "rows": rows,
        }
        log_marker("M-EVAL", "vlm_preprocess", "MODE_DONE", mode=mode.key,
                   filled=report["modes"][mode.key]["mean_filled_fields"],
                   vintage_acc=report["modes"][mode.key]["vintage_accuracy"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== VLM PREPROCESSING REPORT ===")
    print(f"model: {args.model} · samples: {len(picks)}")
    print(f"{'mode':<18} {'fields≥1':<9} {'mean fields':<12} {'vintage acc':<12} {'winery acc':<11} {'p50/p95 ms'}")
    for key, data in report["modes"].items():
        print(f"{key:<18} {data['answers_with_fields_share']:<9} {data['mean_filled_fields']:<12} "
              f"{data['vintage_accuracy']:<12} {data['winery_accuracy']:<11} "
              f"{data['latency_ms']['p50']}/{data['latency_ms']['p95']}")
    print(f"out: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
