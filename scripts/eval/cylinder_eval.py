#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Замер цилиндрического unwrap на реальных фото: читаемость OCR, текстовый потолок, стабильность
# SCOPE: load_cases, ocr_stats, ceiling_for, main
# DEPENDS: json, re, sys, numpy, PIL, packages.common.catalog, packages.common.cylinder,
#          packages.common.images, packages.common.logging, packages.common.ocr,
#          packages.common.text_norm, scripts.eval.text_only_ceiling
# LINKS: M-EVAL, M-OCR, M-NORMALIZE
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# DEFAULT_DIR - каталог реальных фото
# load_cases - фото с текстом этикетки из manifest
# ocr_stats - метрики чтения: строки, кириллические слова, год
# ceiling_for - текстовый потолок по распознанному тексту
# main - прогон до/после unwrap и отчёт
# END_MODULE_MAP
"""Замер цилиндрического unwrap.

Ключевой вопрос — стабильность: на скольких реальных фото автопоиск маркеров вообще
срабатывает. Далее сравнивается чтение OCR до и после (строки, кириллические слова, год)
и текстовый потолок, посчитанный по распознанному тексту.
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
from packages.common.cylinder import unwrap_auto  # noqa: E402
from packages.common.images import flatten_alpha  # noqa: E402
from packages.common.logging import log_marker  # noqa: E402
from packages.common.ocr import parse_fields, run_ocr  # noqa: E402
from scripts.eval.text_only_ceiling import match_strict  # noqa: E402

DEFAULT_DIR = "data/samples/real_field"

__all__ = ["DEFAULT_DIR", "ceiling_for", "load_cases", "main", "ocr_stats"]


def load_cases(directory: Path) -> list[dict]:
    """Читает реальные фото с текстом этикетки из manifest."""
    manifest = directory / "manifest.tsv"
    cases: list[dict] = []
    if not manifest.is_file():
        return cases
    for line in manifest.read_text(encoding="utf-8").splitlines():
        name, _, text = line.partition("\t")
        path = directory / name
        if not path.is_file():
            continue
        try:
            with Image.open(path) as raw:
                image = flatten_alpha(np.asarray(raw.convert("RGBA"))[:, :, [2, 1, 0, 3]])
        except Exception:  # noqa: BLE001
            continue
        cases.append({"file": name, "text": text.strip(), "image": image})
    return cases


def ocr_stats(image: np.ndarray) -> dict:
    """Считает метрики чтения: строки, кириллические слова, год и сладость."""
    result = run_ocr(image, engine="fusion")
    text = result.text or ""
    words = re.findall(r"[а-яА-ЯёЁ]{3,}", text)
    fields = parse_fields([text])
    return {
        "lines": len(result.lines),
        "cyrillic_words": len(words),
        "vintage": fields["vintage"],
        "sweetness": fields["sweetness"],
        "text": text,
    }


def ceiling_for(entry_text: str, catalog: dict, threshold: float = 0.82) -> bool:
    """Проверяет, достаточен ли распознанный текст для правильной идентичности."""
    entry = {"file": "probe", "vlm": {}, "ocr_text": entry_text}
    return bool(match_strict(entry, catalog, threshold)["matched"])


def main(argv: list[str] | None = None) -> int:
    """Прогоняет unwrap и сравнивает чтение до/после."""
    parser = argparse.ArgumentParser(description="Замер цилиндрического unwrap")
    parser.add_argument("--dir", default=str(_ROOT / DEFAULT_DIR))
    parser.add_argument("--out", default=str(_ROOT / "reports/cylinder.json"))
    args = parser.parse_args(argv)

    catalog = load_catalog(_ROOT / "data/catalog/catalog.jsonl")
    cases = load_cases(Path(args.dir))
    rows: list[dict] = []
    applied = 0
    unwrap_ms: list[float] = []
    for case in cases:
        started = time.perf_counter()
        unwrapped, info = unwrap_auto(case["image"], enabled=True)
        latency = (time.perf_counter() - started) * 1000.0
        unwrap_ms.append(latency)
        applied += int(info["applied"])
        before = ocr_stats(case["image"])
        after = ocr_stats(unwrapped) if info["applied"] else before
        rows.append({
            "file": case["file"],
            "applied": info["applied"],
            "confidence": float(info["confidence"]),
            "reason": info["reason"],
            "unwrap_ms": round(latency, 1),
            "lines_before": before["lines"], "lines_after": after["lines"],
            "cyr_before": before["cyrillic_words"], "cyr_after": after["cyrillic_words"],
            "vintage_before": before["vintage"], "vintage_after": after["vintage"],
            # Потолок считается только для результата unwrap: baseline уже измерен в C-027 (6 из 21),
            # а каждый вызов match_strict перебирает весь каталог и стоит секунды.
            "ceiling_after": ceiling_for(after["text"], catalog) if info["applied"] else None,
        })
        print(f"  {case['file'][:30]:<32} applied={info['applied']} conf={info['confidence']:.2f} "
              f"строк {before['lines']}->{after['lines']} кир {before['cyrillic_words']}->{after['cyrillic_words']}",
              flush=True)

    def share(key: str) -> float:
        return round(sum(1 for row in rows if row[key]) / len(rows), 4) if rows else 0.0

    report = {
        "cases": len(rows),
        "auto_points_applied": applied,
        "auto_points_share": round(applied / len(rows), 4) if rows else 0.0,
        "lines_before": sum(row["lines_before"] for row in rows),
        "lines_after": sum(row["lines_after"] for row in rows),
        "cyrillic_before": sum(row["cyr_before"] for row in rows),
        "cyrillic_after": sum(row["cyr_after"] for row in rows),
        "vintage_read_before": share("vintage_before"),
        "vintage_read_after": share("vintage_after"),
        "ceiling_before_c027": 6,  # измерено в C-027 на исходных кадрах
        "ceiling_after_unwrapped": sum(1 for row in rows if row["ceiling_after"]),
        "latency_ms": {"unwrap_p50": round(sorted(unwrap_ms)[len(unwrap_ms) // 2], 1) if unwrap_ms else 0.0,
                       "unwrap_p95": round(sorted(unwrap_ms)[min(len(unwrap_ms) - 1, int(0.95 * len(unwrap_ms)))], 1) if unwrap_ms else 0.0},
        "reasons": {reason: sum(1 for row in rows if row["reason"] == reason)
                    for reason in sorted({row["reason"] for row in rows})},
        "rows": rows,
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== ЦИЛИНДРИЧЕСКИЙ UNWRAP (реальные фото) ===")
    print(f"  автопоиск маркеров сработал: {applied} из {len(rows)} ({report['auto_points_share']})")
    print(f"  строки OCR: {report['lines_before']} -> {report['lines_after']} · "
          f"кириллических слов: {report['cyrillic_before']} -> {report['cyrillic_after']}")
    print(f"  год читается: {report['vintage_read_before']} -> {report['vintage_read_after']}")
    print(f"  текстовый потолок: {report['ceiling_before']} -> {report['ceiling_after']} из {len(rows)}")
    print(f"  latency unwrap p50 {report['latency_ms']['unwrap_p50']} мс")
    print(f"отчёт: {target}")
    log_marker("M-EVAL", "cylinder", "DONE", applied=applied, cases=len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
