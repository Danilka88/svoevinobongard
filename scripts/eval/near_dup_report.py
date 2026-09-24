#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Отчёт по near-дубликатам: группы каталога и метрики дизамбигуации до и после
# SCOPE: load_consensus, build_report, main
# DEPENDS: json, pathlib, packages.common.catalog, packages.common.config, packages.common.logging,
#          packages.common.near_dup
# LINKS: M-EVAL, M-DATA
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# load_consensus - читает консенсус-метки и прогон сканера
# load_labeled - читает разметку C-041 и прогон по ста фото
# build_report - считает метрики на подмножестве дубликатов до и после дизамбигуации
# main - точка входа
# END_MODULE_MAP
"""Отчёт по near-дубликатам.

Считаем группы каталога и метрики на подмножестве, где правильный ответ попадает в группу дубликатов:
именно там система ошибается чаще всего. Дизамбигуация оценивается отдельно, поэтому видно её вклад.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from packages.common.catalog import load_catalog  # noqa: E402
from packages.common.config import settings  # noqa: E402
from packages.common.logging import log_marker  # noqa: E402
from packages.common.near_dup import build_groups, disambiguate, group_stats, near_dup_group  # noqa: E402

__all__ = ["build_report", "load_consensus", "load_labeled", "main"]


def load_consensus(consensus_path: str | Path, scan_path: str | Path) -> list[dict]:
    """Соединяет консенсус-метки (правильный ответ и сигналы) с прогоном сканера."""
    consensus = Path(consensus_path)
    scans = Path(scan_path)
    if not consensus.is_file() or not scans.is_file():
        return []
    gold: dict[str, dict] = {}
    for line in consensus.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        candidates = entry.get("catalog_candidates") or []
        if candidates:
            gold[entry["file"]] = {"gold": str(candidates[0]), "label": entry.get("label") or {}}
    rows: list[dict] = []
    for line in scans.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        reference = gold.get(row.get("file"))
        if reference:
            rows.append({**row, **reference})
    return rows


def load_labeled(labels_path: str | Path, scan_path: str | Path) -> list[dict]:
    """Второй источник подмножества: разметка C-041 и прогон сканера по ста фото.

    Консенсус-меток мало, поэтому подмножество дубликатов дополняется строгими кандидатами разметки:
    без этого метрики считались бы на одном-двух примерах.
    """
    labels = Path(labels_path)
    scans = Path(scan_path)
    if not labels.is_file() or not scans.is_file():
        return []
    gold: dict[str, dict] = {}
    for row in json.loads(labels.read_text(encoding="utf-8")).get("rows", []):
        strict = [candidate for candidate in (row.get("slug_candidates") or [])
                  if candidate.get("winery_score", 0) >= 0.86 and candidate.get("detail_score", 0) >= 0.86]
        if strict:
            gold[row["file"]] = {"gold": str(strict[0]["slug"]),
                                 "label": {"vintage": row.get("year"), "sweetness": row.get("sweetness")}}
    rows: list[dict] = []
    for line in scans.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        reference = gold.get(row.get("file"))
        if reference:
            rows.append({**row, **reference, "top5": row.get("top5") or [row.get("slug")]})
    return rows


def build_report(rows: list[dict], catalog: dict, groups: dict, out_json: str, out_md: str) -> dict:
    """Считает метрики до и после дизамбигуации на подмножестве дубликатов."""
    subset = [row for row in rows if near_dup_group(row["gold"], groups)]
    before_1 = before_5 = after_1 = ambiguous = 0
    examples: list[dict] = []
    for row in subset:
        # В разных прогонах top5 хранит либо строки, либо записи-кандидаты: приводим к slug.
        top = [item.get("slug") if isinstance(item, dict) else str(item)
               for item in (row.get("top5") or [])]
        top = [slug for slug in top if slug]
        if not top and row.get("slug"):
            top = [row["slug"]]
        if top and top[0] == row["gold"]:
            before_1 += 1
        if row["gold"] in top:
            before_5 += 1
        signals = {"vintage": row["label"].get("vintage"), "sweetness": row["label"].get("sweetness"),
                   "color": row["label"].get("color")}
        decision = disambiguate(top, signals, catalog, groups)
        if decision["ambiguous"]:
            ambiguous += 1
        if decision["slug"] == row["gold"]:
            after_1 += 1
        examples.append({"file": row["file"], "gold": row["gold"], "top5": top[:5],
                         "signals": signals, "decision": decision})
    total = len(subset) or 1
    report = {
        "catalog": len(catalog), "groups": group_stats(groups, catalog),
        "subset": len(subset),
        "identity_at_1_before": round(before_1 / total, 4),
        "identity_at_5_before": round(before_5 / total, 4),
        "identity_at_1_after": round(after_1 / total, 4),
        "ambiguous_share": round(ambiguous / total, 4),
        "examples": examples,
        "flag_default": settings.near_dup_enabled,
        "caveat": ("подмножество мало, метки модельные: числа относительные; дизамбигуация включается "
                   "флагом и по умолчанию выключена, поэтому метрики гейта не меняются"),
    }
    Path(out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Near-дубликаты и дизамбигуация (C-065)", "",
             f"Каталог: {report['catalog']} позиций · групп дубликатов: {report['groups']['groups']} "
             f"({report['groups']['positions']} позиций, {report['groups']['share_of_catalog']} каталога)",
             f"· максимальный размер группы: {report['groups']['max_size']}", "",
             "| Метрика | До дизамбигуации | После |", "|---|---|---|",
             f"| identity@1 на подмножестве ({report['subset']}) | {report['identity_at_1_before']} | {report['identity_at_1_after']} |",
             f"| identity@5 на подмножестве | {report['identity_at_5_before']} | — |",
             f"| доля неоднозначных | — | {report['ambiguous_share']} |", "",
             "## Примеры решений", "",
             "| Файл | Правильное вино | Сигналы | Решение |", "|---|---|---|---|"]
    for example in examples[:5]:
        decision = example["decision"]
        verdict = decision["slug"] or f"неоднозначно ({len(decision.get('candidates') or [])})"
        signals = ", ".join(f"{key}={value}" for key, value in example["signals"].items() if value)
        lines.append(f"| {example['file'][:22]} | {example['gold'][:32]} | {signals or '—'} | {verdict} |")
    lines += ["", "## Оговорки", "", f"- {report['caveat']}",
              "- Проверки сигналов продублированы в общем пакете: сервис матчера нельзя импортировать из общего кода."]
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    """Строит группы и пишет отчёты."""
    parser = argparse.ArgumentParser(description="Отчёт по near-дубликатам")
    parser.add_argument("--consensus", default=str(_ROOT / "data" / "annotation" / "consensus.jsonl"))
    parser.add_argument("--scan", default=str(_ROOT / "reports" / "baseline_clean_scan.jsonl"))
    parser.add_argument("--groups", default=str(_ROOT / "reports" / "near_dup_groups.json"))
    parser.add_argument("--labels", default=str(_ROOT / "reports" / "real100_labels.json"))
    parser.add_argument("--labels-scan", default=str(_ROOT / "reports" / "real100_scan.jsonl"))
    parser.add_argument("--out", default=str(_ROOT / "reports" / "near_dup_report.json"))
    parser.add_argument("--md", default=str(_ROOT / "reports" / "near_dup_report.md"))
    args = parser.parse_args(argv)
    catalog = load_catalog(_ROOT / settings.catalog_jsonl)
    groups = build_groups(catalog, settings.near_dup_threshold)
    Path(args.groups).write_text(json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = load_consensus(args.consensus, args.scan) + load_labeled(args.labels, args.labels_scan)
    seen: set[str] = set()
    unique: list[dict] = []
    for row in rows:
        if row['file'] not in seen:
            seen.add(row['file']); unique.append(row)
    rows = unique
    report = build_report(rows, catalog, groups, args.out, args.md)
    print("===== NEAR-DUP =====")
    print(f"  каталог {report['catalog']} · групп {report['groups']['groups']} · позиций в группах {report['groups']['positions']} ({report['groups']['share_of_catalog']})")
    print(f"  подмножество {report['subset']} · identity@1 до {report['identity_at_1_before']} · после {report['identity_at_1_after']} · неоднозначных {report['ambiguous_share']}")
    print(f"  файлы: {args.groups} · {args.out} · {args.md}")
    log_marker("M-EVAL", "near_dup_report", "DONE", groups=report["groups"]["groups"], subset=report["subset"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
