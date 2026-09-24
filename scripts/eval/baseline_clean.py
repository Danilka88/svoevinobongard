#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Честный baseline на подмножестве «вино в каталоге»: метрики пайплайна, near-dup, отказ и инструментация эскалации
# SCOPE: NEAR_DUP_NAME, THRESHOLDS, load_consensus, gt_of, find_siblings, scan_subset, compute_metrics,
#        escalation_table, build_report, main
# DEPENDS: json, subprocess, pathlib, numpy, packages.common.catalog, packages.common.config,
#          packages.common.logging, packages.common.text_norm, scripts.eval.eval_real100
# LINKS: M-EVAL, M-DATA
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# NEAR_DUP_NAME - порог схожести названия для поиска серии-двойника
# THRESHOLDS - пороги уверенности для таблицы эскалации
# GATEWAY - адрес зонда сканирования
# load_consensus - читает консенсус-метки разметки
# gt_of - определяет правильный slug по консенсусу
# find_siblings - ищет серию-двойника в каталоге
# scan_subset - прогон подмножества через живой gateway
# compute_metrics - считает метрики, near-dup, отказ, margin и задержки
# escalation_table - доля эскалаций и точность на неэскалированных по порогам
# build_report - пишет JSON и markdown-отчёт
# main - точка входа
# END_MODULE_MAP
"""Честный baseline на подмножестве «вино в каталоге + бутылка».

Считаем только там, где консенсус разметки принял метку и вино есть в каталоге: так метрика не
размывается случаями вне каталога. Важно: консенсус-метки не истина (они собраны моделями), поэтому
числа — относительные, для сравнения версий и порогов, а не абсолютное качество.

Отдельно инструментируется эскалация: какая доля запросов попала бы под гейт при разных порогах
уверенности. Калибровка порогов — следующий этап, здесь только доля и точность на остатке.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from packages.common.catalog import load_catalog  # noqa: E402
from packages.common.config import settings  # noqa: E402
from packages.common.logging import log_marker  # noqa: E402
from packages.common.text_norm import fold, similarity  # noqa: E402

NEAR_DUP_NAME = 0.86
THRESHOLDS = (0.90, 0.95, 0.99)
GATEWAY = "http://127.0.0.1:8080"

__all__ = ["NEAR_DUP_NAME", "THRESHOLDS", "GATEWAY", "build_report", "compute_metrics",
           "escalation_table", "find_siblings", "gt_of", "load_consensus", "main", "scan_subset"]


def load_consensus(path: str | Path) -> list[dict]:
    """Читает консенсус-метки разметки (принятые и эскалированные)."""
    target = Path(path)
    if not target.is_file():
        return []
    rows: list[dict] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _is_accepted(entry: dict) -> bool:
    """Принят ли случай консенсусом.

    Записи бывают двух форм: чекпоинт прогона (вложенный decision с вердиктом) и датасет разметки
    (плоская запись, куда попадают только принятые). Обе обрабатываются, иначе подмножество пустеет.
    """
    decision = entry.get("decision")
    if isinstance(decision, dict) and decision:
        return decision.get("verdict") == "accept"
    return True


def gt_of(entry: dict) -> str:
    """Правильный slug по консенсусу: лучший кандидат каталога при принятой метке."""
    if not _is_accepted(entry):
        return ""
    decision = entry.get("decision") or {}
    candidates = (decision.get("catalog_candidates") or entry.get("catalog_candidates") or [])
    return str(candidates[0]) if candidates else ""


def find_siblings(slug: str, catalog: dict) -> list[str]:
    """Ищет серию-двойника: та же винодельня и похожее название при иных годе/сладости/категории.

    Это самые трудные случаи: вина одной линейки отличаются только годом или типом, и поиск обязан
    различать их по этикетке.
    """
    base = catalog.get(slug) or {}
    if not base:
        return []
    siblings: list[str] = []
    for other, record in catalog.items():
        if other == slug:
            continue
        if fold(str(record.get("winery") or "")) != fold(str(base.get("winery") or "")):
            continue
        if similarity(str(record.get("name") or ""), str(base.get("name") or "")) < NEAR_DUP_NAME:
            continue
        differs = (str(record.get("year") or "") != str(base.get("year") or "")
                   or str(record.get("sweetness") or "") != str(base.get("sweetness") or "")
                   or str(record.get("category") or "") != str(base.get("category") or ""))
        if differs:
            siblings.append(other)
    return siblings


def _multipart(path: Path) -> tuple[bytes, str]:
    """Собирает тело multipart-запроса с изображением."""
    boundary = "----c060boundary"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
            f"filename=\"{path.name}\"\r\nContent-Type: image/webp\r\n\r\n").encode() + \
        path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def scan_subset(photos: list[Path], timeout: float = 120.0) -> list[dict]:
    """Сканирует подмножество через живой gateway."""
    rows: list[dict] = []
    for index, path in enumerate(photos, start=1):
        body, content_type = _multipart(path)
        record: dict = {"file": path.name}
        try:
            request = urllib.request.Request(f"{GATEWAY}/v1/scan", data=body,
                                             headers={"Content-Type": content_type})  # noqa: S310
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                payload = json.loads(response.read().decode())
            record.update({key: payload.get(key) for key in
                           ("slug", "confidence", "margin", "path", "not_found", "latency_ms")})
            record["top5"] = [item.get("slug") for item in (payload.get("top5") or [])][:5]
            record["card_image"] = ((payload.get("card") or {}).get("image_url") or "")[:80]
        except urllib.error.HTTPError as error:
            record["error"] = f"HTTP {error.code}"
        except Exception as error:  # noqa: BLE001
            record["error"] = type(error).__name__
        rows.append(record)
        print(f"  [{index}/{len(photos)}] {path.name[:26]:<28} {str(record.get('slug'))[:34]:<36} "
              f"conf={record.get('confidence')} margin={record.get('margin')}", flush=True)
    return rows


def _percentile(values: list[float], share: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(share * len(ordered)))], 2)


def compute_metrics(rows: list[dict], consensus: list[dict], catalog: dict,
                    escalations: list[dict] | None = None) -> dict:
    """Считает метрики на принятом подмножестве и отдельно на сериях-двойниках.

    Набор для оценки отказа берётся из эскалированных случаев: если он не передан, выводится из
    консенсуса по вердикту (для записей чекпоинта).
    """
    gt = {entry["file"]: gt_of(entry) for entry in consensus}
    strict = {entry["file"]: str(((entry.get("decision") or {}).get("in_catalog")
                                 or entry.get("in_catalog") or "")) == "strong" for entry in consensus}
    scored = [row for row in rows if gt.get(row.get("file")) and not row.get("error")]
    evaluated: list[dict] = []
    for row in scored:
        target = gt[row["file"]]
        siblings = find_siblings(target, catalog)
        evaluated.append({
            "file": row["file"], "slug": row.get("slug") or "", "gt": target,
            "hit1": row.get("slug") == target,
            "hit5": target in (row.get("top5") or []),
            "near_dup": bool(siblings), "siblings": len(siblings),
            "confidence": float(row.get("confidence") or 0.0),
            "margin": float(row.get("margin") or 0.0),
            "not_found": bool(row.get("not_found")),
            "latency_ms": float(row.get("latency_ms") or 0.0),
            "strict": bool(strict.get(row["file"])),
        })
    near_dup = [row for row in evaluated if row["near_dup"]]
    strict_only = [row for row in evaluated if row["strict"]]
    ood_source = escalations if escalations is not None else [entry for entry in consensus if not _is_accepted(entry)]
    ood_files = {str(entry.get("file")) for entry in ood_source}
    ood_rows = [row for row in rows if row.get("file") in ood_files]
    def rate(items: list[dict], key: str) -> float:
        return round(sum(1 for item in items if item[key]) / len(items), 4) if items else 0.0
    return {
        "subset": len(evaluated),
        "identity_at_1": rate(evaluated, "hit1"),
        "identity_at_5": rate(evaluated, "hit5"),
        "strict_only_at_1": rate(strict_only, "hit1"),
        "strict_subset": len(strict_only),
        "near_dup_subset": len(near_dup),
        "near_dup_at_1": rate(near_dup, "hit1"),
        "near_dup_at_5": rate(near_dup, "hit5"),
        "ood_cases": len(ood_rows),
        "ood_not_found_rate": round(sum(1 for row in ood_rows if row.get("not_found")) / len(ood_rows), 4) if ood_rows else 0.0,
        "margin_p50": _percentile([row["margin"] for row in evaluated], 0.5),
        "margin_p95": _percentile([row["margin"] for row in evaluated], 0.95),
        "latency_p50_ms": _percentile([row["latency_ms"] for row in evaluated], 0.5),
        "latency_p95_ms": _percentile([row["latency_ms"] for row in evaluated], 0.95),
        "rows": evaluated,
    }


def escalation_table(evaluated: list[dict]) -> list[dict]:
    """Доля эскалаций и точность на неэскалированных для набора порогов.

    Уверенность — сырой прокси на шкале поиска (не вероятность): калибровка ещё не сделана, поэтому
    таблица показывает поведение гейта, а не гарантию точности.
    """
    table: list[dict] = []
    notes = [("confidence", [row["confidence"] for row in evaluated])]
    for name, values in notes:
        for threshold in THRESHOLDS:
            escalated = [row for row in evaluated if row[name] < threshold]
            kept = [row for row in evaluated if row[name] >= threshold]
            table.append({
                "signal": name, "threshold": threshold,
                "escalation_share": round(len(escalated) / len(evaluated), 4) if evaluated else 0.0,
                "kept": len(kept),
                "accuracy_on_kept": round(sum(1 for row in kept if row["hit1"]) / len(kept), 4) if kept else 0.0,
            })
    # действующее правило поиска: отказ по порогам score_tau/margin_delta
    kept = [row for row in evaluated if not row["not_found"]]
    table.append({"signal": "score_tau+margindelta (текущее правило)",
                  "threshold": float(settings.score_tau),
                  "escalation_share": round(sum(1 for row in evaluated if row["not_found"]) / len(evaluated), 4) if evaluated else 0.0,
                  "kept": len(kept),
                  "accuracy_on_kept": round(sum(1 for row in kept if row["hit1"]) / len(kept), 4) if kept else 0.0})
    return table


def build_report(metrics: dict, escalation: list[dict], consensus: list[dict], models: dict,
                 processed: int, photos_total: int, out_json: str, out_md: str) -> None:
    """Пишет JSON и markdown-отчёт с числами и оговорками."""
    accept = sum(1 for entry in consensus if (entry.get("decision") or {}).get("verdict") == "accept")
    escalate = len(consensus) - accept
    payload = {"photos_total": photos_total, "processed": processed, "consensus_entries": len(consensus),
               "accepted": accept, "escalated": escalate, "models": models,
               "metrics": {key: value for key, value in metrics.items() if key != "rows"},
               "escalation": escalation, "rows": metrics.get("rows", [])}
    Path(out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Честный baseline на подмножестве «вино в каталоге»", "",
             f"Обработано фото: {processed} из {photos_total} · консенсус-меток: {len(consensus)} "
             f"(принято {accept}, эскалировано {escalate})", "",
             f"Модели: R1 = {models.get('primary', '—')}, R2 = OCR fusion, R3 = {models.get('qwen', '—')}, "
             f"R4 = {models.get('small', '—')}", "",
             "> Консенсус-метки собраны моделями, а не человеком: числа **относительные**, для сравнения",
             "> версий и порогов, а не абсолютное качество распознавания.", "",
             "## Метрики", "",
             "| Метрика | Значение |", "|---|---|",
             f"| подмножество (вино в каталоге) | {metrics['subset']} |",
             f"| identity@1 | **{metrics['identity_at_1']}** |",
             f"| identity@5 | **{metrics['identity_at_5']}** |",
             f"| strict-only @1 | {metrics['strict_only_at_1']} (подмножество {metrics['strict_subset']}) |",
             f"| near-dup @1 / @5 | {metrics['near_dup_at_1']} / {metrics['near_dup_at_5']} (подмножество {metrics['near_dup_subset']}) |",
             f"| отказ на вне-каталоге | {metrics['ood_not_found_rate']} (случаев {metrics['ood_cases']}) |",
             f"| margin p50 / p95 | {metrics['margin_p50']} / {metrics['margin_p95']} |",
             f"| задержка p50 / p95 | {metrics['latency_p50_ms']} / {metrics['latency_p95_ms']} мс |", "",
             "## Инструментация эскалации", "",
             "| Сигнал | Порог | Доля эскалаций | Осталось | Точность на остатке |", "|---|---|---|---|---|"]
    for row in escalation:
        lines.append(f"| {row['signal']} | {row['threshold']} | {row['escalation_share']} | "
                     f"{row['kept']} | {row['accuracy_on_kept']} |")
    lines += ["", "Уверенность — сырой прокси на шкале поиска, а не вероятность: калибровка порогов —",
              "следующий этап, поэтому таблица показывает поведение гейта, а не гарантию точности."]
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  отчёт: {out_json} · {out_md}")


def main(argv: list[str] | None = None) -> int:
    """Запускает baseline: прогон подмножества и отчёт."""
    parser = argparse.ArgumentParser(description="Честный baseline на подмножестве «вино в каталоге»")
    parser.add_argument("--consensus", default=str(_ROOT / "data" / "annotation" / "consensus.jsonl"))
    parser.add_argument("--escalations", default=str(_ROOT / "data" / "annotation" / "escalations.jsonl"))
    parser.add_argument("--photos", default=str(_ROOT / "data" / "samples" / "real100"))
    parser.add_argument("--scan", default=str(_ROOT / "reports" / "baseline_clean_scan.jsonl"))
    parser.add_argument("--out", default=str(_ROOT / "reports" / "baseline_clean.json"))
    parser.add_argument("--md", default=str(_ROOT / "reports" / "baseline_clean.md"))
    parser.add_argument("--labels", default=str(_ROOT / "reports" / "real100_multilabel4.jsonl"))
    parser.add_argument("--no-stack", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)
    consensus = load_consensus(args.consensus)
    escalations = load_consensus(args.escalations)
    if not consensus:
        print("консенсус-меток нет: сначала прогоните разметку")
        return 1
    # Сканируем и принятые, и эскалированные: вторые нужны для оценки корректного отказа.
    photos = [Path(args.photos) / entry["file"] for entry in consensus + escalations]
    photos = [path for path in photos if path.is_file()]
    # остальные фото набора — для оценки отказа вне каталога
    processed_rows: list[dict] = []
    already = Path(args.scan)
    if already.is_file():
        processed_rows = load_consensus(already)
        print(f"  из кэша прогона: {len(processed_rows)} строк")
    todo = [path for path in photos if not any(row.get("file") == path.name for row in processed_rows)]
    processes = []
    if todo and not args.no_stack:
        from scripts.eval.eval_real100 import start_stack, stop_stack  # noqa: PLC0415
        processes = start_stack()
    try:
        if todo:
            processed_rows += scan_subset(todo, args.timeout)
            with already.open("w", encoding="utf-8") as handle:
                for row in processed_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if processes:
            from scripts.eval.eval_real100 import stop_stack  # noqa: PLC0415
            stop_stack(processes)
    catalog = load_catalog(_ROOT / settings.catalog_jsonl)
    metrics = compute_metrics(processed_rows, consensus, catalog, escalations=escalations)
    escalation = escalation_table(metrics["rows"])
    models = {}
    labels = load_consensus(args.labels)
    if labels:
        models = (labels[0].get("models") or {})
    total = len(list(Path(args.photos).glob("*.webp")))
    build_report(metrics, escalation, consensus, models, len(processed_rows), total, args.out, args.md)
    print(f"===== BASELINE =====")
    print(f"  подмножество: {metrics['subset']} · identity@1 {metrics['identity_at_1']} · @5 {metrics['identity_at_5']}")
    print(f"  near-dup: {metrics['near_dup_subset']} · @1 {metrics['near_dup_at_1']} · OOD-отказ {metrics['ood_not_found_rate']}")
    for row in escalation:
        print(f"  гейт {row['signal']} ≥ {row['threshold']}: эскалаций {row['escalation_share']} · точность на остатке {row['accuracy_on_kept']}")
    log_marker("M-EVAL", "baseline_clean", "DONE", subset=metrics["subset"],
               identity_at_1=metrics["identity_at_1"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
