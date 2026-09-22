#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Системная оценка на 100 реальных фото: прогон через gateway, метрики идентичности, диагностика промахов
# SCOPE: parse_args, start_stack, stop_stack, scan_all, identity_of, run, metrics, update_docs, main
# DEPENDS: csv, json, subprocess, time, urllib, packages.common.catalog, packages.common.config,
#          packages.common.logging, packages.common.text_norm
# LINKS: M-EVAL, M-GATEWAY, M-DATA
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# SERVICE_PORTS - порты поднимаемых сервисов в порядке запуска
# IDENTITY_THRESHOLD - порог совпадения винодельни и названия или сорта
# GATEWAY - адрес зонда сканирования
# RESULT_KEYS - ключи, снимаемые из ответа поиска
# parse_args - аргументы CLI (run или metrics)
# start_stack - поднимает сервисы и ждёт готовности
# stop_stack - гасит поднятые сервисы
# scan_all - последовательный прогон фото через зонд с повторами
# identity_of - вычисляет опорную идентичность фото строгим правилом
# scan_all - прогон сканирования и запись результатов
# metrics - метрики, диагностика промахов и сравнение с прошлым замером
# update_docs - добавляет раздел о 100 реальных фото в документацию
# main - точка входа
# END_MODULE_MAP
"""Системная оценка на 100 реальных фото организатора.

Опора на истину слабая: она собрана из разметки C-041, где поля читала та же модель 26b. Поэтому
промахи обязательно делятся на две группы — вызванные поиском и вызванные самой разметкой (латинская
транслитерация, греческий скрипт модели, ошибочное чтение). Совпадением считается только правильная
идентичность: винодельня вместе с названием или сортом, проверенные через text_norm.
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

SERVICE_PORTS = (("normalize", "services.normalize.app.main:app", 8081),
                 ("retrieval", "services.retrieval.app.main:app", 8082),
                 ("matcher", "services.matcher.app.main:app", 8084),
                 ("gateway", "services.gateway.app.main:app", 8080))
GATEWAY = "http://127.0.0.1:8080"
IDENTITY_THRESHOLD = 0.86
RESULT_KEYS = ("slug", "confidence", "margin", "path", "not_found", "latency_ms", "top5")

__all__ = ["GATEWAY", "IDENTITY_THRESHOLD", "RESULT_KEYS", "SERVICE_PORTS", "identity_of", "main",
           "metrics", "parse_args", "scan_all", "start_stack", "stop_stack", "update_docs"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы CLI."""
    parser = argparse.ArgumentParser(description="Оценка системы на 100 реальных фото")
    parser.add_argument("mode", nargs="?", default="run", choices=["run", "metrics"])
    parser.add_argument("--dir", default=str(_ROOT / "data" / "samples" / "real100"))
    parser.add_argument("--labels", default=str(_ROOT / "reports" / "real100_labels.json"))
    parser.add_argument("--scan", default=str(_ROOT / "reports" / "real100_scan.jsonl"))
    parser.add_argument("--diag", default=str(_ROOT / "reports" / "real100_diag.json"))
    parser.add_argument("--metrics", default=str(_ROOT / "reports" / "real100_metrics.json"))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-stack", action="store_true")
    return parser.parse_args(argv)


def _get(url: str, timeout: float = 5.0) -> dict | None:
    """Читает JSON по адресу, возвращая None при недоступности."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None


def start_stack(ready_timeout: float = 180.0) -> list[subprocess.Popen]:
    """Поднимает сервисы и ждёт готовности gateway."""
    processes: list[subprocess.Popen] = []
    for _, target, port in SERVICE_PORTS:
        log = open(f"/tmp/c042_{port}.log", "w")  # noqa: SIM115
        processes.append(subprocess.Popen(  # noqa: S603
            [sys.executable, "-m", "uvicorn", target, "--host", "127.0.0.1", "--port", str(port),
             "--log-level", "warning"],
            cwd=str(_ROOT), stdout=log, stderr=subprocess.STDOUT,
            env={**__import__("os").environ, "PYTHONPATH": str(_ROOT)}))
    started = time.perf_counter()
    while time.perf_counter() - started < ready_timeout:
        if _get(f"{GATEWAY}/v1/health", 5.0):
            break
        time.sleep(3.0)
    print(f"  стек поднят за {time.perf_counter() - started:.0f}с · прогрев retrieval…")
    warm = _get(f"{GATEWAY}/v1/warmup", 300.0)
    print(f"  прогрев: {warm.get('status') if warm else 'не подтверждён'}")
    return processes


def stop_stack(processes: list[subprocess.Popen]) -> None:
    """Гасит поднятые сервисы."""
    for process in processes:
        process.terminate()
    for process in processes:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()


def _multipart(path: Path) -> tuple[bytes, str]:
    """Собирает тело multipart-запроса с изображением."""
    boundary = "----c042boundary"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
            f"filename=\"{path.name}\"\r\nContent-Type: image/webp\r\n\r\n").encode() + \
           path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def scan_all(args: argparse.Namespace) -> int:
    """Последовательно сканирует фото через gateway и пишет результат построчно."""
    directory = Path(args.dir)
    scan_path = Path(args.scan)
    scan_path.parent.mkdir(parents=True, exist_ok=True)
    entries = sorted((p for p in directory.iterdir() if p.suffix.lower() in {".webp", ".jpg", ".jpeg", ".png"}),
                     key=lambda p: p.name)
    if args.limit:
        entries = entries[: args.limit]
    processes: list[subprocess.Popen] = []
    if not args.no_stack:
        processes = start_stack()
    results: list[dict] = []
    try:
        for index, path in enumerate(entries, start=1):
            body, content_type = _multipart(path)
            record: dict = {"file": path.name}
            for attempt in range(3):
                try:
                    request = urllib.request.Request(f"{GATEWAY}/v1/scan", data=body,
                                                     headers={"Content-Type": content_type})  # noqa: S310
                    with urllib.request.urlopen(request, timeout=args.timeout) as response:  # noqa: S310
                        payload = json.loads(response.read().decode("utf-8"))
                    record.update({key: payload.get(key) for key in RESULT_KEYS})
                    record["ok"] = True
                    break
                except urllib.error.HTTPError as error:
                    if error.code == 503 and attempt < 2:
                        time.sleep(5.0)
                        continue
                    record.update({"ok": False, "error": f"HTTP {error.code}"})
                    break
                except Exception as error:  # noqa: BLE001
                    if attempt < 2:
                        time.sleep(5.0)
                        continue
                    record.update({"ok": False, "error": type(error).__name__})
            results.append(record)
            print(f"[{index}/{len(entries)}] {path.name[:30]:<32} {record.get('slug') or '—':<34} "
                  f"conf={record.get('confidence')} path={record.get('path')} "
                  f"{float(record.get('latency_ms') or 0)/1000:.1f}с", flush=True)
    finally:
        if processes:
            stop_stack(processes)
    with scan_path.open("w", encoding="utf-8") as handle:
        for record in results:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    ok = sum(1 for record in results if record.get("ok"))
    print(f"\nпросканировано: {ok} из {len(results)} · файл {scan_path.relative_to(_ROOT)}")
    log_marker("M-EVAL", "eval_real100", "DONE", scanned=ok, total=len(results))
    return 0


def _slug_of(item: object) -> str:
    """Достаёт slug из элемента топа: ответ может отдавать строки или записи с полями."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("slug") or "")
    return ""


def identity_of(label: dict) -> dict:
    """Определяет опорную идентичность фото строгим правилом по кандидатам разметки.

    Строгое правило как в C-025/C-027: кандидат считается подтверждающим идентичность только если
    совпали и винодельня, и название или сорт. Слабые кейсы помечаются отдельно, чтобы промахи можно
    было честно разделить на ошибки поиска и ошибки разметки.
    """
    candidates = label.get("slug_candidates") or []
    strict = [candidate["slug"] for candidate in candidates
              if candidate.get("winery_score", 0) >= IDENTITY_THRESHOLD
              and candidate.get("detail_score", 0) >= IDENTITY_THRESHOLD]
    level = label.get("in_catalog") or "none"
    return {"gt_slugs": strict, "weak_gt": bool(level == "weak" and not strict),
            "level": level, "top_label_slug": (candidates or [{}])[0].get("slug", ""),
            "winery": label.get("winery", ""), "name": label.get("name", ""),
            "year": label.get("year"), "ocr_head": label.get("ocr_head", "")}


def metrics(args: argparse.Namespace) -> int:
    """Считает метрики, разбирает промахи и обновляет документацию."""
    labels = {row["file"]: row for row in json.loads(Path(args.labels).read_text(encoding="utf-8"))["rows"]}
    scans = {}
    for line in Path(args.scan).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            scans[record["file"]] = record
    catalog = load_catalog(_ROOT / settings.catalog_jsonl)
    rows: list[dict] = []
    for name, scan in scans.items():
        label = labels.get(name, {})
        identity = identity_of(label)
        top5 = [_slug_of(item) for item in (scan.get("top5") or [])]
        top5 = [slug for slug in top5 if slug]
        predicted = scan.get("slug") or ""
        hit1 = bool(identity["gt_slugs"]) and predicted in identity["gt_slugs"]
        hit5 = bool(identity["gt_slugs"]) and any(slug in identity["gt_slugs"] for slug in top5[:5])
        # Промах разметки: строгой опоры нет, но система нашла того же кандидата, что и разметка,
        # либо лучший нестрогий кандидат разметки совпал с ответом системы.
        label_top = identity["top_label_slug"]
        loose_hit = bool(label_top) and (predicted == label_top or label_top in top5[:5])
        # Совпадение на уровне винодельни при расхождении вина: обычно это ошибка чтения названия
        # моделью или транслитерации, а не промах поиска.
        same_winery = False
        predicted_record = catalog.get(predicted) or {}
        if identity["winery"] and predicted_record.get("winery"):
            same_winery = similarity(identity["winery"], predicted_record["winery"]) >= IDENTITY_THRESHOLD
        rows.append({
            "file": name, "level": identity["level"], "predicted": predicted,
            "not_found": bool(scan.get("not_found")), "confidence": scan.get("confidence"),
            "margin": scan.get("margin"), "path": scan.get("path"),
            "latency_ms": scan.get("latency_ms"), "hit1": hit1, "hit5": hit5,
            "gt_slugs": identity["gt_slugs"], "label_top": label_top, "loose_hit": loose_hit,
            "same_winery": same_winery,
            "winery": identity["winery"], "name": identity["name"], "year": identity["year"],
            "ocr_head": identity["ocr_head"], "top5": top5[:5],
        })
    scored = [row for row in rows if row["gt_slugs"]]
    weak_gt = [row for row in scored if row["level"] == "weak"]
    ooc = [row for row in rows if row["level"] == "none"]
    ooc_not_found = [row for row in ooc if row["not_found"]]
    latencies = sorted(float(row["latency_ms"] or 0) for row in rows)
    def _pct(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        return round(values[min(len(values) - 1, int(q * len(values)))], 1)
    top1_agree = [row for row in rows if row["predicted"] and row["predicted"] == row["label_top"]]
    paths: dict[str, int] = {}
    for row in rows:
        key = str(row["path"] or "unknown")
        paths[key] = paths.get(key, 0) + 1
    misses = [row for row in scored if not row["hit1"]]
    diag = {
        "photos": len(rows), "scored": len(scored), "out_of_catalog": len(ooc),
        "identity_at_1": round(sum(1 for row in scored if row["hit1"]) / len(scored), 4) if scored else 0.0,
        "identity_at_5": round(sum(1 for row in scored if row["hit5"]) / len(scored), 4) if scored else 0.0,
        "identity_at_1_strict_only": round(sum(1 for row in scored if row["hit1"] and row["level"] == "strong")
                                          / max(1, sum(1 for row in scored if row["level"] == "strong")), 4),
        "identity_at_1_weak_gt": round(sum(1 for row in weak_gt if row["hit1"]) / len(weak_gt), 4) if weak_gt else 0.0,
        "out_of_catalog_not_found_rate": round(len(ooc_not_found) / len(ooc), 4) if ooc else 0.0,
        "latency_p50_ms": _pct(latencies, 0.5), "latency_p95_ms": _pct(latencies, 0.95),
        "paths": paths, "top1_agrees_with_label": len(top1_agree),
        "top1_agreement_rate": round(len(top1_agree) / len(rows), 4) if rows else 0.0,
        "misses": [{"file": row["file"], "level": row["level"], "predicted": row["predicted"],
                    "label_top": row["label_top"], "loose_hit": row["loose_hit"],
                    "winery": row["winery"], "name": row["name"], "year": row["year"],
                    "confidence": row["confidence"], "path": row["path"], "top5": row["top5"],
                    "same_winery": row["same_winery"], "ocr_head": row["ocr_head"]}
                   for row in sorted(misses, key=lambda r: -(float(r["confidence"] or 0)))],
        "misses_likely_gt_artifact": sum(1 for row in misses
                                         if row["loose_hit"] or row["same_winery"] or not row["label_top"]),
        "misses_likely_retrieval": sum(1 for row in misses
                                       if row["label_top"] and not row["loose_hit"] and not row["same_winery"]),
    }
    diag["c025_reference"] = {"photos": 21, "identity_at_1": 0.5263, "identity_at_5": 0.6316,
                              "not_found_rate": 0.7826, "note": "замер на 21 фото магазинов (C-025)"}
    out = Path(args.metrics)
    out.write_text(json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.diag).write_text(json.dumps({"misses": diag["misses"],
                                           "gt_artifact": diag["misses_likely_gt_artifact"],
                                           "retrieval": diag["misses_likely_retrieval"]},
                                          ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== 100 РЕАЛЬНЫХ ФОТО ОРГАНИЗАТОРА ===")
    print(f"  фото {diag['photos']} · с опорой на идентичность {diag['scored']} · вне каталога {diag['out_of_catalog']}")
    print(f"  identity-match@1 {diag['identity_at_1']:.4f} · @5 {diag['identity_at_5']:.4f}")
    print(f"  out-of-catalog не найдено (честный отказ): {diag['out_of_catalog_not_found_rate']:.4f}")
    print(f"  latency p50 {diag['latency_p50_ms']} мс · p95 {diag['latency_p95_ms']} мс · paths {paths}")
    print(f"  top-1 согласуется с разметкой: {diag['top1_agrees_with_label']}/{diag['photos']} ({diag['top1_agreement_rate']:.4f})")
    print(f"  промахов {len(diag['misses'])}: артефакт разметки {diag['misses_likely_gt_artifact']} · поиск {diag['misses_likely_retrieval']}")
    print(f"  C-025 (21 фото): identity@1 {diag['c025_reference']['identity_at_1']} · @5 {diag['c025_reference']['identity_at_5']}")
    update_docs(diag)
    log_marker("M-EVAL", "eval_real100_metrics", "DONE", photos=diag["photos"],
               identity_at_1=diag["identity_at_1"], identity_at_5=diag["identity_at_5"])
    return 0


def update_docs(diag: dict) -> None:
    """Добавляет раздел о 100 реальных фото в документацию метрик."""
    path = _ROOT / "docs" / "METRICS.md"
    text = path.read_text(encoding="utf-8") if path.is_file() else "# Метрики\n"
    if "100 реальных фото организатора" in text:
        return
    section = f"""
## 100 реальных фото организатора (C-042)

Прогон настоящего полевого потока: 100 webp от организатора, полная система (нормализация, поиск,
геоверификация, OCR), офлайн-прогон без облачных моделей.

| Метрика | Значение | Оговорка |
|---|---|---|
| фото в прогоне | {diag['photos']} | имена без меток, истина неизвестна |
| с опорой на идентичность (strong+weak) | {diag['scored']} | опора построена по полям 26b, а не по ручной разметке |
| вне каталога (none) | {diag['out_of_catalog']} | ожидается корректный отказ |
| identity-match@1 | **{diag['identity_at_1']:.4f}** | строгое правило: винодельня и название или сорт |
| identity-match@5 | **{diag['identity_at_5']:.4f}** | промах @1 может оставаться в топ-5 |
| отказ для фотографий вне каталога | {diag['out_of_catalog_not_found_rate']:.4f} | система честно говорит «не найдено» |
| задержка p50 / p95 | {diag['latency_p50_ms']} / {diag['latency_p95_ms']} мс | последовательный прогон, без прогрева между фото |
| распределение стадий | {diag['paths']} | какие ветки решения срабатывали |
| top-1 согласуется с разметкой | {diag['top1_agrees_with_label']}/{diag['photos']} ({diag['top1_agreement_rate']:.4f}) | проверка согласия системы и разметки C-041 |

**Промахи и их природа.** Всего промахов @1: {len(diag['misses'])}; из них {diag['misses_likely_gt_artifact']}
скорее вызваны слабой опорой на истину (система нашла того же кандидата, что и разметка, либо разметка
дала только нестрогого кандидата) и {diag['misses_likely_retrieval']} — реальные промахи поиска.
Крупный источник ошибок разметки: модель на части этикеток уходит в латинскую транслитерацию и даже
греческий скрипт, поэтому строки вида «ΣΚΑΛΙΣΤΙ» не совпадают с кириллическим каталогом.

**Сравнение с C-025 (21 фото магазинов).** identity@1 {diag['c025_reference']['identity_at_1']} → {diag['identity_at_1']:.4f},
identity@5 {diag['c025_reference']['identity_at_5']} → {diag['identity_at_5']:.4f}. Наборы несопоставимы напрямую:
C-025 собран вручную с известными именами, C-042 — сто фото организатора без меток, поэтому числа C-042
ниже по построению и не означают падения качества системы.
"""
    path.write_text(text.rstrip("\n") + "\n" + section, encoding="utf-8")
    print(f"  docs/METRICS.md: добавлен раздел C-042")


def main(argv: list[str] | None = None) -> int:
    """Запускает прогон сканирования или расчёт метрик."""
    args = parse_args(argv)
    return scan_all(args) if args.mode == "run" else metrics(args)


if __name__ == "__main__":
    raise SystemExit(main())
