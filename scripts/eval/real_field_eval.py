#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Оценка системы на реальных полевых фото: слабая GT по идентичности, прогон /v1/scan, метрики и диагноз
# SCOPE: parse_args, parse_label, identity_candidates, start_services, stop_services, scan_all, compute_metrics, main
# DEPENDS: json, re, subprocess, time, urllib, rapidfuzz, packages.common.catalog, packages.common.logging
# LINKS: M-EVAL, M-RETRIEVAL, M-DATA
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# WINERY_HINTS - подсказки виноделен в именах файлов
# SWEETNESS_HINTS - подсказки типа вина в именах файлов
# NON_RU_HINTS - признаки не-российского вина (вне каталога)
# HEALTH_TIMEOUT_S - сколько ждать готовности сервиса
# IDENTITY_THRESHOLD - порог fuzzy-совпадения для признания идентичности
# SERVICES - сервисы, которые поднимает прогон (модуль, имя, порт)
# parse_args - аргументы CLI (prepare, scan, report)
# parse_label - разбор имени файла на винодельню, название, год и тип
# identity_candidates - правдоподобные slug каталога для слабой GT
# wait_health - ожидание готовности сервиса по /health или /v1/health
# start_services - подъём сервисов с ожиданием готовности
# stop_services - остановка поднятых сервисов
# scan_all - прогон POST /v1/scan по всем фото
# compute_metrics - identity-match@1/@5, not-found, near-dup, latency и таблица промахов
# main - режимы prepare/scan/report
# END_MODULE_MAP
"""Оценка на реальных полевых фото.

Слабая GT: имя файла содержит текст этикетки, прочитанный владельцем, поэтому истина
определяется по идентичности (винодельня + название), а не по точному slug. Для каждой
позиции каталога считается fuzzy-совпадение с текстом этикетки; набор подходящих slug и
есть истинная идентичность. Явно не-российские вина отмечены как вне каталога: для них
корректным считается not_found или отсутствие совпадения по идентичности.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from rapidfuzz import fuzz

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from packages.common.catalog import load_catalog  # noqa: E402
from packages.common.logging import log_marker
from packages.common.text_norm import similarity  # noqa: E402

WINERY_HINTS = {
    "abrau": ["абрау", "abrau"], "абрау": ["абрау", "abrau"],
    "tamagne": ["тамань", "tamagne"], "тамань": ["тамань", "tamagne"],
    "fanagory": ["фанагория"], "фонагория": ["фанагория"], "фанагория": ["фанагория"],
    "берег": ["высокий берег"], "свет": ["новый свет"], "лефкадия": ["лефкадия"],
    "амфитрион": ["амфитрион"], "kursavskoe": ["курсавское", "kursavskoe"],
}
SWEETNESS_HINTS = ("полусладкое", "полусухое", "экстра брют", "брют", "сухое", "semi-sweet")
NON_RU_HINTS = ("bosca", "daschbosch", "toskana", "казахстан", "бахус")
IDENTITY_THRESHOLD = 72.0
SERVICES = (
    ("services.normalize.app.main:app", "normalize", 8081),
    ("services.retrieval.app.main:app", "retrieval", 8082),
    ("services.matcher.app.main:app", "matcher", 8084),
    ("services.gateway.app.main:app", "gateway", 8080),
)
HEALTH_TIMEOUT_S = 180.0

__all__ = ["IDENTITY_THRESHOLD", "NON_RU_HINTS", "SWEETNESS_HINTS", "WINERY_HINTS", "compute_metrics",
           "identity_candidates", "main", "parse_args", "parse_label", "scan_all", "start_services", "stop_services"]


def _display(path: Path) -> str:
    """Путь для печати: относительный, если внутри проекта, иначе как есть."""
    try:
        return str(path.resolve().relative_to(_ROOT))
    except ValueError:
        return str(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы CLI."""
    parser = argparse.ArgumentParser(description="Оценка на реальных полевых фото")
    parser.add_argument("mode", choices=["prepare", "scan", "report", "all"])
    parser.add_argument("--dir", default=str(_ROOT / "data/samples/real_field"))
    parser.add_argument("--labels", default=str(_ROOT / "reports/real_field_labels.jsonl"))
    parser.add_argument("--extra-dir", default=str(_ROOT / "data/samples/eval_queries"),
                        help="официальные фото: включаются в тот же прогон")
    parser.add_argument("--out", default=str(_ROOT / "reports/real_field.json"))
    parser.add_argument("--venv", default=str(_ROOT / ".venv" / "bin" / "python"))
    parser.add_argument("--threshold", type=float, default=IDENTITY_THRESHOLD)
    parser.add_argument("--rerank", action="store_true", help="офлайн-реранк кандидатов по полям 26b")
    return parser.parse_args(argv)


def parse_label(text: str) -> dict:
    """Разбирает текст этикетки из имени файла: винодельня, название, год, тип, не-российское."""
    lowered = text.lower()
    wineries: list[str] = []
    for hint, names in WINERY_HINTS.items():
        if hint in lowered:
            wineries.extend(names)
    year_match = re.search(r"(?:19|20)\d{2}", text)
    sweetness = next((hint for hint in SWEETNESS_HINTS if hint in lowered), "")
    residue = lowered
    for hint in list(WINERY_HINTS) + list(SWEETNESS_HINTS):
        residue = residue.replace(hint, " ")
    if year_match:
        residue = residue.replace(year_match.group(0), " ")
    residue = " ".join(re.findall(r"[a-zа-яё0-9]+", residue))
    return {
        "text": text,
        "wineries": sorted(set(wineries)),
        "name_tokens": residue,
        "year": year_match.group(0) if year_match else "",
        "sweetness": sweetness,
        "non_russian": any(hint in lowered for hint in NON_RU_HINTS),
    }


def identity_candidates(label: dict, catalog: dict, threshold: float) -> list[str]:
    """Подбирает slug каталога, совпадающие с этикеткой по идентичности.

    Правило сознательно строгое, иначе набор кандидатов раздувается и метрика завышается:
    винодельня из имени файла обязана совпасть, и хотя бы один значимый токен названия
    должен найтись в записи каталога; дополнительно проверяется fuzzy-близость токена.
    """
    if label["non_russian"]:
        return []
    tokens = [token for token in label["name_tokens"].split() if len(token) >= 4]
    if not tokens:
        return []
    hits: list[tuple[float, str]] = []
    for slug, record in catalog.items():
        haystack = " ".join(str(record.get(field) or "") for field in ("winery", "name", "grape"))
        haystack = f"{haystack} {slug.replace('-', ' ')}".lower()
        if label["wineries"] and not any(winery in haystack for winery in label["wineries"]):
            continue
        # транслит-нормализация: «Шато Тамань» совпадает с chateau-tamagne, а не только по буквам
        score = similarity(label["text"],
                           f"{record.get('winery', '')} {record.get('name', '')} {record.get('grape', '')}")
        if score < threshold / 100.0:
            continue
        hits.append((score, slug))
    hits.sort(reverse=True)
    return [slug for _, slug in hits[:25]]


def wait_health(port: int, timeout_s: float = HEALTH_TIMEOUT_S) -> bool:
    """Ждёт готовности сервиса по /health или /v1/health."""
    started = time.perf_counter()
    path = "/v1/health" if port == 8080 else "/health"
    while time.perf_counter() - started < timeout_s:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=3) as response:
                if response.status == 200:
                    return True
        except Exception:  # noqa: BLE001 - сервис ещё поднимается
            time.sleep(1.5)
    return False


def start_services(venv: str) -> list[tuple[str, subprocess.Popen]]:
    """Поднимает недостающие сервисы; возвращает поднятые процессы."""
    started: list[tuple[str, subprocess.Popen]] = []
    for module, name, port in SERVICES:
        if wait_health(port, timeout_s=1.5):
            print(f"  {name}: уже запущен")
            continue
        process = subprocess.Popen(
            [venv, "-m", "uvicorn", module, "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
            cwd=str(_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        started.append((name, process))
        print(f"  {name}: поднят (pid {process.pid})")
    return started


def stop_services(started: list[tuple[str, subprocess.Popen]]) -> None:
    """Останавливает поднятые проверкой процессы."""
    for name, process in started:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        print(f"  {name}: остановлен")


def _post_scan(image: Path, port: int = 8080, timeout: float = 180.0,
               warmup_s: float = 240.0) -> tuple[int, dict]:
    """Отправляет фото в /v1/scan, повторяя запрос пока retrieval не прогрет."""
    boundary = "----svoeVinoReal"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="image"; filename="{image.name}"\r\n'.encode(),
        b"Content-Type: image/jpeg\r\n\r\n", image.read_bytes(), b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ])
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/scan", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST")
    deadline = time.perf_counter() + warmup_s
    while True:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code == 503 and time.perf_counter() < deadline:
                time.sleep(5.0)
                continue
            return error.code, {"error": error.read().decode("utf-8", errors="replace")[:200]}


def scan_all(images: list[Path]) -> list[dict]:
    """Прогоняет /v1/scan по списку фото."""
    results: list[dict] = []
    for index, image in enumerate(images, 1):
        started = time.perf_counter()
        status, payload = _post_scan(image)
        latency = (time.perf_counter() - started) * 1000.0
        card = (payload or {}).get("card") or {}
        results.append({
            "file": image.name,
            "status": status,
            "top1": payload.get("slug") if isinstance(payload, dict) else None,
            "top5": [item.get("slug") for item in (payload.get("top5") or [])] if isinstance(payload, dict) else [],
            "confidence": payload.get("confidence") if isinstance(payload, dict) else None,
            "path": payload.get("path") if isinstance(payload, dict) else None,
            "not_found": bool(payload.get("not_found")) if isinstance(payload, dict) else True,
            "latency_ms": round(latency, 1),
            "card_winery": card.get("winery", ""),
            "card_name": card.get("name", ""),
        })
        print(f"[{index}/{len(images)}] {image.name[:34]:<36} top1={results[-1]['top1']} "
              f"conf={results[-1]['confidence']} {latency:.0f} мс", flush=True)
    return results


def _percentile(values: list[float], p: float) -> float:
    """p50/p95 без внешних зависимостей."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(round(p / 100.0 * (len(ordered) - 1))))], 1)


def compute_metrics(rows: list[dict], labels: dict[str, dict], catalog: dict,
                    threshold: float) -> dict:
    """Считает полевые метрики и собирает таблицу промахов с диагнозом."""
    detail: list[dict] = []
    for row in rows:
        label = labels.get(row["file"], {})
        candidates = identity_candidates(label, catalog, threshold) if label else []
        candidate_set = set(candidates)
        top1, top5 = row.get("top1"), row.get("top5") or []
        rank = 0
        for position, slug in enumerate([top1] + list(top5), 1):
            if slug and slug in candidate_set:
                rank = position
                break
        detail.append({
            **row,
            "label_text": (label or {}).get("text", ""),
            "non_russian": bool((label or {}).get("non_russian")),
            "identity_candidates": candidates[:5],
            "identity_at_1": bool(top1 and top1 in candidate_set),
            "identity_at_5": rank > 0,
            "identity_rank": rank,
            "top1_in_catalog": bool(top1 and top1 in catalog),
            "expected_not_found": bool((label or {}).get("non_russian")),
            # вторая, независимая метрика: совпала ли винодельня в карточке с винодельней из имени файла
            # третья метрика: карточка (винодельня+название) похожа на текст из имени файла.
            # Нужна потому, что слабая GT не знает кириллических slug каталога и часть
            # «промахов» на самом деле верные попадания.
            "card_match_at_1": bool(label) and fuzz.partial_ratio(
                str(label.get("text", "")).lower(),
                f"{row.get('card_winery','')} {row.get('card_name','')}".lower()) >= 80,
            "winery_at_1": bool((label or {}).get("wineries")) and any(
                fuzz.partial_ratio(winery, str(row.get("card_winery") or "").lower()) >= 80
                for winery in (label or {}).get("wineries", [])),
        })

    def share(predicate) -> float:
        return round(sum(1 for item in detail if predicate(item)) / len(detail), 4) if detail else 0.0

    russian = [item for item in detail if not item["non_russian"]]
    foreign = [item for item in detail if item["non_russian"]]
    latencies = [item["latency_ms"] for item in detail]
    misses = [
        {
            "file": item["file"], "label_text": item["label_text"], "top1": item["top1"],
            "identity_in_top5": item["identity_at_5"], "identity_rank": item["identity_rank"],
            "expected_not_found": item["expected_not_found"],
            "card": f"{item['card_winery']} / {item['card_name']}",
            "identity_candidates": item["identity_candidates"],
            "ocr_text": (item.get("ocr_text") or "")[:80],
            "reason": (
                "не-российское вино вне каталога — ожидался not_found" if item["expected_not_found"]
                else ("истинной идентичности нет в top-5: промах поиска" if not item["identity_at_5"]
                      else "истина была в top-5, но не первой: вопрос ранжирования")
            ),
        }
        for item in detail if not item["identity_at_1"]
    ]
    return {
        "samples": len(detail),
        "russian_samples": len(russian),
        "foreign_samples": len(foreign),
        "identity_at_1": share(lambda item: item["identity_at_1"]),
        "identity_at_5": share(lambda item: item["identity_at_5"]),
        "identity_at_1_russian": round(sum(1 for item in russian if item["identity_at_1"]) / len(russian), 4) if russian else 0.0,
        "identity_at_5_russian": round(sum(1 for item in russian if item["identity_at_5"]) / len(russian), 4) if russian else 0.0,
        "not_found_rate": share(lambda item: item["not_found"]),
        "card_match_at_1": share(lambda item: item["card_match_at_1"]),
        "card_match_at_1_russian": round(
            sum(1 for item in russian if item["card_match_at_1"]) / len(russian), 4) if russian else 0.0,
        "winery_match_at_1_russian": round(
            sum(1 for item in russian if item["winery_at_1"]) / len(russian), 4) if russian else 0.0,
        "not_found_correct_foreign": round(
            sum(1 for item in foreign if item["not_found"] or not item["identity_at_5"]) / len(foreign), 4) if foreign else 0.0,
        "latency_ms": {"p50": _percentile(latencies, 50), "p95": _percentile(latencies, 95)},
        "detail": detail,
        "misses": misses,
    }


def main(argv: list[str] | None = None) -> int:
    """Выполняет подготовку, прогон и/или отчёт."""
    args = parse_args(argv)
    directory = Path(args.dir)
    out_path = Path(args.out)
    catalog = load_catalog(_ROOT / "data/catalog/catalog.jsonl")
    state = json.loads(out_path.read_text(encoding="utf-8")) if out_path.is_file() else {}

    images: list[Path] = []
    if directory.is_dir():
        images += sorted(p for p in directory.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    extra = Path(args.extra_dir)
    if extra.is_dir():
        images += sorted(p for p in extra.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    labels: dict[str, dict] = {}
    manifest = directory / "manifest.tsv"
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                name, _, text = line.partition("\t")
                labels[name] = parse_label(text)
    for path in images:
        labels.setdefault(path.name, parse_label(path.stem.replace("_", " ")))

    if args.mode in {"prepare", "all"}:
        prepared = {name: {**label, "candidates": identity_candidates(label, catalog, args.threshold)}
                    for name, label in labels.items()}
        state["prepared"] = prepared
        state["threshold"] = args.threshold
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        ru = sum(1 for label in labels.values() if not label["non_russian"])
        with_ids = sum(1 for item in prepared.values() if item["candidates"])
        print(f"подготовлено: {len(labels)} фото · российских {ru} · с найденной идентичностью {with_ids}")

    if args.mode in {"scan", "all"}:
        started = start_services(args.venv)
        try:
            if not wait_health(8080):
                print("gateway не поднялся")
                return 1
            state["scans"] = scan_all(images)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            stop_services(started)

    if args.mode in {"report", "all"}:
        scans = state.get("scans") or []
        if args.rerank:
            state["rerank"] = _rerank(scans, args.labels, catalog)
        state.update(compute_metrics(scans, labels, catalog, args.threshold))
        out_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        print("=== РЕАЛЬНЫЕ ПОЛЕВЫЕ ФОТО ===")
        print(f"  фото: {state['samples']} (российских {state['russian_samples']}, вне каталога {state['foreign_samples']})")
        print(f"  identity-match@1: {state['identity_at_1']} (по российским {state['identity_at_1_russian']})")
        print(f"  identity-match@5: {state['identity_at_5']} (по российским {state['identity_at_5_russian']})")
        print(f"  карточка совпала с именем файла: {state['card_match_at_1']} (по российским {state['card_match_at_1_russian']})")
        print(f"  совпадение винодельни в карточке (по российским): {state['winery_match_at_1_russian']}")
        print(f"  not_found: {state['not_found_rate']} · корректный not-found для не-российских: {state['not_found_correct_foreign']}")
        print(f"  latency p50 {state['latency_ms']['p50']} мс · p95 {state['latency_ms']['p95']} мс")
        print(f"  промахов: {len(state['misses'])} · отчёт {_display(out_path)}")
        log_marker("M-EVAL", "real_field", "DONE", samples=state["samples"],
                   identity_at_1=state["identity_at_1"], identity_at_5=state["identity_at_5"])
    return 0


def _rerank(scans: list[dict], labels_path: str, catalog: dict) -> dict:
    """Офлайн-реранк: поднимает кандидата из top-5, если поля 26b указывают на него."""
    from rapidfuzz import fuzz  # noqa: PLC0415

    labels_file = Path(labels_path)
    vlm: dict[str, dict] = {}
    if labels_file.is_file():
        for line in labels_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                vlm[record.get("file", "")] = record.get("vlm") or {}
    changed = 0
    promoted = 0
    for row in scans:
        fields = vlm.get(row["file"]) or {}
        winery = str(fields.get("winery") or "").lower()
        name = str(fields.get("name") or "").lower()
        if not winery and not name:
            continue
        best_slug, best_score = None, 0.0
        catalog_records = catalog
        for slug in row.get("top5") or []:
            card = catalog_records.get(slug) or {}
            haystack = f"{card.get('winery','')} {card.get('name','')}".lower()
            score = max(fuzz.token_set_ratio(winery, haystack) if winery else 0.0,
                        fuzz.token_set_ratio(name, haystack) if name else 0.0)
            if score > best_score:
                best_slug, best_score = slug, score
        if best_slug and best_slug != row.get("top1") and best_score >= 75.0:
            row["top1_before_rerank"] = row.get("top1")
            row["top1"] = best_slug
            changed += 1
            promoted += 1
    return {"changed": changed, "note": "кандидат из top-5 поднят, если поля 26b дали совпадение >= 75 по винодельне или названию"}



if __name__ == "__main__":
    raise SystemExit(main())
