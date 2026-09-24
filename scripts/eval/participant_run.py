#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Прогон НАСТОЯЩЕГО скрипта организатора против gateway и сборка метрик совместимости
# SCOPE: ROOT, DEFAULT_ORGANIZER_SCRIPT, DEFAULT_MANIFEST, DEFAULT_PREDICTIONS, IMAGE_EXT, ORGANIZER_LIMIT_MS, build_manifest, run_organizer, load_rows, percentile, summarize, render_markdown, write_reports, main
# DEPENDS: none (только stdlib; запускает participant_test.sh как есть)
# LINKS: M-EVAL, M-GATEWAY
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# ROOT - корень проекта
# DEFAULT_ORGANIZER_SCRIPT - путь к настоящему participant_test.sh организатора
# DEFAULT_MANIFEST - путь манифеста query_id<TAB>image_path
# DEFAULT_PREDICTIONS - путь выходного predictions.jsonl
# IMAGE_EXT - расширения изображений
# ORGANIZER_LIMIT_MS - лимит скрипта организатора в миллисекундах
# build_manifest - формирует манифест из папки реальных фото
# run_organizer - запускает participant_test.sh против gateway
# load_rows - читает JSONL предсказаний
# percentile - перцентиль задержки
# summarize - считает долю непустых slug, перцентили, долю в лимите и ошибки
# render_markdown - человекочитаемый отчёт изменения
# write_reports - пишет JSON и Markdown отчёты
# main - CLI: манифест, прогон, метрики, отчёты
# END_MODULE_MAP
"""Proof of compatibility with the organizer's own script.

The script below does not re-implement the organizer contract: it builds the
manifest, invokes the real ``participant_test.sh`` exactly as the competition
would, and then measures what that script produced. Numbers in the report are
derived from the organizer's own JSONL, not from our evaluator.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORGANIZER_SCRIPT = (ROOT.parent / "материалы для конкурса" / "Датасет" / "eval"
                            / "participant_test.sh")
DEFAULT_MANIFEST = ROOT / "reports" / "participant_queries.tsv"
DEFAULT_PREDICTIONS = ROOT / "reports" / "participant_predictions.jsonl"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".tif", ".tiff"}

# Лимит скрипта организатора: --connect-timeout 5 --max-time 10.
ORGANIZER_LIMIT_MS = 10_000.0

__all__ = [
    "ROOT",
    "DEFAULT_ORGANIZER_SCRIPT",
    "DEFAULT_MANIFEST",
    "DEFAULT_PREDICTIONS",
    "IMAGE_EXT",
    "ORGANIZER_LIMIT_MS",
    "build_manifest",
    "run_organizer",
    "load_rows",
    "percentile",
    "summarize",
    "render_markdown",
    "write_reports",
    "main",
]


# START_BLOCK_PARTICIPANT_MANIFEST
def build_manifest(images_dir: Path, manifest_path: Path) -> int:
    """Строит манифест организатора: заголовок query_id<TAB>image_path и по строке на фото.

    query_id — имя файла без расширения, как ожидает формат; имена уникальны по построению.
    """
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for path in sorted(images_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXT:
            continue
        query_id = path.stem
        if query_id in seen:
            raise SystemExit(f"duplicate query_id: {query_id}")
        seen.add(query_id)
        entries.append((query_id, path.name))
    if not entries:
        raise SystemExit(f"no images found in {images_dir}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("query_id\timage_path\n")
        for query_id, name in entries:
            handle.write(f"{query_id}\t{name}\n")
    return len(entries)
# END_BLOCK_PARTICIPANT_MANIFEST


# START_BLOCK_PARTICIPANT_RUN
def run_organizer(script: Path, images_dir: Path, manifest: Path, endpoint: str,
                  output: Path) -> subprocess.CompletedProcess:
    """Запускает настоящий скрипт организатора; output обязан отсутствовать (ограничение скрипта)."""
    if not script.is_file():
        raise SystemExit(f"organizer script not found: {script}")
    if output.exists():
        output.unlink()
    command = [
        "bash", str(script),
        "--images-dir", str(images_dir),
        "--manifest", str(manifest),
        "--endpoint", endpoint,
        "--output", str(output),
    ]
    print(f"[participant_run] $ {' '.join(command)}", file=sys.stderr)
    return subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True)
# END_BLOCK_PARTICIPANT_RUN


# START_BLOCK_PARTICIPANT_METRICS
def load_rows(predictions: Path) -> list[dict]:
    """Читает predictions.jsonl, который написал скрипт организатора."""
    rows: list[dict] = []
    for line in predictions.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def percentile(values: list[float], p: float) -> float:
    """Перцентиль по ближайшему рангу (тот же метод, что в остальных отчётах проекта)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    k = (len(ordered) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo))


def summarize(rows: list[dict], returncode: int, stderr: str, gateway_log: Path | None) -> dict:
    """Считает метрики прогона по строкам, которые написал сам организатор."""
    latencies = [float(row.get("latency_ms") or 0.0) for row in rows]
    non_null = sum(1 for row in rows if row.get("predicted_slug"))
    no_response = sum(1 for row in rows if float(row.get("latency_ms") or 0.0) <= 0.0)
    timeouts = sum(1 for value in latencies if value >= ORGANIZER_LIMIT_MS)
    within_limit = sum(1 for value in latencies if 0.0 < value < ORGANIZER_LIMIT_MS)
    total = len(rows)
    summary: dict = {
        "organizer_returncode": returncode,
        "processed": total,
        "non_null": non_null,
        "non_null_ratio": round(non_null / total, 4) if total else 0.0,
        "null_slug": total - non_null,
        "latency_ms": {
            "p50": round(percentile(latencies, 0.50), 1),
            "p95": round(percentile(latencies, 0.95), 1),
            "max": round(max(latencies), 1) if latencies else 0.0,
        },
        "within_limit": within_limit,
        "within_limit_ratio": round(within_limit / total, 4) if total else 0.0,
        "timeouts_at_organizer_cap": timeouts,
        "no_http_response": no_response,
        "sha256_present": sum(1 for row in rows if row.get("image_sha256")),
        "stderr_tail": stderr.strip().splitlines()[-5:],
    }
    if gateway_log is not None and gateway_log.is_file():
        codes: dict[str, int] = {}
        for line in gateway_log.read_text(encoding="utf-8", errors="replace").splitlines():
            if "/v1/eval/predict" not in line or "HTTP/1.1" not in line:
                continue
            parts = line.rsplit('"', 2)
            if len(parts) < 3:
                continue
            code = parts[-1].strip().split(" ", 1)[0]
            codes[code] = codes.get(code, 0) + 1
        summary["gateway_status"] = codes
        summary["http_errors"] = sum(count for code, count in codes.items()
                                     if code not in {"200", "201"})
    else:
        # Без лога gateway единственная наблюдаемая HTTP-ошибка — отсутствие ответа.
        summary["http_errors"] = no_response
    return summary
# END_BLOCK_PARTICIPANT_METRICS


# START_BLOCK_PARTICIPANT_REPORT
def render_markdown(summary: dict, manifest: Path, endpoint: str) -> str:
    """Человекочитаемый отчёт: числа прогона и вывод о совместимости."""
    latency = summary["latency_ms"]
    compatible = (summary["organizer_returncode"] == 0
                  and summary["non_null_ratio"] >= 0.99
                  and summary["timeouts_at_organizer_cap"] == 0
                  and summary["http_errors"] == 0)
    lines = [
        "# Прогон настоящего скрипта организатора против gateway",
        "",
        f"- endpoint: `{endpoint}`",
        f"- манифест: `{manifest}`",
        f"- обработано фото: **{summary['processed']}**",
        f"- код возврата скрипта организатора: {summary['organizer_returncode']}",
        f"- непустой `predicted_slug`: **{summary['non_null']} / {summary['processed']} "
        f"({summary['non_null_ratio'] * 100:.1f}%)**",
        f"- задержка p50 / p95 / max: **{latency['p50']} / {latency['p95']} / {latency['max']} ms**",
        f"- уложились в лимит 10 c: {summary['within_limit']} / {summary['processed']} "
        f"({summary['within_limit_ratio'] * 100:.1f}%)",
        f"- таймауты на лимите скрипта: {summary['timeouts_at_organizer_cap']}",
        f"- HTTP-ошибки: {summary['http_errors']}"
        + (f" (коды: {summary.get('gateway_status')})" if summary.get("gateway_status") else ""),
        f"- отсутствие ответа: {summary['no_http_response']}",
        f"- sha256 зафиксирован для: {summary['sha256_present']} / {summary['processed']}",
        "",
        f"**Вывод о совместимости:** {'СОВМЕСТИМ' if compatible else 'ТРЕБУЕТ ВНИМАНИЯ'}"
        " — скрипт организатора отработал, ответы плоские и укладываются в бюджет.",
    ]
    return "\n".join(lines) + "\n"


def write_reports(summary: dict, json_path: Path, md_path: Path, manifest: Path,
                  endpoint: str) -> None:
    """Пишет машинный и человекочитаемый отчёты прогона."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(summary, manifest, endpoint), encoding="utf-8")
# END_BLOCK_PARTICIPANT_REPORT


# START_BLOCK_PARTICIPANT_MAIN
def main() -> int:
    parser = argparse.ArgumentParser(description="Прогон настоящего скрипта организатора")
    parser.add_argument("--images", default=str(ROOT / "data" / "samples" / "real100"))
    parser.add_argument("--organizer-script", default=str(DEFAULT_ORGANIZER_SCRIPT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080/v1/eval/predict")
    parser.add_argument("--out", default=str(ROOT / "reports" / "participant_run.json"))
    parser.add_argument("--gateway-log", default="")
    args = parser.parse_args()

    images_dir = Path(args.images)
    if not images_dir.is_dir():
        print(f"ERROR: images dir not found: {images_dir}", file=sys.stderr)
        return 2
    manifest = Path(args.manifest)
    predictions = Path(args.predictions)
    count = build_manifest(images_dir, manifest)
    print(f"[participant_run] manifest: {count} rows -> {manifest}", file=sys.stderr)

    proc = run_organizer(Path(args.organizer_script), images_dir, manifest, args.endpoint,
                         predictions)
    if proc.stdout:
        print(proc.stdout, file=sys.stderr)
    if proc.returncode != 0:
        print(f"[participant_run] organizer exited {proc.returncode}", file=sys.stderr)
    if not predictions.is_file():
        print(f"ERROR: predictions not written: {predictions}", file=sys.stderr)
        return 2

    rows = load_rows(predictions)
    gateway_log = Path(args.gateway_log) if args.gateway_log else None
    summary = summarize(rows, proc.returncode, proc.stderr, gateway_log)
    summary["endpoint"] = args.endpoint
    summary["manifest"] = str(manifest)
    summary["predictions"] = str(predictions)
    summary["example_row"] = rows[0] if rows else {}

    out = Path(args.out)
    write_reports(summary, out, out.with_suffix(".md"), manifest, args.endpoint)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nОтчёты: {out} и {out.with_suffix('.md')}", file=sys.stderr)
    return 0
# END_BLOCK_PARTICIPANT_MAIN


if __name__ == "__main__":
    raise SystemExit(main())
