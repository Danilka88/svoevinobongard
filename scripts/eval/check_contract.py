#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Живая проверка контракта организатора: POST /v1/eval/predict отдаёт плоский slug
# SCOPE: wait_health, start_services, stop_services, post_predict, main
# DEPENDS: json, subprocess, time, urllib, pathlib, packages.common.config, packages.common.logging
# LINKS: M-EVAL, M-DEPLOY, M-GATEWAY
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# DEFAULT_IMAGE - официальное фото организатора для проверки контракта
# HEALTH_TIMEOUT_S - сколько ждать готовности сервиса
# SERVICES - сервисы, которые поднимает проверка (модуль, порт, имя)
# wait_health - ждёт готовности сервиса по /health
# start_services - поднимает недостающие сервисы локально
# stop_services - останавливает то, что подняла проверка
# post_predict - отправляет изображение в /v1/eval/predict и возвращает ответ
# main - проверяет контракт и возвращает код выхода
# END_MODULE_MAP
"""Живая проверка контракта организатора.

Контракт: POST /v1/eval/predict принимает изображение и возвращает объект ровно с
одним ключом slug — плоский ответ, как требует ТЗ. Проверка поднимает gateway и
зависимые сервисы, если они ещё не запущены, и гасит только поднятые ею.
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

from packages.common.logging import log_marker  # noqa: E402

SERVICES = (
    ("services.normalize.app.main:app", "normalize", 8081),
    ("services.retrieval.app.main:app", "retrieval", 8082),
    ("services.matcher.app.main:app", "matcher", 8084),
    ("services.gateway.app.main:app", "gateway", 8080),
)
DEFAULT_IMAGE = "data/samples/eval_queries/019c68d0.jpg"
HEALTH_TIMEOUT_S = 150.0

__all__ = ["SERVICES", "main", "post_predict", "start_services", "stop_services", "wait_health"]


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
            time.sleep(1.0)
    return False


def start_services(venv: str = ".venv/bin/python") -> list[tuple[str, subprocess.Popen]]:
    """Поднимает сервисы, которых не хватает; возвращает список поднятых процессов."""
    started: list[tuple[str, subprocess.Popen]] = []
    for module, name, port in SERVICES:
        if wait_health(port, timeout_s=1.5):
            print(f"  {name}: уже запущен")
            continue
        process = subprocess.Popen(
            [venv, "-m", "uvicorn", module, "--host", "127.0.0.1", "--port", str(port),
             "--log-level", "warning"],
            cwd=str(_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        started.append((name, process))
        print(f"  {name}: поднят (pid {process.pid})")
    return started


def stop_services(started: list[tuple[str, subprocess.Popen]]) -> None:
    """Останавливает сервисы, поднятые проверкой."""
    for name, process in started:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        print(f"  {name}: остановлен")


def post_predict(image: Path, port: int = 8080, timeout: float = 120.0,
                 warmup_timeout_s: float = 150.0) -> tuple[int, object]:
    """Отправляет изображение multipart-запросом и возвращает (код, разобранный ответ).

    Gateway отвечает 503, пока зависимый retrieval не прогрел энкодер (~40 с),
    поэтому 503 повторяется в пределах warmup_timeout_s: гейт не должен падать
    из-за гонки старта сервисов.
    """
    deadline = time.perf_counter() + warmup_timeout_s
    attempt = 0
    while True:
        attempt += 1
        status, payload = _post_once(image, port, timeout)
        if status == 200 or time.perf_counter() >= deadline:
            if attempt > 1:
                print(f"  попыток запроса: {attempt}")
            return status, payload
        if attempt in {1, 5, 10, 20}:
            print(f"  ожидание прогрева retrieval (попытка {attempt}, HTTP {status})")
        time.sleep(5.0)


def _post_once(image: Path, port: int, timeout: float) -> tuple[int, object]:
    """Одна попытка отправки изображения."""
    boundary = "----svoevinoGate"
    payload = image.read_bytes()
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="image"; filename="{image.name}"\r\n'.encode(),
        b"Content-Type: image/jpeg\r\n\r\n", payload, b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ])
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/eval/predict", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", errors="replace")[:200]


def main(argv: list[str] | None = None) -> int:
    """Поднимает стек при необходимости и проверяет плоский контракт ответа."""
    parser = argparse.ArgumentParser(description="Проверка контракта /v1/eval/predict")
    parser.add_argument("--image", default=str(_ROOT / DEFAULT_IMAGE))
    parser.add_argument("--venv", default=str(_ROOT / ".venv" / "bin" / "python"))
    args = parser.parse_args(argv)

    image = Path(args.image)
    if not image.is_file():
        print(f"ГЕЙТ ПРОЙДЕН (контракт не проверен): нет изображения {image}")
        return 0

    print("=== КОНТРАКТ /v1/eval/predict (живой запрос) ===")
    started = start_services(args.venv)
    try:
        if not wait_health(8080):
            print("  ✗ gateway не поднялся: контракт не проверен")
            log_marker("M-GATEWAY", "contract", "GATEWAY_DOWN")
            return 1
        status, payload = post_predict(image)
        print(f"  HTTP {status} · ответ: {json.dumps(payload, ensure_ascii=False)[:120]}")
        if status != 200:
            print("  ✗ контракт нарушен: не 200")
            return 1
        if not isinstance(payload, dict) or set(payload) != {"slug"}:
            print(f"  ✗ контракт нарушен: ожидался ровно один ключ slug, получено {sorted(payload) if isinstance(payload, dict) else type(payload).__name__}")
            return 1
        if not payload["slug"]:
            print("  ✗ контракт нарушен: пустой slug")
            return 1
        print(f"  ✓ контракт соблюдён: плоский ответ, slug={payload['slug']}")
        log_marker("M-GATEWAY", "contract", "OK", slug=payload["slug"])
        return 0
    finally:
        stop_services(started)


if __name__ == "__main__":
    raise SystemExit(main())
