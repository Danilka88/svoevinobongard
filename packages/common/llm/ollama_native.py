# START_MODULE_CONTRACT
# PURPOSE: Общий нативный вызов Ollama /api/chat с think=false: обходит reasoning-модели, отдающие пустой content
# SCOPE: ollama_native_chat, _ollama_base
# DEPENDS: json, urllib.request, packages.common.llm.config, packages.common.logging
# LINKS: M-COMMON, M-VLM, M-GATEWAY
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# ollama_native_chat - нативный вызов Ollama: prompt → текст ответа или None (think=false)
# _ollama_base - базовый URL Ollama из providers.yaml (иначе env/локальный дефолт)
# END_MODULE_MAP
"""Нативный вызов локального Ollama.

OpenAI-совместимый путь не даёт отключить рассуждения: reasoning-модели (gemma4, qwen3) тратят весь
бюджет токенов на внутренний монолог и возвращают пустой `content` при finish_reason=length. Нативный
API `/api/chat` принимает `think=false` и `format`, поэтому используется как основной путь для всех
локальных задач проекта (матчер C-074, нормализатор C-077, офлайн-читатели).
"""
from __future__ import annotations

import json

__all__ = ["ollama_native_chat", "_ollama_base"]


def _ollama_base() -> str:
    """Базовый URL Ollama: из providers.yaml (роль ollama), иначе env/локальный дефолт."""
    base = ""
    try:
        from packages.common.llm.config import load_router_config

        entry = (load_router_config().providers or {}).get("ollama")
        base = str(getattr(entry, "base_url", "") or "")
    except Exception:  # noqa: BLE001 - без конфига берём локальный дефолт
        base = ""
    if not base:
        import os

        base = os.environ.get("LLM_OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
    base = base.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base or "http://127.0.0.1:11434"


def ollama_native_chat(prompt: str, model: str, *, max_tokens: int = 256,
                       timeout_s: float = 60.0, fmt: str = "json", num_gpu: int | None = None) -> str | None:
    """Нативный вызов Ollama /api/chat с think=false.

    Args:
        prompt: пользовательский текст запроса.
        model: имя локальной модели (без префикса провайдера).
        max_tokens: лимит генерации (num_predict).
        timeout_s: жёсткий таймаут вызова.
        fmt: ``json`` — требовать JSON-грамматику; пустая строка/``text`` — свободный текст
            (маленькие модели лучше отвечают обычным текстом, чем строгим JSON).
        num_gpu: сколько слоёв отдать GPU; 0 — считать на CPU. Нужно для параллельной работы с
            тяжёлым читателем (26b): маленькая модель на CPU не вытесняет её из VRAM и не платит
            минуты за переключение моделей.

    Returns:
        Текст ответа или None, если вызов не удался (исключение наружу не поднимается).
    """
    import urllib.request

    base = _ollama_base()
    options: dict = {"num_predict": max_tokens, "temperature": 0.0}
    if num_gpu is not None:
        options["num_gpu"] = num_gpu
    payload: dict = {"model": model, "messages": [{"role": "user", "content": prompt}],
                     "stream": False, "think": False, "options": options}
    # Ollama принимает только "json" или отсутствие format; для свободного текста не ставим поле.
    if fmt == "json":
        payload["format"] = "json"
    try:
        request = urllib.request.Request(f"{base}/api/chat", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
            data = json.loads(response.read().decode())
    except Exception as error:  # noqa: BLE001 - запасной/локальный путь не должен ломать вызов
        from packages.common.logging import log_marker

        log_marker("M-COMMON", "ollama_native", "FAILED", model=model, error=type(error).__name__)
        return None
    return str((data.get("message") or {}).get("content") or "").strip()
