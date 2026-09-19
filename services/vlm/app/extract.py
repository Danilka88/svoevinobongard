# START_MODULE_CONTRACT
# PURPOSE: Извлечение полей этикетки через LLM Router с ремонтом JSON, учетом finish_reason и деградацией
# SCOPE: VlmFields, parse_fields, extract_fields
# DEPENDS: json, numpy, packages.common.llm.router, packages.common.llm.errors,
#          packages.common.logging, services.vlm.app.prompt
# LINKS: M-VLM
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# VlmFields - результат извлечения: поля этикетки, статус, причина, провайдер, модель, latency
# parse_fields - устойчивый парсинг и ремонт JSON из ответа модели
# extract_fields - вызов router.chat по роли vision_model с деградацией при отказе или таймауте
# END_MODULE_MAP
"""Извлечение полей этикетки.

Только через ``router.chat(role="vision_model")``: провайдеры, fallback и
брейкеры живут в LLM Router. Любой сбой (таймаут, отказ провайдера, пустой
content при ``finish_reason=length``) возвращает ``VlmFields(ok=False)`` —
вызывающий код деградирует к Tier-1 и не падает.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np

from packages.common.llm.errors import LLMError
from packages.common.llm.router import LLMRouter, get_router
from packages.common.logging import log_marker
from services.vlm.app.prompt import LABEL_FIELDS, build_messages

__all__ = ["VlmFields", "parse_fields", "extract_fields"]

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


@dataclass
class VlmFields:
    """Результат извлечения полей этикетки."""

    winery: str = ""
    wine_name: str = ""
    vintage: str = ""
    wine_type: str = ""
    region: str = ""
    grape: str = ""
    confidence: float = 0.0
    ok: bool = False
    reason: str = ""
    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0
    finish_reason: str = ""
    raw: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Поля без служебных метаданных (то, что уходит в matcher)."""
        return {name: getattr(self, name) for name in LABEL_FIELDS}


def _extract_json_object(text: str) -> dict:
    """Достает первый JSON-объект из текста, снимая markdown-обертки."""
    candidate = text.strip()
    fenced = _FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    start = candidate.find("{")
    if start < 0:
        return {}
    depth = 0
    for index in range(start, len(candidate)):
        char = candidate[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                chunk = candidate[start:index + 1]
                try:
                    return json.loads(chunk)
                except ValueError:
                    return {}
    return {}


def parse_fields(text: str) -> dict:
    """Ремонтирует и парсит ответ модели в словарь полей.

    Args:
        text: сырой content от модели.

    Returns:
        Словарь только с ожидаемыми полями; пустой, если JSON не восстановлен.
    """
    parsed = _extract_json_object(text or "")
    if not isinstance(parsed, dict):
        return {}
    cleaned: dict = {}
    for name in LABEL_FIELDS:
        value = parsed.get(name)
        if value is None:
            continue
        if name == "confidence":
            try:
                cleaned[name] = max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                continue
            continue
        text_value = str(value).strip()
        if text_value and text_value.lower() not in {"null", "none", "unknown", "неизвестно", "-"}:
            cleaned[name] = text_value
    return cleaned


def extract_fields(
    image: np.ndarray,
    router: LLMRouter | None = None,
    *,
    role: str = "vision_model",
    model: str | None = None,
    max_tokens: int = 192,
    timeout_s: float = 4.0,
    extra: dict | None = None,
    candidates: list[dict] | None = None,
) -> VlmFields:
    """Извлекает поля этикетки через VLM-роутер.

    Args:
        image: BGR-изображение (обычно нормализованный кроп).
        router: роутер; по умолчанию singleton из env.
        role: роль маршрутизации.
        model: явная модель (по умолчанию — первая ступень цепочки роли).
        max_tokens: лимит генерации (минимальный ради SLA).
        timeout_s: жёсткий таймаут одного вызова.
        extra: провайдер-специфичные поля; для reasoning-моделей Ollama критично
            ``{"reasoning_effort": "none"}`` — иначе весь бюджет уходит в рассуждения.
        candidates: опциональные кандидаты Top-K для промпта.

    Returns:
        VlmFields; при любом сбое ok=False с заполненной причиной.
    """
    active = router or get_router()
    payload = extra if extra is not None else {"reasoning_effort": "none"}
    try:
        result = active.chat(
            build_messages(image, candidates),
            role=role,
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
            response_format={"type": "json_object"},
            extra=payload,
            timeout_s=timeout_s,
        )
    except LLMError as error:
        log_marker("M-VLM", "extract_fields", "VLM_FAILED", error=type(error).__name__)
        return VlmFields(ok=False, reason=f"{type(error).__name__}: {error}"[:200])

    fields = parse_fields(result.text)
    if not fields:
        reason = ("empty content" if not result.text.strip() else "json not recovered")
        log_marker("M-VLM", "extract_fields", "PARSE_FAILED", reason=reason,
                   finish=result.finish_reason)
        return VlmFields(ok=False, reason=reason, provider=result.provider, model=result.model,
                         latency_ms=result.latency_ms, finish_reason=result.finish_reason)

    log_marker("M-VLM", "extract_fields", "OK", provider=result.provider, model=result.model,
               latency_ms=result.latency_ms, vintage=fields.get("vintage", ""),
               sweetness=fields.get("wine_type", ""))
    return VlmFields(
        winery=fields.get("winery", ""),
        wine_name=fields.get("wine_name", ""),
        vintage=fields.get("vintage", ""),
        wine_type=fields.get("wine_type", ""),
        region=fields.get("region", ""),
        grape=fields.get("grape", ""),
        confidence=float(fields.get("confidence", 0.0) or 0.0),
        ok=True,
        provider=result.provider,
        model=result.model,
        latency_ms=result.latency_ms,
        finish_reason=result.finish_reason,
        raw=fields,
    )
