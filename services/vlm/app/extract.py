# START_MODULE_CONTRACT
# PURPOSE: Извлечение полей этикетки через LLM Router из набора изображений (база + тайлы) с ремонтом JSON
# SCOPE: VlmFields, parse_fields, extract_fields
# DEPENDS: json, re, numpy, packages.common.llm.router, packages.common.llm.errors,
#          packages.common.logging, services.vlm.app.prompt
# LINKS: M-VLM
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# VlmFields - результат извлечения: поля этикетки (включая alcohol, volume, extra_text), статус, метаданные
# parse_fields - устойчивый парсинг и ремонт JSON из ответа модели
# extract_fields - вызов router.chat с несколькими изображениями и num_ctx, деградация при отказе
# END_MODULE_MAP
"""Извлечение полей этикетки из полного кадра.

На вход идёт набор изображений (letterbox-база плюс P&S-тайлы): кроп этикетки
не делается, поэтому в кадр попадают и овальные этикетки, и цилиндры, и блики.
Запрос уходит с ``num_ctx=8192`` (дефолт 2048 обрезал контекст и провоцировал
галлюцинации) и коротким промптом: длинные инструкции reasoning-модели
игнорируют. Любой сбой возвращает ``ok=False`` — вызывающий деградирует к Tier-1.
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
_EMPTY_VALUES = {"null", "none", "unknown", "неизвестно", "-", "нет данных"}


@dataclass
class VlmFields:
    """Результат извлечения полей этикетки."""

    winery: str = ""
    wine_name: str = ""
    vintage: str = ""
    grape: str = ""
    sweetness: str = ""
    region: str = ""
    alcohol: str = ""
    volume: str = ""
    extra_text: list[str] = field(default_factory=list)
    confidence: float = 0.0
    ok: bool = False
    reason: str = ""
    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0
    finish_reason: str = ""
    raw: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Поля в формате matcher.

        Дополнительно отдаётся ``wine_type`` (алиас sweetness), чтобы существующий
        matcher продолжал работать без изменений в его модуле.
        """
        payload = {name: getattr(self, name) for name in LABEL_FIELDS}
        payload["wine_type"] = self.sweetness
        return payload

    def filled(self) -> list[str]:
        """Названия заполненных полей (диагностика «прочитал ли модель что-то»)."""
        names = []
        for name in LABEL_FIELDS:
            value = getattr(self, name)
            if name == "confidence":
                continue
            if value:
                names.append(name)
        return names


def _extract_json_object(text: str) -> dict:
    """Достает первый JSON-объект из текста, снимая markdown-обертки."""
    candidate = (text or "").strip()
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
                    parsed = json.loads(chunk)
                except ValueError:
                    return {}
                return parsed if isinstance(parsed, dict) else {}
    return {}


def parse_fields(text: str) -> dict:
    """Ремонтирует и парсит ответ модели в словарь ожидаемых полей."""
    parsed = _extract_json_object(text)
    if not parsed:
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
        if name == "extra_text":
            items = value if isinstance(value, list) else [value]
            texts = [str(item).strip() for item in items
                     if str(item).strip() and str(item).strip().lower() not in _EMPTY_VALUES]
            if texts:
                cleaned[name] = texts
            continue
        text_value = str(value).strip()
        if text_value and text_value.lower() not in _EMPTY_VALUES:
            cleaned[name] = text_value
    return cleaned


def extract_fields(
    images: list[np.ndarray],
    router: LLMRouter | None = None,
    *,
    role: str = "vision_model",
    model: str | None = None,
    max_tokens: int = 512,
    timeout_s: float = 60.0,
    num_ctx: int = 8192,
    reasoning_effort: str = "none",
    extra: dict | None = None,
) -> VlmFields:
    """Извлекает поля этикетки через VLM-роутер из набора изображений.

    Args:
        images: непустой список BGR-изображений (letterbox-база и P&S-тайлы).
        router: роутер; по умолчанию singleton из env.
        role: роль маршрутизации.
        model: явная модель (по умолчанию первая ступень цепочки роли).
        max_tokens: лимит генерации (384-512 для полного ответа).
        timeout_s: жёсткий таймаут вызова.
        num_ctx: размер контекста (8192; дефолт 2048 режет промпт с тайлами).
        reasoning_effort: none отключает внутренние рассуждения.
        extra: дополнительные поля запроса; options.num_ctx добавляется автоматически.

    Returns:
        VlmFields; при любом сбое ok=False с заполненной причиной.
    """
    if not images:
        return VlmFields(ok=False, reason="no images provided")
    active = router or get_router()
    payload = dict(extra or {})
    payload.setdefault("reasoning_effort", reasoning_effort)
    options = dict(payload.get("options") or {})
    options.setdefault("num_ctx", num_ctx)
    payload["options"] = options

    try:
        result = active.chat(
            build_messages(images),
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
                   finish=result.finish_reason, images=len(images))
        return VlmFields(ok=False, reason=reason, provider=result.provider, model=result.model,
                         latency_ms=result.latency_ms, finish_reason=result.finish_reason)

    fields_out = VlmFields(
        winery=fields.get("winery", ""),
        wine_name=fields.get("wine_name", ""),
        vintage=fields.get("vintage", ""),
        grape=fields.get("grape", ""),
        sweetness=fields.get("sweetness", ""),
        region=fields.get("region", ""),
        alcohol=fields.get("alcohol", ""),
        volume=fields.get("volume", ""),
        extra_text=list(fields.get("extra_text") or []),
        confidence=float(fields.get("confidence", 0.0) or 0.0),
        ok=True,
        provider=result.provider,
        model=result.model,
        latency_ms=result.latency_ms,
        finish_reason=result.finish_reason,
        raw=fields,
    )
    log_marker("M-VLM", "extract_fields", "OK", provider=result.provider, model=result.model,
               images=len(images), latency_ms=result.latency_ms,
               filled=len(fields_out.filled()), vintage=fields_out.vintage)
    return fields_out
