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
from packages.common.observer import OBSERVER_INSTRUCTION
from services.vlm.app.prompt import LABEL_FIELDS, build_messages, encode_image_data_url

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


from packages.common.llm.structured import parse_structured


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


def parse_fields(text: str, finish_reason: str = "") -> dict:
    """Ремонтирует и парсит ответ модели в словарь ожидаемых полей."""
    # C-052: источником данных становится структурный разбор по схеме (фенсы, кавычки, регистр,
    # синонимы значений), а прежняя обработка полей ниже сохраняется полностью: приведение
    # confidence к диапазону, очистка пустых значений и сбор extra_text. При невалидном
    # структурном разборе используется прежний путь, поэтому поведение по умолчанию не меняется.
    structured = parse_structured(text, "label_fields", finish_reason=finish_reason)
    parsed = structured["data"] if (structured["valid"] and structured["data"]) else _extract_json_object(text)
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


def _strip_reasoning(text: str) -> str:
    """Убирает блок рассуждений thinking-модели: наблюдению нужен только сам ответ."""
    value = str(text or "")
    for marker in ("<｜end▁of▁thinking｜>", "[/THINK]", "[/thinking]"):
        if marker in value:
            value = value.split(marker, 1)[-1]
    value = re.sub(r"^\s*(thinking|размышлени[а-я]*)\s*:.*?\n\n", "", value, flags=re.IGNORECASE | re.DOTALL)
    return value.strip()



def _resize_for_observer(image, max_side: int):
    """Один уменьшенный кадр для наблюдателя: тайлы не нужны, важнее бюджет времени."""
    import numpy as _np
    from PIL import Image as _Image
    height, width = image.shape[:2]
    scale = float(max_side) / max(height, width, 1)
    if scale >= 1.0:
        return image
    resized = _Image.fromarray(image[:, :, ::-1]).resize(
        (max(1, int(width * scale)), max(1, int(height * scale))), _Image.BILINEAR)
    return _np.asarray(resized)[:, :, ::-1].copy()


def _observe_native_ollama(images: list, model: str, *, max_tokens: int, timeout_s: float,
                           prompt: str | None = None) -> dict | None:
    """Нативный вызов Ollama для thinking-моделей (C-063).

    Совместимый с OpenAI путь не даёт отключить рассуждения: модель тратит бюджет на reasoning и
    возвращает пустой ответ при finish_reason=length. Нативный API принимает think=false и images,
    поэтому используется как запасной путь только для таких случаев.

    C-073: ``prompt`` позволяет задать инструкцию под конкретный читатель (например, чистый OCR-промпт
    для deepseek-ocr); по умолчанию берётся штатная инструкция наблюдателя.
    """
    import base64
    import json
    import urllib.request

    from packages.common.config import settings

    base = str(getattr(settings, "ollama_base_url", "") or "http://127.0.0.1:11434").rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    payload = {"model": model, "prompt": prompt or OBSERVER_INSTRUCTION, "stream": False, "think": False,
               "images": [base64.b64encode(_png_bytes(image)).decode() for image in images[:2]],
               "options": {"num_predict": max_tokens, "temperature": 0.0}}
    try:
        request = urllib.request.Request(f"{base}/api/generate",
                                         data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
            data = json.loads(response.read().decode())
    except Exception as error:  # noqa: BLE001 - запасной путь не должен ломать вызов
        log_marker("M-VLM", "observe_native", "FAILED", error=type(error).__name__)
        return None
    text = str(data.get("response") or "").strip()
    return {"ok": bool(text), "raw_text": text, "model": model, "native": True,
            "finish_reason": "stop"}


def _png_bytes(image) -> bytes:
    """Кодирует кадр в PNG для нативного API."""
    import io

    from PIL import Image as _Image
    buffer = io.BytesIO()
    _Image.fromarray(image[:, :, ::-1]).save(buffer, format="PNG")
    return buffer.getvalue()


def observe_label(
    images: list,
    router=None,
    *,
    role: str = "vision_model",
    model: str | None = None,
    max_tokens: int | None = None,
    timeout_s: float | None = None,
    num_ctx: int = 8192,
    reasoning_effort: str = "none",
    instruction: str | None = None,
) -> dict:
    """Наблюдение этикетки в жёстком бюджете времени.

    Скрипт организатора даёт десять секунд на запрос, поэтому наблюдатель обязан уложиться в свой
    таймаут: вызывается один уменьшенный кадр, рассуждения отключены, а на превышении бюджета
    возвращается неуспех — gateway в этом случае отдаёт результат поиска, а не ошибку.

    C-073: ``instruction`` переопределяет штатную инструкцию наблюдателя (нужно бенчу читателей и
    чисто OCR-моделям); по умолчанию поведение не меняется — берётся ``OBSERVER_INSTRUCTION``.
    """
    from packages.common.config import settings

    if not images:
        return {"ok": False, "reason": "no images provided"}
    selected = model or settings.observer_model
    budget_tokens = int(max_tokens or getattr(settings, "observer_max_tokens", 200))
    max_side = int(getattr(settings, "observer_max_side", 512))
    hard_budget = min(float(timeout_s or 1e9), float(getattr(settings, "observer_timeout_s", 6.5)))
    frame = _resize_for_observer(images[0], max_side)

    # Локальная модель: нативный путь Ollama. Он принимает think=false и один кадр, поэтому
    # укладывается в бюджет; совместимый с OpenAI путь оставлен для облачных провайдеров.
    if "/" not in str(selected):
        native = _observe_native_ollama([frame], str(selected), max_tokens=budget_tokens,
                                        timeout_s=hard_budget, prompt=instruction)
        if native is not None:
            return native
        # Локальная модель не уложилась в бюджет: второй путь не пробуем, иначе суммарное время
        # удвоится и лимит организатора (10 с) будет нарушен.
        return {"ok": False, "reason": "observer budget exceeded", "model": str(selected),
                "budget_s": hard_budget}

    active = router or get_router()
    content = [{"type": "text", "text": instruction or OBSERVER_INSTRUCTION}]
    content.append({"type": "image_url", "image_url": {"url": encode_image_data_url(frame)}})
    payload = {"reasoning_effort": reasoning_effort, "options": {"num_ctx": num_ctx}, "think": False}
    try:
        result = active.chat([{"role": "user", "content": content}], role=role, model=selected,
                             max_tokens=budget_tokens, temperature=0.0, extra=payload,
                             timeout_s=hard_budget)
    except LLMError as error:
        log_marker("M-VLM", "observe_label", "VLM_FAILED", error=type(error).__name__)
        return {"ok": False, "reason": f"{type(error).__name__}: {error}"[:200]}
    text = _strip_reasoning(str(getattr(result, "text", "") or ""))
    return {"ok": bool(text.strip()), "raw_text": text, "model": str(selected),
            "finish_reason": str(getattr(result, "finish_reason", "") or ""),
            "latency_ms": float(getattr(result, "latency_ms", 0.0) or 0.0)}


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
