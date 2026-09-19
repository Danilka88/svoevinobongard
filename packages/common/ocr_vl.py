# START_MODULE_CONTRACT
# PURPOSE: PaddleOCR-VL (Apache-2.0) через transformers: чтение полного кадра без кропа этикетки
# SCOPE: resolve_vl, load_vl, run_vl, VL_DEVICES
# DEPENDS: numpy, PIL, torch, transformers, packages.common.config, packages.common.logging
# LINKS: M-OCR
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# VL_DEVICES - порядок устройств для инференса (MPS, затем CPU)
# resolve_vl - доступна ли модель PaddleOCR-VL по конфигу
# load_vl - ленивая загрузка модели и процессора
# run_vl - распознавание полного кадра: строки текста и latency
# END_MODULE_MAP
"""PaddleOCR-VL 0.9B через transformers.

Лицензия Apache-2.0 (подтверждена по карточке HF). Используется transformers-путь,
чтобы не тянуть paddlepaddle на arm64: установленный transformers 5.17 знает
архитектуру PaddleOCRVLForConditionalGeneration.

Вход — полный кадр (принцип проекта: этикетку не кропим). Выход — текстовые строки,
которые дальше разбираются общим parse_fields, поэтому годы и сладость читаются тем же
кодом, что и для остальных движков.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from packages.common.config import settings
from packages.common.logging import log_marker

VL_DEVICES = ("mps", "cpu")
DEFAULT_PROMPT = "OCR:"

__all__ = ["VL_DEVICES", "resolve_vl", "load_vl", "run_vl"]

_cache: dict[str, object] = {}


def resolve_vl() -> str | None:
    """Путь к локальным весам PaddleOCR-VL, если они скачаны.

    Returns:
        Путь к каталогу модели или None, если весов нет.
    """
    path = Path(str(getattr(settings, "ocr_vl_model", "") or "data/models/paddleocr_vl"))
    if not path.is_absolute():
        path = Path.cwd() / path
    if not (path / "config.json").is_file():
        log_marker("M-OCR", "vl", "SKIP", reason="weights-missing", path=str(path))
        return None
    return str(path)


def load_vl() -> tuple[object, object, str] | None:
    """Ленивая загрузка модели и процессора на первое доступное устройство.

    Returns:
        (model, processor, device) или None, если веса недоступны или загрузка упала.
    """
    if "model" in _cache:
        return _cache["model"], _cache["processor"], _cache["device"]  # type: ignore[return-value]
    path = resolve_vl()
    if path is None:
        return None
    try:
        import torch  # noqa: PLC0415
        # Нативный класс transformers обязателен: remote-code модели написан под другой
        # набор ROPE_INIT_FUNCTIONS и на transformers 5.17 падает с KeyError 'default'.
        from transformers import (  # noqa: PLC0415
            AutoProcessor,
            PaddleOCRVLForConditionalGeneration,
        )

        processor = AutoProcessor.from_pretrained(path, trust_remote_code=False)
        model = None
        device = "cpu"
        for candidate in VL_DEVICES:
            if candidate == "mps" and not torch.backends.mps.is_available():
                continue
            try:
                dtype = torch.float16 if candidate == "mps" else torch.float32
                model = PaddleOCRVLForConditionalGeneration.from_pretrained(
                    path, dtype=dtype)
                model = model.to(candidate).eval()
                device = candidate
                break
            except Exception as error:  # noqa: BLE001 - пробуем следующее устройство
                        log_marker("M-OCR", "vl", "DEVICE_FAILED", device=candidate,
                           error=type(error).__name__, detail=str(error)[:100])
        if model is None:
            return None
        _cache["model"], _cache["processor"], _cache["device"] = model, processor, device
        log_marker("M-OCR", "vl", "LOADED", device=device, path=path)
        return model, processor, device
    except Exception as error:  # noqa: BLE001 - отсутствие весов не должно валить пайплайн
        log_marker("M-OCR", "vl", "LOAD_FAILED", error=type(error).__name__,
                   detail=str(error)[:120])
        return None


def run_vl(image: np.ndarray, prompt: str = DEFAULT_PROMPT,
           max_new_tokens: int | None = None) -> tuple[list[str], float]:
    """Распознаёт полный кадр моделью PaddleOCR-VL.

    Args:
        image: BGR-кадр целиком (кроп этикетки не делается).
        prompt: задача для модели, по умолчанию распознавание текста.
        max_new_tokens: ограничение длины ответа.

    Returns:
        (строки текста, latency_ms). Пустой список, если модель недоступна или упала.
    """
    loaded = load_vl()
    if loaded is None:
        return [], 0.0
    model, processor, device = loaded
    limit = int(max_new_tokens or getattr(settings, "ocr_vl_max_tokens", 1024))
    started = time.perf_counter()
    try:
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        rgb = Image.fromarray(image[:, :, ::-1]).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": rgb},
            {"type": "text", "text": prompt},
        ]}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt")
        inputs = {key: value.to(device) if hasattr(value, "to") else value
                  for key, value in inputs.items()}
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=limit, do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        text = processor.decode(output[0][prompt_len:], skip_special_tokens=True)
    except Exception as error:  # noqa: BLE001 - сбой модели не должен валить пайплайн
        log_marker("M-OCR", "vl", "INFER_FAILED", error=type(error).__name__)
        return [], (time.perf_counter() - started) * 1000.0
    latency = (time.perf_counter() - started) * 1000.0
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    log_marker("M-OCR", "vl", "DONE", lines=len(lines), latency_ms=round(latency, 1),
               device=device)
    return lines, latency
