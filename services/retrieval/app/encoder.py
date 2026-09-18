# START_MODULE_CONTRACT
# PURPOSE: Плагин image-энкодера siglip2 | dinov2 | clip на MPS с fallback на CPU; L2-нормированные эмбеддинги
# SCOPE: ImageEncoder, load_encoder, resolve_device
# DEPENDS: torch, transformers, numpy, packages.common.config
# LINKS: M-RETRIEVAL
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# ImageEncoder - загрузка модели и батчевое кодирование изображений
# load_encoder - фабрика энкодера по имени из env
# resolve_device - выбор устройства: auto -> mps -> cpu
# END_MODULE_MAP
"""Image-энкодер Tier-1.

Permissive-лицензии: SigLIP 2 (Apache-2.0), DINOv2 (Apache-2.0), OpenCLIP/CLIP (MIT).
Инференс на MPS (Apple Silicon) с автоматическим fallback на CPU.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from packages.common.config import encoder_model_for, settings
from packages.common.logging import log_marker

__all__ = ["ImageEncoder", "load_encoder", "resolve_device"]

MASK_TOKEN = "<mask>"
SIGLIP_IMAGE_TOKEN = "<image>"


def resolve_device(requested: str = "auto") -> str:
    """Определяет устройство инференса.

    Args:
        requested: auto | mps | cuda | cpu.

    Returns:
        Строка устройства для torch.
    """
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _extract_features(result: object) -> torch.Tensor | None:
    """Достает тензор признаков из разных форм ответа HF-модели.

    В transformers 5.x часть моделей (SigLIP 2) возвращает из
    ``get_image_features`` не тензор, а ``BaseModelOutputWithPooling``.
    """
    if isinstance(result, torch.Tensor):
        return result
    for attribute in ("image_embeds", "pooler_output"):
        value = getattr(result, attribute, None)
        if isinstance(value, torch.Tensor):
            return value
    hidden = getattr(result, "last_hidden_state", None)
    if isinstance(hidden, torch.Tensor):
        return _pool_hidden_state(hidden)
    return None


def _pool_hidden_state(hidden_state: torch.Tensor) -> torch.Tensor:
    """Mean-pooling patch-токенов с исключением CLS-токена (для DINOv2)."""
    if hidden_state.dim() == 3 and hidden_state.shape[1] > 1:
        return hidden_state[:, 1:, :].mean(dim=1)
    if hidden_state.dim() == 3:
        return hidden_state.mean(dim=1)
    return hidden_state


class ImageEncoder:
    """Обертка над HF-моделью, возвращающая L2-нормированные image-эмбеддинги."""

    def __init__(self, model_id: str, device: str = "auto", batch_size: int = 16) -> None:
        self.model_id = model_id
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()
        self.model.to(self.device)
        self.dim = self._infer_dim()

    def _infer_dim(self) -> int:
        """Определяет размерность эмбеддинга пробным прогоном."""
        probe = Image.new("RGB", (224, 224), (255, 255, 255))
        vector = self.encode_images([probe])
        return int(vector.shape[1])

    def _forward(self, inputs: dict) -> torch.Tensor:
        """Достает image-эмбеддинги из модели независимо от ее типа.

        Аргументы процессора передаются как есть: naFlex-модели SigLIP 2 требуют
        дополнительные ``spatial_shapes`` и ``pixel_attention_mask``.
        """
        with torch.no_grad():
            get_features = getattr(self.model, "get_image_features", None)
            if callable(get_features):
                try:
                    extracted = _extract_features(get_features(**inputs))
                except TypeError:
                    extracted = None
                if extracted is not None:
                    return extracted
            outputs = self.model(**inputs)
            extracted = _extract_features(outputs)
            if extracted is not None:
                return extracted
            raise RuntimeError(f"cannot extract image features from {self.model_id}")

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """Кодирует изображения и возвращает L2-нормированную матрицу (N, D)."""
        if not images:
            return np.zeros((0, 0), dtype=np.float32)
        batches: list[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            chunk = images[start:start + self.batch_size]
            raw = self.processor(images=chunk, return_tensors="pt")
            inputs = {
                key: (value.to(self.device) if isinstance(value, torch.Tensor) else value)
                for key, value in raw.items()
            }
            features = self._forward(inputs)
            vectors = features.detach().to("cpu", dtype=torch.float32).numpy()
            batches.append(vectors)
        matrix = np.concatenate(batches, axis=0)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (matrix / norms).astype(np.float32)

    def encode_arrays(self, arrays: list[np.ndarray]) -> np.ndarray:
        """Кодирует BGR-массивы OpenCV (конвертация в RGB PIL)."""
        import cv2

        rgb_images = [Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB)) for array in arrays]
        return self.encode_images(rgb_images)


def load_encoder(
    encoder: str | None = None,
    model_id: str | None = None,
    device: str | None = None,
) -> ImageEncoder:
    """Создает энкодер по параметрам окружения.

    Args:
        encoder: siglip2 | dinov2 | clip (по умолчанию из env ENCODER).
        model_id: переопределение HF-идентификатора модели.
        device: переопределение устройства.
    """
    resolved_encoder = encoder or settings.encoder
    resolved_model = model_id or (
        settings.encoder_model if resolved_encoder == settings.encoder else ""
    ) or encoder_model_for(resolved_encoder, "")
    resolved_device = device or settings.encoder_device
    log_marker("M-RETRIEVAL", "load_encoder", "LOAD", model=resolved_model, device=resolve_device(resolved_device))
    return ImageEncoder(resolved_model, device=resolved_device)
