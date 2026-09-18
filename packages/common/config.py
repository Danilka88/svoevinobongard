# START_MODULE_CONTRACT
# PURPOSE: Конфигурация сервисов из переменных окружения с дефолтами и валидацией
# SCOPE: AppSettings, settings, encoder_model_for
# DEPENDS: none (stdlib)
# LINKS: M-COMMON
# ROLE: CONFIG
# MAP_MODE: NONE
# END_MODULE_CONTRACT
"""Настройки сервисов из окружения.

Все параметры читаются один раз при импорте. Подмена провайдера/бэкенда —
только через env (ADR-003).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

__all__: list[str] = []

DEFAULT_ENCODER_MODELS = {
    "siglip2": "google/siglip2-base-patch16-224",
    "dinov2": "facebook/dinov2-small",
    "clip": "openai/clip-vit-base-patch32",
}


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def encoder_model_for(encoder: str, override: str = "") -> str:
    """Возвращает HF-идентификатор модели для выбранного энкодера."""
    if override:
        return override
    return DEFAULT_ENCODER_MODELS.get(encoder, DEFAULT_ENCODER_MODELS["siglip2"])


@dataclass(frozen=True)
class AppSettings:
    """Снимок окружения для всех сервисов."""

    # gateway
    gateway_host: str = field(default_factory=lambda: _env("GATEWAY_HOST", "127.0.0.1"))
    gateway_port: int = field(default_factory=lambda: _env_int("GATEWAY_PORT", 8080))
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 10))

    # service endpoints (HTTP/JSON only — services never import each other)
    normalize_url: str = field(default_factory=lambda: _env("NORMALIZE_URL", "http://127.0.0.1:8081"))
    retrieval_url: str = field(default_factory=lambda: _env("RETRIEVAL_URL", "http://127.0.0.1:8082"))
    service_timeout_s: float = field(default_factory=lambda: _env_float("SERVICE_TIMEOUT_S", 60.0))

    # encoder / index
    encoder: str = field(default_factory=lambda: _env("ENCODER", "siglip2"))
    encoder_model: str = field(default_factory=lambda: _env("ENCODER_MODEL", ""))
    encoder_device: str = field(default_factory=lambda: _env("ENCODER_DEVICE", "auto"))
    vector_backend: str = field(default_factory=lambda: _env("VECTOR_BACKEND", "numpy"))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 30))

    # decision thresholds (used by C-003; kept here so env stays single-sourced)
    score_tau: float = field(default_factory=lambda: _env_float("SCORE_TAU", 0.55))
    margin_delta: float = field(default_factory=lambda: _env_float("MARGIN_DELTA", 0.12))

    # data paths (project-root relative)
    catalog_jsonl: str = field(default_factory=lambda: _env("CATALOG_JSONL", "data/catalog/catalog.jsonl"))
    slug_to_image_json: str = field(
        default_factory=lambda: _env("SLUG_TO_IMAGE_JSON", "data/catalog/slug_to_image.json")
    )
    index_dir: str = field(default_factory=lambda: _env("INDEX_DIR", "data/index"))

    @property
    def resolved_encoder_model(self) -> str:
        return encoder_model_for(self.encoder, self.encoder_model)


settings = AppSettings()
