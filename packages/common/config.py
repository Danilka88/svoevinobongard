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
    vlm_url: str = field(default_factory=lambda: _env("VLM_URL", "http://127.0.0.1:8083"))
    matcher_url: str = field(default_factory=lambda: _env("MATCHER_URL", "http://127.0.0.1:8084"))
    service_timeout_s: float = field(default_factory=lambda: _env_float("SERVICE_TIMEOUT_S", 60.0))

    # VLM fallback (C-004): online path uses a FAST model with reasoning disabled.
    # Default OFF: замер C-004 показал, что локальный VLM не даёт прироста точности
    # (поля пустые у gemma4:e4b), но поднимает p95 до ~5 с. Включается осознанно.
    vlm_fallback_enabled: bool = field(
        default_factory=lambda: _env("VLM_FALLBACK_ENABLED", "0") not in {"0", "false", "no"})
    vlm_online_model: str = field(default_factory=lambda: _env("VLM_ONLINE_MODEL", "gemma4:e4b"))
    vlm_max_tokens: int = field(default_factory=lambda: _env_int("VLM_MAX_TOKENS", 192))
    vlm_timeout_s: float = field(default_factory=lambda: _env_float("VLM_TIMEOUT_S", 4.0))
    vlm_reasoning_effort: str = field(
        default_factory=lambda: _env("VLM_REASONING_EFFORT", "none"))

    # OCR text signal (C-010): дешёвая замена VLM-fallback в онлайн-пути
    ocr_rec_model: str = field(default_factory=lambda: _env("OCR_REC_MODEL", ""))
    # C-020: выбор движка OCR со стороны запроса и параметры PaddleOCR-VL
    ocr_engine: str = field(default_factory=lambda: _env("OCR_ENGINE", ""))
    ocr_vl_model: str = field(default_factory=lambda: _env("OCR_VL_MODEL", "data/models/paddleocr_vl"))
    ocr_vl_max_tokens: int = field(default_factory=lambda: _env_int("OCR_VL_MAX_TOKENS", 1024))
    ocr_det_side: int = field(default_factory=lambda: _env_int("OCR_DET_SIDE", 1280))
    ocr_rec_width: int = field(default_factory=lambda: _env_int("OCR_REC_WIDTH", 640))
    ocr_rec_keys: str = field(default_factory=lambda: _env("OCR_REC_KEYS", ""))
    ocr_url: str = field(default_factory=lambda: _env("OCR_URL", "http://127.0.0.1:8085"))
    ocr_enabled: bool = field(
        default_factory=lambda: _env("OCR_ENABLED", "1") not in {"0", "false", "no"})
    ocr_max_side: int = field(default_factory=lambda: _env_int("OCR_MAX_SIDE", 768))
    ocr_timeout_s: float = field(default_factory=lambda: _env_float("OCR_TIMEOUT_S", 1.5))

    # full-frame preprocessing (C-012): полный кадр — база, кроп только за флагом
    normalize_crop: bool = field(
        default_factory=lambda: _env("NORMALIZE_CROP", "0") not in {"0", "false", "no"})
    normalize_enhance: bool = field(
        default_factory=lambda: _env("NORMALIZE_ENHANCE", "0") not in {"0", "false", "no"})
    pas_tiles: int = field(default_factory=lambda: _env_int("PAS_TILES", 4))
    pas_tiles_uncertain: int = field(default_factory=lambda: _env_int("PAS_TILES_UNCERTAIN", 6))
    vlm_num_ctx: int = field(default_factory=lambda: _env_int("VLM_NUM_CTX", 8192))

    # flywheel / active-learning (C-006): логи запросов и human-in-the-loop
    flywheel_enabled: bool = field(
        default_factory=lambda: _env("FLYWHEEL_ENABLED", "1") not in {"0", "false", "no"})
    flywheel_log_path: str = field(
        default_factory=lambda: _env("FLYWHEEL_LOG_PATH", "data/flywheel/queries.jsonl"))
    flywheel_db_path: str = field(
        default_factory=lambda: _env("FLYWHEEL_DB_PATH", "data/flywheel/flywheel.db"))
    flywheel_confirmed_path: str = field(
        default_factory=lambda: _env("FLYWHEEL_CONFIRMED_PATH", "data/flywheel/confirmed.jsonl"))

    # matcher fusion (C-004)
    matcher_text_weight: float = field(default_factory=lambda: _env_float("MATCHER_TEXT_WEIGHT", 0.6))
    matcher_vintage_bonus: float = field(default_factory=lambda: _env_float("MATCHER_VINTAGE_BONUS", 0.3))
    matcher_sweetness_bonus: float = field(default_factory=lambda: _env_float("MATCHER_SWEETNESS_BONUS", 0.25))

    # text fusion + enriched catalog (C-013); флаг по умолчанию выключен до валидации
    text_fusion_enabled: bool = field(
        default_factory=lambda: _env("TEXT_FUSION_ENABLED", "0").lower() not in {"0", "false", "no", ""})
    text_fusion_weight: float = field(default_factory=lambda: _env_float("TEXT_FUSION_WEIGHT", 0.6))
    text_fusion_year_bonus: float = field(
        default_factory=lambda: _env_float("TEXT_FUSION_YEAR_BONUS", 0.35))
    text_fusion_sweetness_bonus: float = field(
        default_factory=lambda: _env_float("TEXT_FUSION_SWEETNESS_BONUS", 0.25))
    catalog_enriched_jsonl: str = field(
        default_factory=lambda: _env("CATALOG_ENRICHED_JSONL", "data/catalog/catalog_enriched.jsonl"))
    text_fusion_mode: str = field(default_factory=lambda: _env("TEXT_FUSION_MODE", "conditional"))
    text_fusion_trigger_max_gap: float = field(
        default_factory=lambda: _env_float("TEXT_FUSION_TRIGGER_MAX_GAP", 0.05))
    # C-017: гастросочетания и цифровые поля из официального API как опциональный сигнал
    text_fusion_use_dishes: bool = field(
        default_factory=lambda: _env("TEXT_FUSION_USE_DISHES", "0").lower() not in {"0", "false", "no", ""})
    text_fusion_use_numbers: bool = field(
        default_factory=lambda: _env("TEXT_FUSION_USE_NUMBERS", "0").lower() not in {"0", "false", "no", ""})
    # C-022: двухуровневый поиск по прототипам виноделен (по умолчанию выключен до подтверждённого выигрыша)
    winery_proto_enabled: bool = field(
        default_factory=lambda: _env("WINERY_PROTO_ENABLED", "0").lower() not in {"0", "false", "no", ""})
    winery_proto_weight: float = field(default_factory=lambda: _env_float("WINERY_PROTO_WEIGHT", 0.15))
    query_tta: int = field(default_factory=lambda: _env_int("QUERY_TTA", 1))
    catalog_v3_jsonl: str = field(
        default_factory=lambda: _env("CATALOG_V3_JSONL", "data/catalog/catalog_enriched_v3.jsonl"))

    # encoder / index
    encoder: str = field(default_factory=lambda: _env("ENCODER", "siglip2"))
    encoder_model: str = field(default_factory=lambda: _env("ENCODER_MODEL", ""))
    encoder_device: str = field(default_factory=lambda: _env("ENCODER_DEVICE", "auto"))
    vector_backend: str = field(default_factory=lambda: _env("VECTOR_BACKEND", "numpy"))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 30))

    # decision thresholds: перекалиброваны в C-021 на текущем пайплайне (200 сэмплов, seed 42).
# C-022 пересчитал их на мультивью-галерее (4 вида): score_tau 0.7615 (+0.97 п.п. F1),
# margin_delta 0.0058 (+13.45 п.п. F1), geo_weight 0.5 (лучшие кандидаты делают сильную
# геоверификацию вредной: acc@1 0.720 -&gt; 0.730). confidence_temperature оставлена 0.2.
    score_tau: float = field(default_factory=lambda: _env_float("SCORE_TAU", 0.7615))
    margin_delta: float = field(default_factory=lambda: _env_float("MARGIN_DELTA", 0.0058))
    confidence_temperature: float = field(
        default_factory=lambda: _env_float("CONFIDENCE_TEMPERATURE", 0.2))

    # geo-verification (C-003)
    geo_weight: float = field(default_factory=lambda: _env_float("GEO_WEIGHT", 0.5))
    geo_topk: int = field(default_factory=lambda: _env_int("GEO_TOPK", 10))
    geo_min_matches: int = field(default_factory=lambda: _env_int("GEO_MIN_MATCHES", 8))
    warmup_on_start: bool = field(
        default_factory=lambda: _env("WARMUP_ON_START", "1") not in {"0", "false", "no"})

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
