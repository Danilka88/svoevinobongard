# START_MODULE_CONTRACT
# PURPOSE: Логирование запросов сканера без PII: одна JSON-строка на запрос, best-effort
# SCOPE: DEFAULT_LOG_PATH, build_log_record, append_query_log
# DEPENDS: hashlib, json, datetime, pathlib
# LINKS: M-GATEWAY
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# DEFAULT_LOG_PATH - путь по умолчанию для JSONL-лога запросов
# build_log_record - сборка записи лога без PII (хеш изображения, скоры, path, latency)
# append_query_log - дозапись строки в лог; сбой не поднимается наружу
# END_MODULE_MAP
"""Лог запросов для flywheel.

Пишется одна JSON-строка на запрос. Сырые изображения и персональные данные не
сохраняются: только sha256 картинки, ранжирование, решение и тайминги. Запись
best-effort — сбой логирования не должен ломать ответ скана и не влияет на SLA.
Формат — контракт между gateway и хранилищем flywheel (SQLite его же и читает).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["DEFAULT_LOG_PATH", "build_log_record", "append_query_log"]

DEFAULT_LOG_PATH = "data/flywheel/queries.jsonl"


def build_log_record(
    image_bytes: bytes,
    response,
    *,
    margin_threshold: float,
    normalize_method: str = "",
) -> dict:
    """Собирает запись лога из ответа скана.

    Args:
        image_bytes: сырые байты запроса (хешируются, не сохраняются).
        response: ScanResponse-совместимый объект.
        margin_threshold: порог margin, ниже которого кейс помечается uncertain.
        normalize_method: метод нормализации для диагностики.

    Returns:
        Словарь без PII, готовый к записи в JSONL.
    """
    top_k = [
        {"slug": item.slug, "score": round(float(item.score), 6)}
        for item in (getattr(response, "top5", None) or [])
    ]
    margin = float(getattr(response, "margin", 0.0) or 0.0)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "slug": getattr(response, "slug", None),
        "confidence": round(float(getattr(response, "confidence", 0.0) or 0.0), 6),
        "margin": round(margin, 6),
        "path": getattr(response, "path", ""),
        "latency_ms": round(float(getattr(response, "latency_ms", 0.0) or 0.0), 1),
        "top_k": top_k,
        "not_found": bool(getattr(response, "not_found", False)),
        "uncertain": margin < margin_threshold,
        "normalize_method": normalize_method,
    }


def append_query_log(record: dict, path: str | Path = DEFAULT_LOG_PATH) -> bool:
    """Дозаписывает строку в JSONL-лог. Любая ошибка глушится.

    Returns:
        True при успешной записи, False при сбое.
    """
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except Exception:  # noqa: BLE001 - логирование не должно ломать скан
        return False
