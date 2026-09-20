# START_MODULE_CONTRACT
# PURPOSE: Cross-encoder реранк пар (запрос, кандидат) на bge-reranker-v2-m3 с деградацией без модели
# SCOPE: CE_MODEL_ID, CE_DIR, resolve_ce, load_ce, candidate_text, query_text, rerank
# DEPENDS: numpy, packages.common.catalog, packages.common.config, packages.common.logging, packages.common.text_norm
# LINKS: M-TEXT, M-MATCHER
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# CE_MODEL_ID - идентификатор модели cross-encoder
# CE_DIR - локальный каталог весов (gitignored)
# resolve_ce - доступны ли локальные веса
# load_ce - ленивая загрузка модели и токенизатора (ONNX, иначе transformers на CPU)
# candidate_text - текст кандидата из полей каталога
# query_text - текст запроса из полей OCR/VLM, приведённый нормализацией
# rerank - реранк пар с бюджетом времени; None при недоступности или перерасходе
# END_MODULE_MAP
"""Cross-encoder реранк.

Стандартный этап «Retrieve, Match, Escalate»: кандидаты отбираются дешёвым поиском,
а пары (запрос, кандидат) оцениваются cross-encoder-ом, который видит оба текста вместе
и потому точнее точечного сравнения.

Модель ``BAAI/bge-reranker-v2-m3`` (Apache-2.0, мультиязычная, включая русский). В официальном
репозитории публикуются только веса safetensors, ONNX-файла нет, поэтому основной путь —
transformers на CPU; ветка ONNX остаётся на случай локального ONNX-экспорта. Если модель
недоступна или реранк не уложился в бюджет, возвращается None, и вызывающий оставляет
путь без изменений.
"""
from __future__ import annotations

import time
from pathlib import Path

from packages.common.logging import log_marker
from packages.common.text_norm import fold

CE_MODEL_ID = "BAAI/bge-reranker-v2-m3"
CE_DIR = "data/models/bge_reranker_v2_m3"

__all__ = ["CE_DIR", "CE_MODEL_ID", "candidate_text", "load_ce", "query_text", "rerank", "resolve_ce"]

_cache: dict = {}


def resolve_ce() -> str | None:
    """Путь к локальным весам, если они скачаны."""
    path = Path(CE_DIR)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not (path / "config.json").is_file():
        return None
    return str(path)


def load_ce() -> tuple[object, object, str] | None:
    """Ленивая загрузка модели и токенизатора.

    Сначала пробуется ONNX-файл (если он есть локально), затем transformers на CPU.

    Returns:
        (model, tokenizer, backend) или None, если модель недоступна.
    """
    if "loaded" in _cache:
        return _cache["loaded"]
    path = resolve_ce()
    if path is None:
        log_marker("M-TEXT", "text_ce", "SKIP", reason="weights-missing", path=CE_DIR)
        return None
    started = time.perf_counter()
    try:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path, torch_dtype=torch.float32)
        model = model.to("cpu").eval()
        _cache["loaded"] = (model, tokenizer, "transformers-cpu")
        log_marker("M-TEXT", "text_ce", "LOADED", backend="transformers-cpu",
                   latency_ms=round((time.perf_counter() - started) * 1000.0, 1))
        return _cache["loaded"]
    except Exception as error:  # noqa: BLE001 - отсутствие модели не должно валить путь
        log_marker("M-TEXT", "text_ce", "LOAD_FAILED", error=type(error).__name__,
                   detail=str(error)[:120])
        return None


def candidate_text(record: dict) -> str:
    """Текст кандидата: винодельня, название, сорт, регион, сладость, год."""
    parts = [str(record.get(field) or "") for field in
             ("winery", "name", "grape", "region", "sweetness", "year")]
    grapes = record.get("grapes") or []
    if grapes:
        parts.append(" ".join(str(item) for item in grapes))
    return " ".join(part for part in parts if part).strip()


def query_text(fields: dict) -> str:
    """Текст запроса из полей OCR/VLM, приведённый к единому виду нормализацией."""
    parts = [str(fields.get(key) or "") for key in
             ("winery", "wine_name", "name", "grape", "region", "sweetness", "wine_type", "vintage")]
    return fold(" ".join(part for part in parts if part))


def rerank(query_fields: dict, candidates: list[tuple[str, float]], store,
           *, top_n: int = 20, budget_ms: float | None = None) -> list[tuple[str, float]] | None:
    """Переранжирует кандидатов cross-encoder-ом.

    Args:
        query_fields: поля запроса (из OCR/VLM).
        candidates: список (slug, score) в порядке убывания скора.
        store: каталог (``slug -> запись`` через ``get``).
        top_n: сколько верхних кандидатов оценивать парой.
        budget_ms: предел времени; при перерасходе результат не применяется.

    Returns:
        Новый список (slug, score) или None, если модель недоступна либо бюджет исчерпан.
    """
    loaded = load_ce()
    if loaded is None or not candidates:
        return None
    model, tokenizer, backend = loaded
    query = query_text(query_fields)
    if not query:
        return None
    head = candidates[:top_n]
    tail = candidates[top_n:]
    pairs: list[tuple[str, str]] = []
    for slug, _ in head:
        record = store.get(slug) if hasattr(store, "get") else (store or {}).get(slug)
        pairs.append((query, candidate_text(record or {})))
    started = time.perf_counter()
    try:
        import torch  # noqa: PLC0415

        with torch.no_grad():
            batch = tokenizer([pair[0] for pair in pairs], [pair[1] for pair in pairs],
                              padding=True, truncation=True, max_length=256, return_tensors="pt")
            logits = model(**batch).logits.view(-1)
            scores = torch.sigmoid(logits).tolist()
    except Exception as error:  # noqa: BLE001 - сбой модели не должен валить путь
        log_marker("M-TEXT", "text_ce", "INFER_FAILED", error=type(error).__name__)
        return None
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if budget_ms is not None and elapsed_ms > budget_ms:
        log_marker("M-TEXT", "text_ce", "BUDGET_EXCEEDED", latency_ms=round(elapsed_ms, 1),
                   budget_ms=budget_ms)
        return None
    reranked = [(slug, float(score)) for (slug, _), score in zip(head, scores)]
    reranked.sort(key=lambda item: item[1], reverse=True)
    log_marker("M-TEXT", "text_ce", "DONE", pairs=len(pairs), backend=backend,
               latency_ms=round(elapsed_ms, 1))
    return reranked + tail
