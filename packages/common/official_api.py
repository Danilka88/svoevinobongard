# START_MODULE_CONTRACT
# PURPOSE: Общий клиент официального API vino-svoe.ru с дисковым кэшем и graceful деградацией
# SCOPE: API_ROOT, fetch_json, cached_json, fetch_similar, absolute_url
# DEPENDS: json, urllib, pathlib, packages.common.config, packages.common.logging
# LINKS: M-COMMON, M-CATALOG, M-GATEWAY
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# API_ROOT - корень публичного API
# SIMILAR_SIZE_MAX - максимум похожих у официального API (24)
# fetch_json - GET с ретраями, таймаутом и вежливой паузой
# cached_json - чтение из дискового кэша или загрузка с записью
# fetch_similar - официальные похожие вина; None при недоступности API
# absolute_url - абсолютный URL для относительных путей API
# END_MODULE_MAP
"""Клиент официального API.

Один и тот же код нужен скрипту синхронизации и gateway, а сервисы не имеют права
импортировать друг друга, поэтому клиент живёт в packages/common.

Кэш на диске обязателен: UI и gateway не должны ходить в сеть на каждый запрос,
а прогон синхронизации должен переживать прерывание.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from packages.common.logging import log_marker

API_ROOT = "https://api.vino-svoe.ru/v1"
_ORIGIN = API_ROOT.rsplit("/v1", 1)[0]
_USER_AGENT = "svoe-vino-scanner/1.0 (+offline enrichment)"
SIMILAR_SIZE_MAX = 24  # проверено: size > 24 API отвергает с 400

__all__ = ["API_ROOT", "SIMILAR_SIZE_MAX", "fetch_json", "cached_json", "fetch_similar", "absolute_url"]


def fetch_json(url: str, timeout: float = 25.0, retries: int = 3, pause: float = 0.15) -> object:
    """GET с ретраями и вежливой паузой между запросами."""
    import time  # noqa: PLC0415

    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": _USER_AGENT, "Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            time.sleep(pause)
            return payload
        except Exception as error:  # noqa: BLE001 - сеть шумит, нужен повтор
            last = error
            time.sleep(pause + 0.4 * (attempt + 1))
    log_marker("M-COMMON", "official_api", "FETCH_FAILED", url=url[:120], error=type(last).__name__)
    raise RuntimeError(f"fetch failed: {url}: {last}")


def cached_json(url: str, cache_path: str | Path, *, refresh: bool = False,
                timeout: float = 25.0, retries: int = 3, pause: float = 0.15) -> object | None:
    """Читает JSON из кэша, при отсутствии — качает и кладёт в кэш.

    Returns:
        Данные или None, если сети нет и кэш пуст (вызывающий решает, что делать).
    """
    target = Path(cache_path)
    if target.is_file() and not refresh:
        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log_marker("M-COMMON", "official_api", "CACHE_CORRUPT", path=str(target))
    try:
        payload = fetch_json(url, timeout=timeout, retries=retries, pause=pause)
    except RuntimeError:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def fetch_similar(slug: str, size: int = 6, cache_dir: str | Path = "data/api_cache/similar",
                  *, refresh: bool = False, timeout: float = 6.0) -> list[dict] | None:
    """Официальные похожие вина для slug.

    Returns:
        Список карточек или None, если API недоступен — тогда вызывающий использует
        запасную логику по метаданным.
    """
    limit = max(1, min(int(size), SIMILAR_SIZE_MAX))
    path = Path(cache_dir) / f"{slug}_{limit}.json"
    payload = cached_json(f"{API_ROOT}/wines/{slug}/similar?size={limit}", path,
                          refresh=refresh, timeout=timeout, retries=1, pause=0.05)
    if payload is None:
        return None
    if isinstance(payload, dict):  # API может вернуть ошибку вместо списка
        log_marker("M-COMMON", "official_api", "SIMILAR_ERROR", slug=slug,
                   error=str(payload)[:120])
        return None
    return [item for item in payload if isinstance(item, dict)]


def absolute_url(path: str | None) -> str:
    """Приводит относительный путь API к абсолютному URL."""
    if not path:
        return ""
    return path if path.startswith("http") else f"{_ORIGIN}{path}"
