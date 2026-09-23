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
# image_proxy_url - рабочий адрес картинки портала через прокси изображений
# proxy_images_in - рекурсивная замена адресов картинок в данных API
# resize_proxy_url - смена размера у уже проксированного адреса
# is_image_path - отличает изображение от данных по расширению
# IMAGE_EXTENSIONS - расширения, которые считаются изображениями
# MEDIA_SIZE_CARD - размер картинки карточки по умолчанию
# MEDIA_SIZE_HERO - размер крупной картинки по умолчанию
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

import os
import re
from urllib.parse import urlsplit

API_ROOT = "https://api.vino-svoe.ru/v1"
_ORIGIN = API_ROOT.rsplit("/v1", 1)[0]
_USER_AGENT = "svoe-vino-scanner/1.0 (+offline enrichment)"
SIMILAR_SIZE_MAX = 24  # проверено: size > 24 API отвергает с 400

__all__ = ["API_ROOT", "SIMILAR_SIZE_MAX", "fetch_json", "cached_json", "fetch_similar", "absolute_url",
           "image_proxy_url", "proxy_images_in", "resize_proxy_url", "is_image_path",
           "MEDIA_SIZE_CARD", "MEDIA_SIZE_HERO", "IMAGE_EXTENSIONS"]


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
                timeout: float = 25.0, retries: int = 3, pause: float = 0.15,
                max_age_s: float | None = None) -> object | None:
    """Читает JSON из кэша, при отсутствии — качает и кладёт в кэш.

    Args:
        refresh: игнорировать кэш и скачать заново.
        max_age_s: максимальный возраст кэша в секундах; None — срок не проверяется
            (поведение по умолчанию, как раньше). Устаревший кэш перекачивается.

    Returns:
        Данные или None, если сети нет и кэш пуст (вызывающий решает, что делать).
    """
    target = Path(cache_path)
    stale = False
    if target.is_file() and max_age_s is not None:
        import time  # noqa: PLC0415

        stale = (time.time() - target.stat().st_mtime) > max_age_s
    if target.is_file() and not refresh and not stale:
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


# C-050: сырые адреса /uploads/... отдают 404, рабочий путь — прокси изображений портала.
PROXY_TEMPLATE = "https://api.vino-svoe.ru/v1/img/str-api/{size}/{size}/resize/uploads/{name}"
PORTAL_HOSTS = ("api.vino-svoe.ru", "vino-svoe.ru")
IMAGE_EXTENSIONS = (".webp", ".jpg", ".jpeg", ".png", ".avif")
MEDIA_SIZE_CARD = int(os.environ.get("MEDIA_SIZE_CARD", "600"))
MEDIA_SIZE_HERO = int(os.environ.get("MEDIA_SIZE_HERO", "1200"))


def image_proxy_url(path_or_url: str | None, size: int = MEDIA_SIZE_CARD) -> str:
    """Возвращает рабочий адрес картинки портала через прокси изображений.

    Принимает как относительный путь портала (``/uploads/x.webp``), так и полный адрес того же
    домена. Локальные адреса (``/media/...``) и адреса других доменов возвращаются без изменений:
    переписывать чужие ссылки нельзя.
    """
    raw = str(path_or_url or "").strip()
    if not raw:
        return ""
    name = ""
    if raw.startswith("/uploads/"):
        name = raw[len("/uploads/"):]
    elif raw.startswith("http"):
        parsed = urlsplit(raw)
        if parsed.netloc not in PORTAL_HOSTS or not parsed.path.startswith("/uploads/"):
            return raw
        name = parsed.path[len("/uploads/"):]
    else:
        return raw
    if not name or not is_image_path(name):
        # Данные (например geojson) прокси не отдаёт: на них приходит 404, поэтому путь не трогаем.
        return raw
    return PROXY_TEMPLATE.format(size=int(size), name=name)


def is_image_path(path: str | None) -> bool:
    """Отличает изображение от данных: расширение важнее расположения."""
    value = str(path or "").split("?")[0].lower()
    return value.endswith(IMAGE_EXTENSIONS)


def resize_proxy_url(url: str | None, size: int = MEDIA_SIZE_HERO) -> str:
    """Меняет размер уже проксированного адреса (для крупных изображений).

    Сторонние и не-проксированные адреса возвращаются без изменений.
    """
    raw = str(url or "")
    if "/v1/img/str-api/" not in raw:
        return raw
    return re.sub(r"/v1/img/str-api/\d+/\d+/", f"/v1/img/str-api/{int(size)}/{int(size)}/", raw)


def proxy_images_in(node: object, size: int = MEDIA_SIZE_CARD) -> object:
    """Рекурсивно заменяет адреса картинок портала на прокси-адреса в структуре данных.

    Применяется к данным официального API: адреса вида ``/uploads/...`` встречаются на разной
    глубине (бутылка, сорта, блюда, регион, туры, статьи), поэтому обход идёт по всей структуре,
    а чужие и локальные адреса остаются как есть.
    """
    if isinstance(node, dict):
        return {key: proxy_images_in(value, size) for key, value in node.items()}
    if isinstance(node, list):
        return [proxy_images_in(item, size) for item in node]
    if isinstance(node, str) and "/uploads/" in node:
        return image_proxy_url(node, size)
    return node


def absolute_url(path: str | None) -> str:
    """Приводит относительный путь API к абсолютному URL."""
    if not path:
        return ""
    return path if path.startswith("http") else f"{_ORIGIN}{path}"
