# START_MODULE_CONTRACT
# PURPOSE: HTTP-клиенты внутренних сервисов (normalize, retrieval) — единственный канал связи между сервисами
# SCOPE: ServiceClients, post_image
# DEPENDS: httpx, packages.common.config, packages.common.logging
# LINKS: M-GATEWAY
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# ServiceClients - асинхронные клиенты normalize и retrieval
# post_image - multipart-POST изображения с обработкой ошибки связи
# END_MODULE_MAP
"""Клиенты внутренних сервисов.

Сервисы не импортируют код друг друга: gateway общается с normalize и retrieval
только по HTTP/JSON (AR-2). Ошибка связи возвращает None, чтобы вызывающий код
мог деградировать, а не падать.
"""
from __future__ import annotations

import httpx

from packages.common.config import settings
from packages.common.logging import log_marker

__all__ = ["ServiceClients", "post_image"]


async def post_image(
    client: httpx.AsyncClient,
    url: str,
    data: bytes,
    filename: str,
    timeout: float,
) -> dict | None:
    """Отправляет изображение на сервис.

    Args:
        client: httpx-клиент.
        url: полный URL эндпоинта.
        data: байты изображения.
        filename: имя файла в multipart.
        timeout: таймаут запроса в секундах.

    Returns:
        Разобранный JSON или None при ошибке связи/статуса.
    """
    try:
        response = await client.post(
            url,
            files={"image": (filename, data, "application/octet-stream")},
            timeout=timeout,
        )
    except httpx.HTTPError as error:
        log_marker("M-GATEWAY", "post_image", "CONN_ERROR", url=url, error=type(error).__name__)
        return None
    if response.status_code != 200:
        log_marker("M-GATEWAY", "post_image", "BAD_STATUS", url=url, status=response.status_code)
        return None
    try:
        return response.json()
    except ValueError:
        log_marker("M-GATEWAY", "post_image", "BAD_JSON", url=url)
        return None


class ServiceClients:
    """Клиенты внутренних сервисов пайплайна."""

    def __init__(
        self,
        normalize_url: str | None = None,
        retrieval_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.normalize_url = (normalize_url or settings.normalize_url).rstrip("/")
        self.retrieval_url = (retrieval_url or settings.retrieval_url).rstrip("/")
        self.timeout = timeout if timeout is not None else settings.service_timeout_s
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Ленивый общий httpx-клиент."""
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self._client

    async def normalize(self, data: bytes, filename: str = "image.jpg") -> dict | None:
        """Нормализует изображение (POST /normalize)."""
        return await post_image(
            self.client, f"{self.normalize_url}/normalize", data, filename, self.timeout
        )

    async def search(self, data: bytes, filename: str = "image.jpg") -> dict | None:
        """Ищет Top-K по каталогу (POST /search)."""
        return await post_image(
            self.client, f"{self.retrieval_url}/search", data, filename, self.timeout
        )

    async def aclose(self) -> None:
        """Закрывает HTTP-клиент."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def warmup(self) -> dict | None:
        """Прогревает retrieval (POST /warmup), чтобы первый запрос не платил за загрузку модели."""
        try:
            response = await self.client.post(f"{self.retrieval_url}/warmup", timeout=self.timeout)
        except httpx.HTTPError as error:
            log_marker("M-GATEWAY", "warmup", "CONN_ERROR", error=type(error).__name__)
            return None
        if response.status_code != 200:
            log_marker("M-GATEWAY", "warmup", "BAD_STATUS", status=response.status_code)
            return None
        try:
            return response.json()
        except ValueError:
            return None
