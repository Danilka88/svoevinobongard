# START_MODULE_CONTRACT
# PURPOSE: OpenAI-совместимые адаптеры провайдеров: chat и embed с типизацией ошибок и признаком retryable
# SCOPE: ChatResult, EmbedResult, BaseAdapter, OpenAICompatAdapter, build_adapter
# DEPENDS: httpx, packages.common.llm.config, packages.common.llm.errors
# LINKS: M-COMMON
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# ChatResult - ответ чата: текст, модель, провайдер, токены, latency, finish_reason
# EmbedResult - ответ эмбеддингов: векторы, модель, провайдер, latency
# BaseAdapter - интерфейс адаптера провайдера
# OpenAICompatAdapter - единый адаптер для OpenAI-совместимых API (Ollama, LM Studio, OpenRouter, YandexGPT, custom)
# build_adapter - фабрика адаптера по конфигурации провайдера
# END_MODULE_MAP
"""Адаптеры провайдеров.

Один OpenAI-совместимый адаптер закрывает Ollama, LM Studio, OpenRouter, YandexGPT
и произвольный custom endpoint: различаются только base_url, ключ и заголовки,
которые берутся из providers.yaml/env. Провайдер-специфика не проникает в код.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import httpx

from packages.common.llm.config import ProviderConfig
from packages.common.llm.errors import ProviderError

__all__ = [
    "ChatResult",
    "EmbedResult",
    "BaseAdapter",
    "OpenAICompatAdapter",
    "build_adapter",
]

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class ChatResult:
    """Ответ чат-комплишена.

    ``finish_reason`` важен для reasoning-моделей: при ``length`` бюджет токенов
    закончился (у qwen3.5 он уходит в поле reasoning) и текст может быть пустым.
    """

    text: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = ""


@dataclass(frozen=True)
class EmbedResult:
    """Ответ эмбеддингов."""

    vectors: list[list[float]] = field(default_factory=list)
    model: str = ""
    provider: str = ""
    total_tokens: int = 0
    latency_ms: float = 0.0


class BaseAdapter:
    """Интерфейс адаптера провайдера."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        """Имя провайдера."""
        return self.config.name

    def chat(self, model: str, messages: list[dict], *, max_tokens: int = 512,
             temperature: float = 0.0, response_format: dict | None = None) -> ChatResult:
        """Выполняет чат-комплишен."""
        raise NotImplementedError

    def embed(self, model: str, texts: list[str]) -> EmbedResult:
        """Считает эмбеддинги."""
        raise NotImplementedError

    def health(self) -> bool:
        """Проверяет доступность провайдера без генерации."""
        try:
            with httpx.Client(timeout=min(self.config.timeout_s, 5.0)) as client:
                response = client.get(f"{self.config.base_url.rstrip('/')}/models",
                                      headers=self._headers())
            return response.status_code < 500
        except httpx.HTTPError:
            return False

    def _headers(self) -> dict[str, str]:
        """Заголовки запроса: авторизация плюс провайдер-специфичные."""
        headers = {"Content-Type": "application/json", **self.config.headers}
        if self.config.api_key:
            headers.setdefault("Authorization", f"Bearer {self.config.api_key}")
        return headers


class OpenAICompatAdapter(BaseAdapter):
    """Единый адаптер для OpenAI-совместимых провайдеров."""

    def _post(self, path: str, payload: dict) -> tuple[dict, float]:
        """Выполняет POST и возвращает (json, latency_ms).

        Raises:
            ProviderError: при сетевом сбое или не-2xx ответе.
        """
        url = f"{self.config.base_url.rstrip('/')}{path}"
        started = time.perf_counter()
        try:
            with httpx.Client(timeout=self.config.timeout_s) as client:
                response = client.post(url, json=payload, headers=self._headers())
        except (httpx.TimeoutException, httpx.TransportError) as error:
            raise ProviderError(self.name, str(payload.get("model", "")),
                                f"{type(error).__name__}: {error}", retryable=True) from error
        latency_ms = (time.perf_counter() - started) * 1000.0
        if response.status_code >= 400:
            raise ProviderError(
                self.name,
                str(payload.get("model", "")),
                f"HTTP {response.status_code}: {response.text[:200]}",
                retryable=response.status_code in _RETRYABLE_STATUS,
            )
        try:
            return response.json(), latency_ms
        except ValueError as error:
            raise ProviderError(self.name, str(payload.get("model", "")),
                                f"invalid json: {error}", retryable=True) from error

    def chat(self, model: str, messages: list[dict], *, max_tokens: int = 512,
             temperature: float = 0.0, response_format: dict | None = None) -> ChatResult:
        """Выполняет чат-комплишен через /chat/completions."""
        payload: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        data, latency_ms = self._post("/chat/completions", payload)
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError(self.name, model, "empty choices", retryable=True)
        message = choices[0].get("message") or {}
        usage = data.get("usage") or {}
        return ChatResult(
            text=str(message.get("content", "")),
            model=str(data.get("model", model)),
            provider=self.name,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            total_tokens=int(usage.get("total_tokens", 0) or 0),
            latency_ms=round(latency_ms, 1),
            finish_reason=str(choices[0].get("finish_reason") or ""),
        )

    def embed(self, model: str, texts: list[str]) -> EmbedResult:
        """Считает эмбеддинги через /embeddings."""
        payload = {"model": model, "input": texts}
        data, latency_ms = self._post("/embeddings", payload)
        items = sorted(data.get("data") or [], key=lambda item: item.get("index", 0))
        vectors = [list(item.get("embedding") or []) for item in items]
        if not vectors:
            raise ProviderError(self.name, model, "empty embeddings", retryable=True)
        usage = data.get("usage") or {}
        return EmbedResult(
            vectors=vectors,
            model=str(data.get("model", model)),
            provider=self.name,
            total_tokens=int(usage.get("total_tokens", 0) or 0),
            latency_ms=round(latency_ms, 1),
        )


def build_adapter(config: ProviderConfig) -> BaseAdapter:
    """Создает адаптер по конфигурации провайдера.

    Raises:
        ValueError: для неизвестного типа провайдера.
    """
    if config.type in {"openai-compat", "openai", "custom"}:
        return OpenAICompatAdapter(config)
    raise ValueError(f"unsupported provider type: {config.type}")
