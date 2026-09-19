# START_MODULE_CONTRACT
# PURPOSE: Единый LLM Router: chat и embed с обходом цепочки провайдеров, брейкерами и локальным режимом
# SCOPE: LLMRouter, get_router, reset_router
# DEPENDS: packages.common.llm.adapters, packages.common.llm.circuit, packages.common.llm.config,
#          packages.common.llm.errors, packages.common.logging
# LINKS: M-COMMON
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# LLMRouter - роутер чата и эмбеддингов: разрешение роли, цепочка провайдеров, брейкеры, cost-контроль
# get_router - ленивый singleton роутера
# reset_router - сброс singleton (для тестов и смены конфигурации)
# END_MODULE_MAP
"""Единый LLM Router.

Подмена провайдера или модели выполняется только через providers.yaml и env:
в коде нет захардкоженных имён. Запрос идет по цепочке роли; повторяемые отказы
копятся в per-provider circuit breaker, а расход токенов контролируется глобальным
cost-velocity breaker. При ``LLM_LOCAL_ONLY=1`` облачные провайдеры исключаются из
цепочки, поэтому отсутствие сети не ломает вызов.
"""
from __future__ import annotations

import time
from typing import Callable

from packages.common.llm.adapters import BaseAdapter, ChatResult, EmbedResult, build_adapter
from packages.common.llm.circuit import CircuitBreaker, CostVelocityBreaker
from packages.common.llm.config import RoutingEntry, RouterConfig, load_router_config
from packages.common.llm.errors import (
    AllProvidersExhausted,
    CostVelocityExceeded,
    ProviderError,
    RoutingError,
)
from packages.common.logging import log_marker

__all__ = ["LLMRouter", "get_router", "reset_router"]

DEFAULT_CHAT_ROLE = "text_model"
DEFAULT_EMBED_ROLE = "embedding_model"


class LLMRouter:
    """Роутер LLM с цепочкой провайдеров и защитой от отказов."""

    def __init__(
        self,
        config: RouterConfig | None = None,
        adapters: dict[str, BaseAdapter] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.config = config or load_router_config()
        self._adapters: dict[str, BaseAdapter] = dict(adapters or {})
        self._breakers: dict[str, CircuitBreaker] = {}
        self._clock = clock or time.monotonic
        self.cost_breaker = CostVelocityBreaker(
            window_s=self.config.cost_window_s,
            multiplier=self.config.cost_multiplier,
            open_s=self.config.cost_open_s,
            plan_tokens_per_min=self.config.cost_plan_tokens_per_min,
            clock=self._clock,
        )

    def adapter(self, provider: str) -> BaseAdapter | None:
        """Возвращает адаптер провайдера, если он пригоден."""
        if provider in self._adapters:
            return self._adapters[provider]
        config = self.config.providers.get(provider)
        if config is None or not config.usable:
            return None
        return build_adapter(config)

    def breaker(self, provider: str) -> CircuitBreaker:
        """Возвращает (создавая при необходимости) брейкер провайдера."""
        if provider not in self._breakers:
            self._breakers[provider] = CircuitBreaker(
                provider,
                failure_threshold=self.config.circuit_failure_threshold,
                cooldown_s=self.config.circuit_cooldown_s,
                clock=self._clock,
            )
        return self._breakers[provider]

    def chain(self, role: str, model: str | None = None) -> list[RoutingEntry]:
        """Строит цепочку роли; явная модель переопределяет первую ступень.

        Raises:
            RoutingError: если роль не сконфигурирована.
        """
        entries = self.config.chain(role)
        if not entries:
            raise RoutingError(role)
        if model:
            entries = [RoutingEntry(provider=entries[0].provider, model=model), *entries[1:]]
        return entries

    def _run(self, role: str, model: str | None, caller) -> object:
        """Общий обход цепочки: брейкеры, fallback, cost-контроль."""
        entries = self.chain(role, model)
        errors: list[ProviderError] = []
        for entry in entries:
            if not self.cost_breaker.allow():
                raise CostVelocityExceeded(self.cost_breaker.tokens_in_window(),
                                           self.cost_breaker.window_s)
            adapter = self.adapter(entry.provider)
            if adapter is None:
                errors.append(ProviderError(entry.provider, entry.model,
                                            "provider not configured or unusable", retryable=False))
                continue
            breaker = self.breaker(entry.provider)
            if not breaker.allow():
                errors.append(ProviderError(entry.provider, entry.model,
                                            "circuit breaker open", retryable=True))
                continue
            try:
                result = caller(adapter, entry)
            except ProviderError as error:
                if error.retryable:
                    breaker.record_failure()
                errors.append(error)
                log_marker("M-COMMON", "LLM", "FALLBACK", provider=entry.provider,
                           model=entry.model, reason=error.reason[:120])
                continue
            breaker.record_success()
            self.cost_breaker.record(getattr(result, "total_tokens", 0))
            return result
        log_marker("M-COMMON", "LLM", "ALL_FAILED", role=role, attempts=len(entries))
        raise AllProvidersExhausted(role, errors)

    def chat(
        self,
        messages: list[dict],
        *,
        role: str = DEFAULT_CHAT_ROLE,
        model: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        response_format: dict | None = None,
    ) -> ChatResult:
        """Выполняет чат-комплишен по цепочке роли.

        Args:
            messages: сообщения OpenAI-формата.
            role: роль маршрутизации (vision_model, text_model, ...).
            model: явная модель; переопределяет первую ступень цепочки.
            max_tokens: лимит генерации.
            temperature: температура (по умолчанию 0).
            response_format: например {"type": "json_object"}.

        Returns:
            ChatResult с текстом, провайдером, моделью и расходом токенов.
        """
        return self._run(  # type: ignore[return-value]
            role,
            model,
            lambda adapter, entry: adapter.chat(
                entry.model, messages,
                max_tokens=max_tokens, temperature=temperature,
                response_format=response_format,
            ),
        )

    def embed(self, texts: list[str], *, role: str = DEFAULT_EMBED_ROLE,
              model: str | None = None) -> EmbedResult:
        """Считает эмбеддинги по цепочке роли."""
        return self._run(  # type: ignore[return-value]
            role,
            model,
            lambda adapter, entry: adapter.embed(entry.model, texts),
        )


_router: LLMRouter | None = None


def get_router() -> LLMRouter:
    """Ленивый singleton роутера."""
    global _router
    if _router is None:
        _router = LLMRouter()
        log_marker("M-COMMON", "LLM", "ROUTER_READY", source=_router.config.source,
                   local_only=str(_router.config.local_only))
    return _router


def reset_router() -> None:
    """Сбрасывает singleton (тесты, смена конфигурации)."""
    global _router
    _router = None
