# START_MODULE_CONTRACT
# PURPOSE: Типизированные ошибки LLM-роутера: отказ провайдера, исчерпание цепочки, cost-лимит, неизвестная роль
# SCOPE: LLMError, ProviderError, AllProvidersExhausted, CostVelocityExceeded, RoutingError
# DEPENDS: none (stdlib)
# LINKS: M-COMMON
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# LLMError - базовое исключение LLM-слоя
# ProviderError - отказ одного провайдера с признаком retryable
# AllProvidersExhausted - вся цепочка провайдеров отказала
# CostVelocityExceeded - сработал cost-velocity breaker
# RoutingError - роль не найдена в конфигурации
# END_MODULE_MAP
"""Ошибки LLM-роутера.

Разделение retryable и не-retryable важно: в circuit breaker идут только
повторяемые сбои (5xx, 429, таймаут, ошибка соединения), а ошибки вида
«неверный запрос» не должны открывать брейкер.
"""
from __future__ import annotations

__all__ = [
    "LLMError",
    "ProviderError",
    "AllProvidersExhausted",
    "CostVelocityExceeded",
    "RoutingError",
]


class LLMError(Exception):
    """Базовое исключение LLM-слоя."""


class ProviderError(LLMError):
    """Отказ одного провайдера."""

    def __init__(self, provider: str, model: str, reason: str, *, retryable: bool = True) -> None:
        super().__init__(f"{provider}/{model}: {reason}")
        self.provider = provider
        self.model = model
        self.reason = reason
        self.retryable = retryable


class AllProvidersExhausted(LLMError):
    """Цепочка провайдеров исчерпана."""

    def __init__(self, role: str, errors: list[ProviderError]) -> None:
        details = "; ".join(str(error) for error in errors) or "no usable providers"
        super().__init__(f"all providers failed for role '{role}': {details}")
        self.role = role
        self.errors = errors


class CostVelocityExceeded(LLMError):
    """Сработал cost-velocity breaker: расход токенов выше планового."""

    def __init__(self, tokens_in_window: int, window_s: float) -> None:
        super().__init__(
            f"cost-velocity breaker open: {tokens_in_window} tokens in {window_s:.0f}s"
        )
        self.tokens_in_window = tokens_in_window
        self.window_s = window_s


class RoutingError(LLMError):
    """Роль не найдена или у роли нет пригодных провайдеров."""

    def __init__(self, role: str) -> None:
        super().__init__(f"role '{role}' is not configured or has no enabled providers")
        self.role = role
