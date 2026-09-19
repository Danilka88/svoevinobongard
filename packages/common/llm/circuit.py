# START_MODULE_CONTRACT
# PURPOSE: Отказоустойчивость LLM: per-provider circuit breaker и глобальный cost-velocity breaker
# SCOPE: CircuitState, CircuitBreaker, CostVelocityBreaker
# DEPENDS: packages.common.logging
# LINKS: M-COMMON
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# CircuitState - состояние брейкера: closed, open, half_open
# CircuitBreaker - per-provider брейкер по повторяемым отказам с cooldown и half-open пробой
# CostVelocityBreaker - глобальный брейкер по скорости расхода токенов за окно
# END_MODULE_MAP
"""Отказоустойчивость.

Circuit breaker изолирует «мертвый» провайдер: после N повторяемых отказов он
пропускается без вызова, через cooldown — пробуется снова (half-open).
Cost-velocity breaker ловит runaway-циклы, которые возвращают HTTP 200, но
сжигают бюджет: расход за окно выше планового множителя означает паузу.
"""
from __future__ import annotations

import time
from collections import deque
from enum import Enum
from typing import Callable

from packages.common.logging import log_marker

__all__ = ["CircuitState", "CircuitBreaker", "CostVelocityBreaker"]


class CircuitState(str, Enum):
    """Состояние circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Per-provider брейкер с cooldown и half-open пробой."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        cooldown_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self.failure_threshold = max(1, int(failure_threshold))
        self.cooldown_s = float(cooldown_s)
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._probe_pending = False

    @property
    def failures(self) -> int:
        """Число подряд идущих повторяемых отказов."""
        return self._failures

    @property
    def state(self) -> CircuitState:
        """Текущее состояние с учетом истекшего cooldown."""
        if self._opened_at is None:
            return CircuitState.CLOSED
        if self._clock() - self._opened_at < self.cooldown_s:
            return CircuitState.OPEN
        if not self._probe_pending:
            self._probe_pending = True
            log_marker("M-COMMON", "LLM", "CB_HALF_OPEN", provider=self.name)
        return CircuitState.HALF_OPEN

    def allow(self) -> bool:
        """Пускает вызов? OPEN — нет, CLOSED и HALF_OPEN — да."""
        if self.state is CircuitState.OPEN:
            log_marker("M-COMMON", "LLM", "CB_SKIP", provider=self.name,
                       failures=self._failures)
            return False
        return True

    def record_success(self) -> None:
        """Успешный вызов: закрывает брейкер."""
        if self._opened_at is not None:
            log_marker("M-COMMON", "LLM", "CB_CLOSED", provider=self.name)
        self._failures = 0
        self._opened_at = None
        self._probe_pending = False

    def record_failure(self) -> None:
        """Повторяемый отказ: открывает брейкер после порога."""
        self._failures += 1
        self._probe_pending = False
        if self._failures >= self.failure_threshold:
            self._opened_at = self._clock()
            log_marker("M-COMMON", "LLM", "CB_OPEN", provider=self.name,
                       failures=self._failures)


class CostVelocityBreaker:
    """Глобальный контроль скорости расхода токенов."""

    def __init__(
        self,
        window_s: float = 60.0,
        multiplier: float = 10.0,
        open_s: float = 30.0,
        plan_tokens_per_min: float = 20000.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.window_s = float(window_s)
        self.multiplier = float(multiplier)
        self.open_s = float(open_s)
        self.plan_tokens_per_min = float(plan_tokens_per_min)
        self._clock = clock
        self._events: deque[tuple[float, int]] = deque()
        self._opened_at: float | None = None

    @property
    def limit(self) -> float:
        """Порог расхода за окно."""
        return self.plan_tokens_per_min * self.multiplier

    def _prune(self) -> None:
        cutoff = self._clock() - self.window_s
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def tokens_in_window(self) -> int:
        """Сколько токенов израсходовано за окно."""
        self._prune()
        return sum(tokens for _, tokens in self._events)

    def record(self, tokens: int) -> None:
        """Учитывает расход токенов."""
        if tokens <= 0:
            return
        self._events.append((self._clock(), int(tokens)))
        self._prune()
        if self.tokens_in_window() > self.limit and self._opened_at is None:
            self._opened_at = self._clock()
            log_marker("M-COMMON", "LLM", "COST_BREAK", tokens=self.tokens_in_window(),
                       limit=int(self.limit), open_s=int(self.open_s))

    def allow(self) -> bool:
        """Пускает вызов? Открытый cost-брейкер блокирует на open_s."""
        if self._opened_at is None:
            return True
        if self._clock() - self._opened_at >= self.open_s:
            self._opened_at = None
            self._events.clear()
            return True
        log_marker("M-COMMON", "LLM", "COST_SKIP", tokens=self.tokens_in_window())
        return False
