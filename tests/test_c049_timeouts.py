"""Тесты C-049: таймауты вызовов gateway и отсутствие необработанных отклонений.

Поведение браузера в питоновском тесте не воспроизвести, поэтому здесь проверяются гарантии в
исходниках: каждый вызов ограничен по времени, значения берутся из конфигурации, а срабатывание
таймаута описано как состояние недоступности. Живая проверка механизма выполняется скриптом
проверки на «висящем» соединении тем же кодом, что использует клиент.
"""
from __future__ import annotations

import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "apps" / "web"


def _read(relative: str) -> str:
    return (WEB / relative).read_text(encoding="utf-8")


def test_timeout_utility_exists_and_classifies() -> None:
    source = _read("utils/request.ts")
    assert "export function requestTimeout" in source
    assert "AbortSignal.timeout" in source
    assert "export function isTimeout" in source
    assert "TimeoutError" in source and "AbortError" in source


def test_every_gateway_call_has_timeout() -> None:
    """Ни один вызов gateway не должен ждать ответ бесконечно."""
    source = _read("composables/useGateway.ts")
    calls = re.findall(r"\$fetch<", source)
    signals = re.findall(r"signal: requestTimeout\(", source)
    # Инвариант: таймаут стоит на КАЖДОМ вызове. Точное число вызовов растёт от изменения к изменению,
    # поэтому проверяется равенство, а не жёсткая константа (C-054 добавил вызов обратной связи).
    assert len(calls) >= 8, f"вызовов gateway: {len(calls)}"
    assert len(signals) == len(calls), f"таймаутов: {len(signals)} на {len(calls)} вызовов"


def test_scan_gets_larger_budget_than_auxiliary_calls() -> None:
    source = _read("composables/useGateway.ts")
    assert "scanTimeoutMs" in source and "timeoutMs" in source
    scan = int(re.search(r"apiScanTimeoutMs \|\| (\d+)", source).group(1))
    plain = int(re.search(r"apiTimeoutMs \|\| (\d+)", source).group(1))
    assert scan > plain, f"бюджет сканирования {scan} должен быть больше {plain}"
    assert 10000 <= scan <= 20000
    assert 3000 <= plain <= 10000


def test_timeouts_are_configurable() -> None:
    config = _read("nuxt.config.ts")
    assert "apiTimeoutMs" in config
    assert "apiScanTimeoutMs" in config


def test_timeout_is_treated_as_unavailable_not_exception() -> None:
    """Срабатывание таймаута ведёт к различимому результату, а не к исключению."""
    source = _read("composables/useGateway.ts")
    assert source.count("catch {") >= 7
    assert "unavailable: true" in source


def test_telemetry_sending_is_not_time_bounded() -> None:
    """Телеметрия не должна зависеть от основного сценария и по времени не ограничивается."""
    source = _read("composables/useTelemetry.ts")
    assert "requestTimeout" not in source


def test_timeout_check_uses_real_utility() -> None:
    """Проверка на зависшем соединении использует ту же утилиту, что и клиент."""
    script = _read("e2e/timeout-check.ts")
    assert "from '../utils/request'" in script
    assert "unhandledRejection" in script
    assert "unavailable" in script
