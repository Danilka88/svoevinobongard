"""Тесты C-048: состояние недоступности сервиса и отсутствие необработанных отклонений.

Поведение браузера в питоновском тесте не воспроизвести, поэтому проверяются гарантии в исходниках
интерфейса: каждый экран различает сбой связи и пустой результат, а вспомогательные вызовы gateway
не выбрасывают исключений. Живая проверка выполняется поведенческим тестом интерфейса.
"""
from __future__ import annotations

from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "apps" / "web"
PAGES = ("index", "metrics", "catalog", "sommelier", "how")


def _read(relative: str) -> str:
    return (WEB / relative).read_text(encoding="utf-8")


def test_component_exists_and_has_retry() -> None:
    component = _read("components/ServiceUnavailable.vue")
    assert "Сервис недоступен" in component
    assert "emit('retry')" in component


def test_every_page_wires_unavailable_state() -> None:
    """Сбой связи должен показываться явно, а не как пустой результат."""
    for page in PAGES:
        source = _read(f"pages/{page}.vue")
        assert "unavailable" in source, f"{page}: нет состояния недоступности"
        assert "<ServiceUnavailable" in source, f"{page}: блок недоступности не подключён"


def test_catalog_does_not_mask_outage_as_empty() -> None:
    """Пустая выдача и сбой связи — разные состояния."""
    source = _read("pages/catalog.vue")
    assert 'v-else-if="!unavailable"' in source


def test_metrics_distinguishes_outage_from_no_data() -> None:
    source = _read("pages/metrics.vue")
    assert "?.unavailable" in source


def test_sommelier_distinguishes_outage_from_empty_pick() -> None:
    source = _read("pages/sommelier.vue")
    assert "source === 'unavailable'" in source


def test_auxiliary_gateway_calls_do_not_throw() -> None:
    """Ни один вспомогательный вызов не должен создавать необработанный отказ промиса."""
    source = _read("composables/useGateway.ts")
    assert "scanExplain" in source
    assert source.count("catch {") >= 7, "не все вспомогательные вызовы обёрнуты"


def test_telemetry_sending_is_guarded() -> None:
    source = _read("composables/useTelemetry.ts")
    assert "sendBeacon" in source
    assert source.count("catch {") >= 3


def test_error_state_e2e_covers_unavailable_block() -> None:
    """Проверка ошибок описывает новое состояние: блок и кнопка повтора."""
    script = _read("e2e/error-state.js")
    assert script.count("unavailable block with retry") >= 3
    assert "does not mask outage as empty result" in script
