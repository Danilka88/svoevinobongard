# START_MODULE_CONTRACT
# PURPOSE: Верификация LLM Router без сети: fallback chain, circuit breaker, cost-velocity breaker, локальный режим
# SCOPE: test_config_resolves_env, test_config_marks_unusable_without_key, test_chat_falls_back_to_next_provider, test_chat_all_fail_raises, test_model_override_keeps_fallback, test_circuit_opens_and_skips_provider, test_circuit_half_open_recovers, test_cost_breaker_opens_and_blocks, test_local_only_chain_excludes_cloud, test_routing_error_for_unknown_role, test_embed_uses_chain
# DEPENDS: packages.common.llm.router, packages.common.llm.adapters, packages.common.llm.circuit, packages.common.llm.config, packages.common.llm.errors
# LINKS: M-COMMON
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# StubAdapter - адаптер-заглушка со сценарием успехов и отказов
# test_config_resolves_env - ${VAR} и ${VAR:-default} подставляются, отсутствие ключа делает провайдера непригодным
# test_config_marks_unusable_without_key - облачный провайдер без ключа выпадает из цепочки
# test_chat_falls_back_to_next_provider - отказ первого провайдера переводит вызов на следующий
# test_chat_all_fail_raises - исчерпание цепочки поднимает AllProvidersExhausted со списком причин
# test_model_override_keeps_fallback - явная модель переопределяет первую ступень и сохраняет fallback
# test_circuit_opens_and_skips_provider - после порога отказов провайдер пропускается без вызова
# test_circuit_half_open_recovers - cooldown переводит брейкер в half-open, успех закрывает его
# test_cost_breaker_opens_and_blocks - превышение расхода открывает cost-брейкер и блокирует вызовы
# test_cost_breaker_blocks_router - открытый cost-брейкер останавливает chat роутера
# test_local_only_chain_excludes_cloud - локальный режим оставляет только локальных провайдеров
# test_routing_error_for_unknown_role - неизвестная роль дает RoutingError
# test_embed_uses_chain - embed идет по цепочке роли и возвращает векторы
# END_MODULE_MAP
"""V-M-COMMON: LLM Router без сети (stub-адаптеры)."""
from __future__ import annotations

import pytest

from packages.common.llm.adapters import BaseAdapter, ChatResult, EmbedResult
from packages.common.llm.circuit import CircuitBreaker, CircuitState, CostVelocityBreaker
from packages.common.llm.config import (
    ProviderConfig,
    RoutingEntry,
    RouterConfig,
    load_router_config,
)
from packages.common.llm.errors import (
    AllProvidersExhausted,
    CostVelocityExceeded,
    ProviderError,
    RoutingError,
)
from packages.common.llm.router import LLMRouter

__all__ = [
    "StubAdapter",
    "test_config_resolves_env",
    "test_config_marks_unusable_without_key",
    "test_chat_falls_back_to_next_provider",
    "test_chat_all_fail_raises",
    "test_model_override_keeps_fallback",
    "test_circuit_opens_and_skips_provider",
    "test_circuit_half_open_recovers",
    "test_cost_breaker_opens_and_blocks",
    "test_cost_breaker_blocks_router",
    "test_local_only_chain_excludes_cloud",
    "test_routing_error_for_unknown_role",
    "test_embed_uses_chain",
]


class StubAdapter(BaseAdapter):
    """Адаптер-заглушка: отвечает или падает по сценарию."""

    def __init__(self, name: str, *, fail: int = 0, retryable: bool = True,
                 text: str = "ok", local: bool = True) -> None:
        super().__init__(ProviderConfig(name=name, base_url="http://stub", api_key="stub", local=local))
        self.calls = 0
        self.fail = fail
        self.retryable = retryable
        self.text = text
        self.last_extra: dict | None = None

    def _maybe_fail(self, model: str) -> None:
        self.calls += 1
        if self.fail > 0:
            self.fail -= 1
            raise ProviderError(self.name, model, "stub failure", retryable=self.retryable)

    def chat(self, model: str, messages: list[dict], *, max_tokens: int = 512,
             temperature: float = 0.0, response_format: dict | None = None,
             extra: dict | None = None, timeout_s: float | None = None) -> ChatResult:
        self._maybe_fail(model)
        self.last_extra = extra
        return ChatResult(text=f"{self.text}:{model}", model=model, provider=self.name,
                          prompt_tokens=10, completion_tokens=5, total_tokens=15,
                          latency_ms=1.0, finish_reason="stop")

    def embed(self, model: str, texts: list[str]) -> EmbedResult:
        self._maybe_fail(model)
        return EmbedResult(vectors=[[0.1, 0.2] for _ in texts], model=model,
                           provider=self.name, total_tokens=7, latency_ms=1.0)


def _config(local_only: bool = False, threshold: int = 3) -> RouterConfig:
    """Конфигурация с двумя локальными и одним облачным провайдером."""
    return RouterConfig(
        providers={
            "p1": ProviderConfig(name="p1", base_url="http://p1", api_key="k", local=True),
            "p2": ProviderConfig(name="p2", base_url="http://p2", api_key="k", local=True),
            "cloud": ProviderConfig(name="cloud", base_url="http://cloud", api_key="", local=False),
        },
        routing={
            "text_model": [RoutingEntry("p1", "m1"), RoutingEntry("p2", "m2"),
                           RoutingEntry("cloud", "m3")],
            "embedding_model": [RoutingEntry("p1", "e1"), RoutingEntry("p2", "e2")],
        },
        local_only=local_only,
        circuit_failure_threshold=threshold,
        cost_plan_tokens_per_min=100.0,
        cost_multiplier=10.0,
    )


def _router(**kwargs) -> LLMRouter:
    """Роутер со stub-адаптерами."""
    return LLMRouter(
        config=kwargs.pop("config", None) or _config(**kwargs),
        adapters={"p1": StubAdapter("p1", text="first"),
                  "p2": StubAdapter("p2", text="second"),
                  "cloud": StubAdapter("cloud", text="cloud", local=False)},
    )


def test_config_resolves_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("STUB_KEY", "secret-value")
    monkeypatch.delenv("STUB_MISSING", raising=False)
    source = tmp_path / "providers.yaml"
    source.write_text(
        "providers:\n"
        "  local:\n"
        "    base_url: ${STUB_URL:-http://127.0.0.1:9/v1}\n"
        "    api_key: ${STUB_KEY}\n"
        "    local: true\n"
        "  cloud:\n"
        "    base_url: https://cloud.example/v1\n"
        "    api_key: ${STUB_MISSING}\n"
        "routing:\n"
        "  text_model:\n"
        "    - {provider: local, model: m1}\n"
        "    - {provider: cloud, model: m2}\n",
        encoding="utf-8",
    )
    config = load_router_config(source)
    assert config.providers["local"].base_url == "http://127.0.0.1:9/v1"
    assert config.providers["local"].api_key == "secret-value"
    assert config.providers["local"].usable is True
    assert config.providers["cloud"].api_key == ""
    assert config.providers["cloud"].usable is False
    assert [entry.model for entry in config.chain("text_model")] == ["m1", "m2"]


def test_config_marks_unusable_without_key() -> None:
    config = _config()
    assert config.providers["cloud"].usable is False
    router = LLMRouter(config=config, adapters={"p1": StubAdapter("p1")})
    assert router.adapter("cloud") is None
    assert router.adapter("p1") is not None


def test_chat_falls_back_to_next_provider() -> None:
    router = _router(config=_config())
    router._adapters["p1"] = StubAdapter("p1", fail=1)
    result = router.chat([{"role": "user", "content": "hi"}])
    assert result.provider == "p2"
    assert result.model == "m2"
    assert result.total_tokens == 15


def test_chat_all_fail_raises() -> None:
    router = _router(config=_config(threshold=99))
    for name in ("p1", "p2", "cloud"):
        router._adapters[name] = StubAdapter(name, fail=99)
    with pytest.raises(AllProvidersExhausted) as error:
        router.chat([{"role": "user", "content": "hi"}])
    assert error.value.role == "text_model"
    assert len(error.value.errors) >= 2


def test_model_override_keeps_fallback() -> None:
    router = _router(config=_config())
    chain = router.chain("text_model", model="custom-model")
    assert chain[0].model == "custom-model"
    assert chain[0].provider == "p1"
    assert [entry.model for entry in chain[1:]] == ["m2", "m3"]


def test_circuit_opens_and_skips_provider() -> None:
    router = _router(config=_config(threshold=3))
    router._adapters["p1"] = StubAdapter("p1", fail=99)
    for _ in range(3):
        router.chat([{"role": "user", "content": "hi"}])
    assert router.breaker("p1").state is CircuitState.OPEN
    calls_before = router._adapters["p1"].calls
    result = router.chat([{"role": "user", "content": "hi"}])
    assert result.provider == "p2"
    assert router._adapters["p1"].calls == calls_before  # открытый брейкер пропускает вызов


def test_circuit_half_open_recovers() -> None:
    now = {"value": 0.0}
    breaker = CircuitBreaker("p", failure_threshold=2, cooldown_s=10.0,
                             clock=lambda: now["value"])
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    assert breaker.allow() is False
    now["value"] = 11.0
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.allow() is True
    breaker.record_success()
    assert breaker.state is CircuitState.CLOSED


def test_cost_breaker_opens_and_blocks() -> None:
    now = {"value": 0.0}
    breaker = CostVelocityBreaker(window_s=60, multiplier=10, open_s=30,
                                  plan_tokens_per_min=100, clock=lambda: now["value"])
    assert breaker.allow() is True
    breaker.record(5000)
    assert breaker.tokens_in_window() == 5000
    assert breaker.allow() is False
    now["value"] = 31.0
    assert breaker.allow() is True


def test_cost_breaker_blocks_router() -> None:
    router = _router(config=_config())
    router.cost_breaker.record(10_000)
    with pytest.raises(CostVelocityExceeded):
        router.chat([{"role": "user", "content": "hi"}])


def test_local_only_chain_excludes_cloud() -> None:
    config = _config(local_only=True)
    router = _router(config=config)
    chain = router.chain("text_model")
    assert [entry.provider for entry in chain] == ["p1", "p2"]
    assert "cloud" not in {entry.provider for entry in chain}


def test_routing_error_for_unknown_role() -> None:
    router = _router(config=_config())
    with pytest.raises(RoutingError):
        router.chat([{"role": "user", "content": "hi"}], role="missing_role")


def test_embed_uses_chain() -> None:
    router = _router(config=_config())
    result = router.embed(["a", "b"])
    assert result.provider == "p1"
    assert len(result.vectors) == 2
    assert result.total_tokens == 7
