# START_MODULE_CONTRACT
# PURPOSE: Загрузка providers.yaml: провайдеры, роли маршрутизации и лимиты с подстановкой env
# SCOPE: ProviderConfig, RoutingEntry, RouterConfig, resolve_env, load_router_config
# DEPENDS: yaml, os, dataclasses, pathlib
# LINKS: M-COMMON
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# ProviderConfig - конфигурация одного провайдера
# RoutingEntry - пара провайдер и модель в цепочке роли
# RouterConfig - полная конфигурация роутера: провайдеры, роли, лимиты, локальный режим
# resolve_env - подстановка ${VAR} и ${VAR:-default} из окружения
# load_router_config - чтение providers.yaml с подстановкой env и валидацией
# END_MODULE_MAP
"""Конфигурация LLM-роутера.

Секреты живут только в окружении: значения вида ``${OPENROUTER_API_KEY}``
подставляются при загрузке. Если ключ не задан, провайдер помечается
непригодным и просто выпадает из цепочки — роутер не падает.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

__all__ = [
    "ProviderConfig",
    "RoutingEntry",
    "RouterConfig",
    "resolve_env",
    "load_router_config",
]

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
DEFAULT_PROVIDERS_FILE = "config/providers.yaml"
DEFAULT_ROLE = "text_model"


def resolve_env(value: str) -> str:
    """Подставляет ``${VAR}`` и ``${VAR:-default}`` значениями окружения."""

    def _replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        found = os.getenv(name)
        if found not in (None, ""):
            return found
        return default if default is not None else ""

    return _ENV_PATTERN.sub(_replace, value)


def _resolve_tree(node: object) -> object:
    """Рекурсивно подставляет env во все строки дерева конфигурации."""
    if isinstance(node, str):
        return resolve_env(node)
    if isinstance(node, dict):
        return {key: _resolve_tree(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_resolve_tree(item) for item in node]
    return node


@dataclass(frozen=True)
class ProviderConfig:
    """Конфигурация одного провайдера."""

    name: str
    type: str = "openai-compat"
    base_url: str = ""
    api_key: str = ""
    local: bool = False
    timeout_s: float = 60.0
    headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True

    @property
    def usable(self) -> bool:
        """Провайдер пригоден: включен, есть base_url и ключ (или он локальный)."""
        if not self.enabled or not self.base_url:
            return False
        return bool(self.api_key) or self.local


@dataclass(frozen=True)
class RoutingEntry:
    """Одна ступень цепочки: провайдер и модель."""

    provider: str
    model: str


@dataclass(frozen=True)
class RouterConfig:
    """Полная конфигурация роутера."""

    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    routing: dict[str, list[RoutingEntry]] = field(default_factory=dict)
    local_only: bool = False
    circuit_failure_threshold: int = 3
    circuit_cooldown_s: float = 30.0
    cost_window_s: float = 60.0
    cost_multiplier: float = 10.0
    cost_open_s: float = 30.0
    cost_plan_tokens_per_min: float = 20000.0
    source: str = ""

    def chain(self, role: str) -> list[RoutingEntry]:
        """Цепочка роли с учетом локального режима."""
        entries = list(self.routing.get(role, []))
        if not self.local_only:
            return entries
        return [entry for entry in entries
                if (self.providers.get(entry.provider) or ProviderConfig(name=entry.provider)).local]


def load_router_config(path: str | Path | None = None) -> RouterConfig:
    """Читает providers.yaml и собирает RouterConfig.

    Args:
        path: путь к файлу; по умолчанию env LLM_PROVIDERS_FILE или config/providers.yaml.

    Returns:
        RouterConfig с подставленными env-значениями.

    Raises:
        FileNotFoundError: если файл конфигурации отсутствует.
    """
    target = Path(path or os.getenv("LLM_PROVIDERS_FILE") or DEFAULT_PROVIDERS_FILE)
    if not target.is_file():
        raise FileNotFoundError(f"providers config not found: {target}")
    raw = _resolve_tree(yaml.safe_load(target.read_text(encoding="utf-8")) or {})
    providers: dict[str, ProviderConfig] = {}
    for name, payload in (raw.get("providers") or {}).items():
        data = payload or {}
        providers[name] = ProviderConfig(
            name=name,
            type=str(data.get("type", "openai-compat")),
            base_url=str(data.get("base_url", "")),
            api_key=str(data.get("api_key", "")),
            local=bool(data.get("local", False)),
            timeout_s=float(data.get("timeout_s", 60.0)),
            headers={str(key): str(value) for key, value in (data.get("headers") or {}).items()},
            enabled=bool(data.get("enabled", True)),
        )

    routing: dict[str, list[RoutingEntry]] = {}
    for role, entries in (raw.get("routing") or {}).items():
        routing[str(role)] = [
            RoutingEntry(provider=str(item["provider"]), model=str(item["model"]))
            for item in (entries or [])
            if item.get("provider") and item.get("model")
        ]

    limits = raw.get("limits") or {}
    local_only = os.getenv("LLM_LOCAL_ONLY", "0").lower() not in {"0", "false", "no", ""}
    return RouterConfig(
        providers=providers,
        routing=routing,
        local_only=local_only,
        circuit_failure_threshold=int(limits.get("circuit_failure_threshold", 3)),
        circuit_cooldown_s=float(limits.get("circuit_cooldown_s", 30.0)),
        cost_window_s=float(limits.get("cost_window_s", 60.0)),
        cost_multiplier=float(limits.get("cost_multiplier", 10.0)),
        cost_open_s=float(limits.get("cost_open_s", 30.0)),
        cost_plan_tokens_per_min=float(limits.get("cost_plan_tokens_per_min", 20000.0)),
        source=str(target),
    )
