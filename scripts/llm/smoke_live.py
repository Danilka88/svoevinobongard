#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Живой smoke-test LLM Router против реальных провайдеров (Ollama) с печатью фактического ответа
# SCOPE: parse_args, main
# DEPENDS: packages.common.llm.router, packages.common.llm.config
# LINKS: M-COMMON
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# parse_args - разбор аргументов CLI
# main - один короткий чат и один вызов эмбеддингов с печатью провайдера и латентности
# END_MODULE_MAP
"""Живой smoke-test роутера.

Проверяет реальную цепочку провайдеров: один короткий чат и один вызов
эмбеддингов. Если провайдер недоступен, роутер уходит на следующий, а при
исчерпании цепочки скрипт печатает ошибку и завершается кодом 1 — без имитации
успеха.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from packages.common.llm.config import load_router_config  # noqa: E402
from packages.common.llm.errors import AllProvidersExhausted, CostVelocityExceeded  # noqa: E402
from packages.common.llm.router import LLMRouter  # noqa: E402

__all__ = ["parse_args", "main"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы CLI."""
    parser = argparse.ArgumentParser(description="Живой smoke-test LLM Router")
    parser.add_argument("--role", default="text_model")
    parser.add_argument("--model", default="")
    parser.add_argument("--prompt", default="Ответь одним словом: столица Франции?")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--embed-role", default="embedding_model")
    parser.add_argument("--skip-embed", action="store_true")
    parser.add_argument("--config", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Выполняет живой чат и эмбеддинги, печатает фактический результат."""
    args = parse_args(argv)
    config = load_router_config(args.config or None)
    router = LLMRouter(config=config)
    print(f"config:      {config.source}")
    print(f"local_only:  {config.local_only}")
    print(f"role chain:  {args.role} -> "
          + ", ".join(f"{entry.provider}/{entry.model}" for entry in config.chain(args.role)))

    try:
        result = router.chat(
            [{"role": "user", "content": args.prompt}],
            role=args.role,
            model=args.model or None,
            max_tokens=args.max_tokens,
            temperature=0.0,
        )
    except (AllProvidersExhausted, CostVelocityExceeded) as error:
        print(f"CHAT FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    print("\n=== LIVE CHAT ===")
    print(f"provider:    {result.provider}")
    print(f"model:       {result.model}")
    print(f"latency_ms:  {result.latency_ms}")
    print(f"tokens:      prompt {result.prompt_tokens}, completion {result.completion_tokens}, "
          f"total {result.total_tokens}")
    print(f"finish:      {result.finish_reason}")
    print(f"reply:       {result.text.strip()[:300]}")
    if not result.text.strip():
        print("WARNING: пустой content при finish_reason=length: модель reasoning-стиля "
              "(qwen3.5, gemma4:12b) израсходовала бюджет на внутренние рассуждения. "
              "Увеличьте --max-tokens.", file=sys.stderr)

    if args.skip_embed:
        return 0
    try:
        embedded = router.embed(["сухое красное вино", "белое полусладкое"], role=args.embed_role)
    except (AllProvidersExhausted, CostVelocityExceeded) as error:
        print(f"\nEMBED FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print("\n=== LIVE EMBED ===")
    print(f"provider:    {embedded.provider}")
    print(f"model:       {embedded.model}")
    print(f"vectors:     {len(embedded.vectors)} x {len(embedded.vectors[0]) if embedded.vectors else 0}")
    print(f"latency_ms:  {embedded.latency_ms}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
