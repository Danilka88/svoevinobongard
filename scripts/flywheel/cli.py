#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: CLI flywheel: ingest, stats, queue, review, export, train, registry, eval — весь цикл human-in-the-loop
# SCOPE: parse_args, main
# DEPENDS: argparse, json, subprocess, packages.common.catalog, packages.common.config,
#          services.active_learning.app.dataset, services.active_learning.app.store
# LINKS: M-ACTIVE-LEARNING, M-EVAL, M-CATALOG
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# REGISTRY_PATH - путь к реестру версий модели (models/registry.json)
# parse_args - разбор аргументов CLI
# main - выполнение команды flywheel: ingest, stats, queue, review, export, train, registry, eval
# END_MODULE_MAP
"""CLI flywheel.

Полный цикл без HTTP: ingest → queue → review → export → train → registry.
Переобучение вызывает существующие скрипты C-009 (`synthetic_pairs.py`,
`train_head.py`) и оценку C-003 (`baseline_report.py`), а не дублирует их.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from packages.common.catalog import CatalogStore  # noqa: E402
from packages.common.config import settings  # noqa: E402
from services.active_learning.app.dataset import export_confirmed  # noqa: E402
from services.active_learning.app.store import FlywheelStore, ingest_jsonl  # noqa: E402

__all__ = ["parse_args", "main"]

REGISTRY_PATH = _ROOT / "models" / "registry.json"


def _run(command: list[str]) -> int:
    """Запускает дочерний процесс, наследуя вывод."""
    print(f"$ {' '.join(command)}")
    return subprocess.call(command, cwd=str(_ROOT))


def _load_registry() -> list[dict]:
    if not REGISTRY_PATH.is_file():
        return []
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except ValueError:
        return []


def _save_registry(entries: list[dict]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы CLI."""
    parser = argparse.ArgumentParser(description="Flywheel: логи, ревью, датасет, переобучение")
    parser.add_argument("command", choices=["ingest", "stats", "queue", "review", "export",
                                            "train", "registry", "eval"])
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--decision", default="confirmed",
                        choices=["confirmed", "corrected", "rejected"])
    parser.add_argument("--corrected-slug", default="")
    parser.add_argument("--reviewer", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--per-slug", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Выполняет выбранную команду flywheel."""
    args = parse_args(argv)
    store = FlywheelStore(settings.flywheel_db_path)
    try:
        if args.command == "ingest":
            added = ingest_jsonl(store, settings.flywheel_log_path)
            print(f"ingest: добавлено {added} записей из {settings.flywheel_log_path}")
            print(json.dumps(store.stats().as_dict(), ensure_ascii=False))
            return 0

        if args.command == "stats":
            print(json.dumps(store.stats().as_dict(), ensure_ascii=False, indent=2))
            return 0

        if args.command == "queue":
            items = store.queue(limit=args.limit)
            print(f"queue: {len(items)} кейсов (uncertain впереди)")
            for item in items:
                print(f"  id={item['id']:<4} uncertain={int(item['uncertain'])} "
                      f"margin={item['margin']:.4f} slug={item['slug']} path={item['path']}")
            return 0

        if args.command == "review":
            if not args.id:
                print("ERROR: --id обязателен", file=sys.stderr)
                return 2
            store.review(args.id, args.decision, args.corrected_slug, args.reviewer)
            print(f"review: id={args.id} -> {args.decision}"
                  + (f" ({args.corrected_slug})" if args.corrected_slug else ""))
            print(json.dumps(store.stats().as_dict(), ensure_ascii=False))
            return 0

        if args.command == "export":
            out = args.out or settings.flywheel_confirmed_path
            catalog = CatalogStore.from_jsonl(settings.catalog_jsonl)
            summary = export_confirmed(store, catalog, out)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0

        if args.command == "train":
            pairs = _ROOT / "data" / "train-confirmed" / "pairs.jsonl"
            embeddings_dir = _ROOT / "data" / "train-confirmed"
            if not pairs.is_file():
                code = _run([sys.executable, str(_ROOT / "scripts" / "train" / "synthetic_pairs.py"),
                             "--slugs-file", str(_ROOT / settings.flywheel_confirmed_path),
                             "--per-slug", str(args.per_slug), "--seed", str(args.seed),
                             "--out", str(embeddings_dir)])
                if code != 0:
                    return code
            head_out = args.out or "models/head-flywheel.pt"
            report = "reports/train_head_flywheel.json"
            code = _run([sys.executable, str(_ROOT / "scripts" / "train" / "train_head.py"),
                         "--pairs", str(embeddings_dir / "pairs.jsonl"),
                         "--embeddings", str(embeddings_dir / "embeddings.npy"),
                         "--epochs", str(args.epochs), "--out", head_out, "--report", report])
            if code != 0:
                return code
            metrics = {}
            if (_ROOT / report).is_file():
                metrics = json.loads((_ROOT / report).read_text(encoding="utf-8")).get("metrics", {})
            entries = _load_registry()
            version = f"flywheel-{len(entries) + 1}"
            entries.append({
                "version": version,
                "head": head_out,
                "report": report,
                "confirmed_slugs": len(store.confirmed()),
                "metrics": metrics,
                "source": "flywheel (human-in-the-loop)",
            })
            _save_registry(entries)
            print(f"registry: записана версия {version} ({head_out})")
            return 0

        if args.command == "registry":
            entries = _load_registry()
            if not entries:
                print("registry пуст: запустите 'train' после ревью подтверждённых кейсов")
                return 0
            print(json.dumps(entries, ensure_ascii=False, indent=2))
            return 0

        if args.command == "eval":
            return _run([sys.executable, str(_ROOT / "scripts" / "eval" / "baseline_report.py"),
                         "--index", "data/index", "--samples", str(args.samples),
                         "--seed", str(args.seed), "--normalize", "--geo",
                         "--out", args.out or "reports/baseline-flywheel.json"])
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
