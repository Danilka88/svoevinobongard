# START_MODULE_CONTRACT
# PURPOSE: Датасет-билдер: экспорт ТОЛЬКО подтверждённых человеком меток в формат scripts/train/*
# SCOPE: export_confirmed
# DEPENDS: json, pathlib, packages.common.catalog, packages.common.logging, services.active_learning.app.store
# LINKS: M-ACTIVE-LEARNING, M-CATALOG
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# export_confirmed - выгрузка подтверждённых (и исправленных) меток в JSONL для обучения
# END_MODULE_MAP
"""Экспорт подтверждённых данных.

В датасет попадают только записи, по которым человек вынес решение
``confirmed`` или ``corrected``. Кейсы со статусом ``pending`` и ``rejected``
исключаются: именно в этом смысл human-in-the-loop.
"""
from __future__ import annotations

import json
from pathlib import Path

from packages.common.catalog import CatalogStore
from packages.common.logging import log_marker
from services.active_learning.app.store import FlywheelStore

__all__ = ["export_confirmed"]


def export_confirmed(store: FlywheelStore, catalog: CatalogStore,
                     out_path: str | Path) -> dict:
    """Экспортирует подтверждённые метки в JSONL для scripts/train.

    Args:
        store: хранилище flywheel.
        catalog: каталог (чтобы приложить метаданные для hard negatives).
        out_path: путь к JSONL со списком подтверждённых slug.

    Returns:
        Сводка: сколько подтверждено, исправлено, отброшено и уникальных slug.
    """
    rows = store.confirmed()
    per_slug: dict[str, dict] = {}
    corrected = 0
    dropped = 0

    for row in rows:
        slug = row.get("slug")
        if row.get("decision") == "corrected" and row.get("corrected_slug"):
            slug = row["corrected_slug"]
            corrected += 1
        if not slug or not catalog.get(slug):
            dropped += 1
            continue
        record = per_slug.setdefault(slug, {"slug": slug, "confirmed": 0, "sources": []})
        record["confirmed"] += 1
        record["sources"].append(row.get("decision", "confirmed"))

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for slug, payload in sorted(per_slug.items()):
            record = catalog.get(slug) or {}
            handle.write(json.dumps({
                "slug": slug,
                "image": record.get("image"),
                "winery": record.get("winery", ""),
                "grape": record.get("grape", ""),
                "region": record.get("region", ""),
                "confirmed": payload["confirmed"],
            }, ensure_ascii=False) + "\n")

    summary = {
        "reviewed_rows": len(rows),
        "confirmed": len(rows) - corrected,
        "corrected": corrected,
        "dropped_unknown_slug": dropped,
        "unique_slugs": len(per_slug),
        "out": str(target),
    }
    log_marker("M-ACTIVE-LEARNING", "dataset", "EXPORTED", slugs=len(per_slug),
               confirmed=summary["confirmed"], corrected=corrected, dropped=dropped)
    return summary
