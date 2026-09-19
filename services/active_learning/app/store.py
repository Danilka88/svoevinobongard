# START_MODULE_CONTRACT
# PURPOSE: SQLite-хранилище flywheel: приём JSONL-логов, очередь ревью, human-in-the-loop решения и статистика
# SCOPE: FlywheelStore, ingest_jsonl
# DEPENDS: json, sqlite3, dataclasses, pathlib, packages.common.logging
# LINKS: M-ACTIVE-LEARNING
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# FlywheelStore - хранилище запросов и очереди ревью с ручным подтверждением метки
# ingest_jsonl - загрузка JSONL-лога gateway в SQLite без дублей по хешу и времени
# END_MODULE_MAP
"""Хранилище flywheel.

Метка становится «золотой» только после решения человека: до этого запись висит
в очереди ревью. Автоматическая разметка сознательно не используется — C-004 и
C-010 показали, что локальные VLM и OCR не дают надёжного текстового сигнала.

Без PII: хранится sha256 изображения, ранжирование, решение и тайминги.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from packages.common.logging import log_marker

__all__ = ["FlywheelStore", "ingest_jsonl"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    image_sha256 TEXT NOT NULL,
    slug TEXT,
    confidence REAL DEFAULT 0,
    margin REAL DEFAULT 0,
    path TEXT DEFAULT '',
    latency_ms REAL DEFAULT 0,
    not_found INTEGER DEFAULT 0,
    uncertain INTEGER DEFAULT 0,
    top_k TEXT DEFAULT '[]',
    normalize_method TEXT DEFAULT '',
    decision TEXT DEFAULT 'pending',
    corrected_slug TEXT,
    reviewer TEXT,
    reviewed_at TEXT,
    UNIQUE(image_sha256, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_queries_decision ON queries(decision);
CREATE INDEX IF NOT EXISTS idx_queries_uncertain ON queries(uncertain);
"""

VERDICTS = ("pending", "confirmed", "corrected", "rejected")


@dataclass
class ReviewStats:
    """Сводка по хранилищу."""

    total: int = 0
    pending: int = 0
    confirmed: int = 0
    corrected: int = 0
    rejected: int = 0
    uncertain: int = 0

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "pending": self.pending,
            "confirmed": self.confirmed,
            "corrected": self.corrected,
            "rejected": self.rejected,
            "uncertain": self.uncertain,
        }


class FlywheelStore:
    """SQLite-хранилище запросов и ревью."""

    def __init__(self, db_path: str | Path = "data/flywheel/flywheel.db") -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        """Закрывает соединение."""
        self._conn.close()

    def record(self, entry: dict) -> int:
        """Записывает один лог запроса. Дубли по (image_sha256, timestamp) игнорируются."""
        cursor = self._conn.execute(
            """INSERT OR IGNORE INTO queries
               (timestamp, image_sha256, slug, confidence, margin, path, latency_ms,
                not_found, uncertain, top_k, normalize_method)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                str(entry.get("timestamp", "")),
                str(entry.get("image_sha256", "")),
                entry.get("slug"),
                float(entry.get("confidence", 0.0) or 0.0),
                float(entry.get("margin", 0.0) or 0.0),
                str(entry.get("path", "")),
                float(entry.get("latency_ms", 0.0) or 0.0),
                int(bool(entry.get("not_found", False))),
                int(bool(entry.get("uncertain", False))),
                json.dumps(entry.get("top_k", []), ensure_ascii=False),
                str(entry.get("normalize_method", "")),
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid or 0)

    def queue(self, limit: int = 20, only_pending: bool = True) -> list[dict]:
        """Очередь ревью: неуверенные и не найденные кейсы."""
        clause = "WHERE decision = 'pending'" if only_pending else ""
        rows = self._conn.execute(
            f"""SELECT * FROM queries {clause}
                ORDER BY uncertain DESC, margin ASC, id ASC LIMIT ?""",
            (int(limit),),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def review(self, entry_id: int, decision: str, corrected_slug: str = "",
               reviewer: str = "") -> bool:
        """Выносит решение человека: confirmed | corrected | rejected.

        Raises:
            ValueError: для недопустимого решения или отсутствующей записи.
        """
        if decision not in VERDICTS or decision == "pending":
            raise ValueError(f"unsupported decision: {decision}")
        from datetime import datetime, timezone

        cursor = self._conn.execute(
            """UPDATE queries SET decision = ?, corrected_slug = ?, reviewer = ?, reviewed_at = ?
               WHERE id = ?""",
            (decision, corrected_slug or None, reviewer or None,
             datetime.now(timezone.utc).isoformat(timespec="seconds"), int(entry_id)),
        )
        self._conn.commit()
        if cursor.rowcount == 0:
            raise ValueError(f"entry not found: {entry_id}")
        log_marker("M-ACTIVE-LEARNING", "review", "DECISION", id=entry_id, decision=decision,
                   corrected=corrected_slug or "-", reviewer=reviewer or "-")
        return True

    def confirmed(self) -> list[dict]:
        """Подтверждённые человеком записи (confirmed и corrected) — источник датасета."""
        rows = self._conn.execute(
            """SELECT * FROM queries WHERE decision IN ('confirmed', 'corrected')
               ORDER BY id ASC""",
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def stats(self) -> ReviewStats:
        """Сводка по хранилищу."""
        stats = ReviewStats()
        for row in self._conn.execute(
            "SELECT decision, COUNT(*) AS n FROM queries GROUP BY decision"
        ).fetchall():
            count = int(row["n"])
            stats.total += count
            if row["decision"] in ReviewStats.__dataclass_fields__:
                setattr(stats, row["decision"], count)
        stats.uncertain = int(
            self._conn.execute("SELECT COUNT(*) AS n FROM queries WHERE uncertain = 1")
            .fetchone()["n"]
        )
        return stats

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        data = dict(row)
        try:
            data["top_k"] = json.loads(data.get("top_k") or "[]")
        except ValueError:
            data["top_k"] = []
        return data


def ingest_jsonl(store: FlywheelStore, path: str | Path) -> int:
    """Загружает JSONL-лог gateway в хранилище.

    Returns:
        Число записей, реально добавленных в БД.
    """
    source = Path(path)
    if not source.is_file():
        return 0
    before = store.stats().total
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            store.record(json.loads(line))
        except ValueError:
            continue
    added = store.stats().total - before
    log_marker("M-ACTIVE-LEARNING", "ingest", "DONE", file=str(source), added=added)
    return added
