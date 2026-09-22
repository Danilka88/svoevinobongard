# START_MODULE_CONTRACT
# PURPOSE: Обезличенная телеметрия интерфейса: санитайз событий, запись в ротируемый JSONL и сводка агрегатов
# SCOPE: ALLOWED_TYPES, ALLOWED_FIELDS, sanitize_event, append_events, load_events, telemetry_summary
# DEPENDS: json, datetime, pathlib, collections
# LINKS: M-GATEWAY, M-DATA
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# ALLOWED_TYPES - какие типы событий принимаются
# ALLOWED_FIELDS - белый список полей события
# MAX_EVENTS_PER_BATCH - предел числа событий в батче
# MAX_STRING - предел длины строкового значения
# MAX_FILE_BYTES - порог ротации файла событий
# sanitize_event - приводит событие к белому списку и обрезает строки
# append_events - дописывает батч в JSONL с ротацией, best-effort
# load_events - читает сохранённые события
# telemetry_summary - агрегаты для самостоятельного разбора после сессии
# END_MODULE_MAP
"""Обезличенная телеметрия интерфейса.

Приватность по умолчанию: принимаются только структурные поля из белого списка, любые другие
ключи отбрасываются, строки обрезаются по длине. Изображения, свободный текст и персональные
данные не принимаются и не сохраняются — идентификатор сессии случаен и с человеком не связан.

Запись best-effort: сбой телеметрии не должен влиять ни на интерфейс, ни на API-ответ.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "ALLOWED_TYPES",
    "ALLOWED_FIELDS",
    "MAX_EVENTS_PER_BATCH",
    "MAX_STRING",
    "MAX_FILE_BYTES",
    "sanitize_event",
    "append_events",
    "load_events",
    "telemetry_summary",
]

ALLOWED_TYPES = ("page_view", "action", "scan", "sommelier", "catalog", "theme", "api_error", "js_error")
ALLOWED_FIELDS = ("type", "route", "action", "value", "ok", "latency_ms", "detail", "seq")
MAX_EVENTS_PER_BATCH = 200
MAX_STRING = 64
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_ROUTE = 120


def _short(value: object, limit: int = MAX_STRING) -> str:
    """Обрезает строковое значение: длинные строки не должны попадать в лог."""
    text = str(value).strip().replace("\n", " ").replace("\r", " ")
    return text[:limit]


def sanitize_event(event: object) -> dict | None:
    """Приводит событие к белому списку полей.

    Возвращает None, если событие не является объектом или тип не поддерживается: так в лог не
    попадают ни произвольные структуры, ни неизвестные действия.
    """
    if not isinstance(event, dict):
        return None
    kind = _short(event.get("type"))
    if kind not in ALLOWED_TYPES:
        return None
    clean: dict = {"type": kind}
    for field in ("route", "action", "value", "detail"):
        if field in event and event[field] not in (None, ""):
            limit = MAX_ROUTE if field == "route" else MAX_STRING
            text = _short(event[field], limit)
            if text:
                clean[field] = text
    if "ok" in event:
        clean["ok"] = bool(event["ok"])
    if "latency_ms" in event:
        try:
            latency = float(event["latency_ms"])
        except (TypeError, ValueError):
            latency = -1.0
        if latency >= 0:
            clean["latency_ms"] = round(latency, 1)
    if "seq" in event:
        try:
            clean["seq"] = int(event["seq"])
        except (TypeError, ValueError):
            pass
    return clean


def append_events(events: list[dict], session_id: str, path: str | Path) -> int:
    """Дописывает обезличенные события в JSONL, ротируя файл по размеру.

    Best-effort: при любой ошибке возвращается 0, исключение не поднимается — телеметрия не имеет
    права ломать основной сценарий.
    """
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and target.stat().st_size > MAX_FILE_BYTES:
            rotated = target.with_suffix(target.suffix + ".1")
            rotated.unlink(missing_ok=True)
            target.rename(rotated)
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        written = 0
        with target.open("a", encoding="utf-8") as handle:
            for event in events:
                clean = sanitize_event(event)
                if clean is None:
                    continue
                clean.update({"session_id": _short(session_id, 40), "ts": stamp,
                              "recv": round(datetime.now().timestamp(), 3)})
                handle.write(json.dumps(clean, ensure_ascii=False) + "\n")
                written += 1
        return written
    except Exception:  # noqa: BLE001 - телеметрия не влияет на основной сценарий
        return 0


def load_events(path: str | Path, limit: int = 0) -> list[dict]:
    """Читает сохранённые события, пропуская повреждённые строки."""
    target = Path(path)
    if not target.is_file():
        return []
    events: list[dict] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events[-limit:] if limit else events


def _percentile(values: list[float], share: float) -> float:
    """Считает перцентиль по уже отсортированному списку."""
    if not values:
        return 0.0
    return round(values[min(len(values) - 1, int(share * len(values)))], 1)


def telemetry_summary(path: str | Path, last_errors: int = 20) -> dict:
    """Собирает агрегаты по сохранённым событиям для разбора после сессии."""
    events = load_events(path)
    by_route: Counter = Counter()
    by_action: Counter = Counter()
    by_type: Counter = Counter()
    sessions: set[str] = set()
    latencies: list[float] = []
    errors: list[dict] = []
    scans = not_found = sommelier_steps = catalog_searches = theme_switches = 0
    for event in events:
        kind = str(event.get("type") or "")
        by_type[kind] += 1
        if event.get("session_id"):
            sessions.add(str(event["session_id"]))
        if event.get("route"):
            by_route[str(event["route"])] += 1
        if event.get("action"):
            by_action[str(event["action"])] += 1
        if isinstance(event.get("latency_ms"), (int, float)):
            latencies.append(float(event["latency_ms"]))
        if kind == "scan":
            scans += 1
            if event.get("ok") is False:
                not_found += 1
        elif kind == "sommelier":
            sommelier_steps += 1
        elif kind == "catalog":
            catalog_searches += 1
        elif kind == "theme":
            theme_switches += 1
        if kind in {"js_error", "api_error"}:
            errors.append({"ts": event.get("ts", ""), "type": kind, "route": event.get("route", ""),
                           "action": event.get("action", ""), "detail": event.get("detail", "")})
    latencies.sort()
    return {
        "events": len(events),
        "sessions": len(sessions),
        "by_type": dict(by_type.most_common()),
        "by_route": dict(by_route.most_common(20)),
        "by_action": dict(by_action.most_common(20)),
        "errors": {kind: by_type.get(kind, 0) for kind in ("js_error", "api_error")},
        "scenarios": {"scans": scans, "scans_not_found": not_found,
                      "sommelier_steps": sommelier_steps, "catalog_searches": catalog_searches,
                      "theme_switches": theme_switches},
        "latency": {"p50_ms": _percentile(latencies, 0.5), "p95_ms": _percentile(latencies, 0.95),
                    "samples": len(latencies)},
        "last_errors": errors[-last_errors:],
        "note": "обезличенная телеметрия интерфейса: без изображений, свободного текста и персональных данных",
    }
