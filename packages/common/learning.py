# START_MODULE_CONTRACT
# PURPOSE: Голоса пользователей за вино, правила доверия (анти-накрутка) и очередь модерации
# SCOPE: VOTES, FEEDBACK_REASONS, MIN_TRUSTED_SESSIONS, TRUST_AGREEMENT, RATE_LIMIT_PER_MINUTE,
#        sanitize_feedback, vote_key, dedupe_votes, rate_limit_exceeded, evaluate_trust,
#        append_feedback, load_feedback, feedback_summary, moderation_queue
# DEPENDS: json, datetime, pathlib, packages.common.logging
# LINKS: M-GATEWAY, M-DATA
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# VOTES - допустимые голоса
# FEEDBACK_REASONS - чипы причин: что именно не так
# MIN_TRUSTED_SESSIONS - сколько независимых сессий нужно для доверия
# TRUST_AGREEMENT - какая доля голосов должна совпасть
# RATE_LIMIT_PER_MINUTE - предел голосов с одной сессии в минуту
# DEFAULT_PATH - путь файла голосов по умолчанию
# sanitize_feedback - приводит голос к белому списку полей
# vote_key - ключ дедупликации: сессия и изображение
# dedupe_votes - оставляет по одному голосу на сессию и изображение
# rate_limit_exceeded - проверяет превышение частоты голосов
# evaluate_trust - правила доверия: независимые сессии, согласованность и консенсус агентов
# append_feedback - дописывает голоса в хранилище, best-effort
# load_feedback - читает сохранённые голоса
# feedback_summary - агрегаты для сводки обучения
# moderation_queue - очередь на ручную проверку с провенансом
# END_MODULE_MAP
"""Голоса пользователей и правила доверия.

Главный принцип: одиночный голос никогда не влияет на датасет или каталог. Влияние появляется
только когда несколько независимых сессий согласованно голосуют И это согласуется с консенсусом
агентов (ридеры и каталог из офлайн-разметки). Всё остальное попадает в очередь модератора.

Модуль намеренно не знает ни о сети, ни о gateway: чистая логика плюс JSONL-хранилище, поэтому
правила проверяются юнит-тестами без поднятия сервисов.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from packages.common.logging import log_marker

__all__ = [
    "VOTES",
    "FEEDBACK_REASONS",
    "MIN_TRUSTED_SESSIONS",
    "TRUST_AGREEMENT",
    "RATE_LIMIT_PER_MINUTE",
    "DEFAULT_PATH",
    "sanitize_feedback",
    "vote_key",
    "dedupe_votes",
    "rate_limit_exceeded",
    "evaluate_trust",
    "append_feedback",
    "load_feedback",
    "feedback_summary",
    "moderation_queue",
]

VOTES = ("up", "down")
FEEDBACK_REASONS = ("wrong_wine", "wrong_year", "wrong_grape", "bad_photo", "other")
# Русские подписи чипов: UI присылает либо код, либо подпись — приводим к коду.
REASON_ALIASES = {"не то вино": "wrong_wine", "неверный год": "wrong_year", "не тот сорт": "wrong_grape",
                  "плохое фото": "bad_photo", "другое": "other", "wrong wine": "wrong_wine",
                  "wrong year": "wrong_year", "wrong grape": "wrong_grape", "bad photo": "bad_photo"}
MIN_TRUSTED_SESSIONS = 3
TRUST_AGREEMENT = 0.67
RATE_LIMIT_PER_MINUTE = 6
MAX_REASONS = 3
DEFAULT_PATH = "data/flywheel/feedback.jsonl"


def sanitize_feedback(payload: object, session_id: str = "") -> dict | None:
    """Приводит голос к белому списку полей; возвращает None, если голос недопустим.

    Свободный текст не принимается: причины — только из фиксированного набора чипов.
    """
    if not isinstance(payload, dict):
        return None
    vote = str(payload.get("vote") or "").strip().lower()
    if vote not in VOTES:
        return None
    reasons: list[str] = []
    for item in (payload.get("reason") or payload.get("reasons") or []):
        code = str(item).strip().lower()
        code = REASON_ALIASES.get(code, code)
        if code in FEEDBACK_REASONS and code not in reasons:
            reasons.append(code)
        if len(reasons) >= MAX_REASONS:
            break
    return {
        "vote": vote,
        "reason": reasons,
        "slug": str(payload.get("slug") or "")[:120],
        "image_sha256": str(payload.get("image_sha256") or "")[:64],
        "session_id": str(payload.get("session_id") or session_id or "anonymous")[:40],
        "source": str(payload.get("source") or "user")[:16],
        "timestamp": str(payload.get("timestamp") or datetime.now(timezone.utc).isoformat(timespec="seconds")),
    }


def vote_key(vote: dict) -> tuple[str, str]:
    """Ключ независимости голоса: одна сессия — один голос на изображение.

    Если хеш изображения неизвестен (голос из интерфейса), используется slug: иначе один и тот же
    пользователь мог бы накрутить счётчик повторными нажатиями на том же результате.
    """
    identity = vote.get("image_sha256") or f"slug:{vote.get('slug') or 'unknown'}"
    return str(vote.get("session_id") or "anonymous"), str(identity)


def dedupe_votes(votes: list[dict]) -> list[dict]:
    """Оставляет первый голос каждой сессии на каждое изображение."""
    seen: set[tuple[str, str]] = set()
    result: list[dict] = []
    for vote in votes:
        key = vote_key(vote)
        if key in seen:
            continue
        seen.add(key)
        result.append(vote)
    return result


def rate_limit_exceeded(votes: list[dict], session_id: str, now: datetime | None = None,
                        limit: int = RATE_LIMIT_PER_MINUTE, window_s: float = 60.0) -> bool:
    """Проверяет, превысила ли сессия допустимую частоту голосов."""
    moment = now or datetime.now(timezone.utc)
    recent = 0
    for vote in votes:
        if str(vote.get("session_id")) != session_id:
            continue
        try:
            stamp = datetime.fromisoformat(str(vote.get("timestamp")))
        except (TypeError, ValueError):
            continue
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        if (moment - stamp).total_seconds() <= window_s:
            recent += 1
    return recent >= limit


def evaluate_trust(votes: list[dict], *, consensus: dict | None = None,
                   image_sha256: str = "", slug: str = "") -> dict:
    """Решает, становится ли голос доверенным вкладом.

    Правило влияния: не меньше трёх независимых сессий с согласованным голосом И согласие с
    консенсусом агентов. Одиночный голос, разноголосица или голос против сильного консенсуса
    влияния не получают — они уходят в очередь модератора.
    """
    relevant = [vote for vote in dedupe_votes(votes)
                if (image_sha256 and vote.get("image_sha256") == image_sha256)
                or (slug and vote.get("slug") == slug)
                or (not image_sha256 and not slug)]
    sessions = {str(vote.get("session_id")) for vote in relevant}
    up = sum(1 for vote in relevant if vote.get("vote") == "up")
    down = sum(1 for vote in relevant if vote.get("vote") == "down")
    total = up + down
    majority = "up" if up >= down else "down"
    agreement = round(max(up, down) / total, 4) if total else 0.0
    agent_vote = str((consensus or {}).get("vote") or "")
    agent_agrees = bool(agent_vote) and agent_vote == majority
    reasons: list[str] = []
    if len(sessions) < MIN_TRUSTED_SESSIONS:
        reasons.append(f"независимых сессий {len(sessions)} < {MIN_TRUSTED_SESSIONS}")
    if total and agreement < TRUST_AGREEMENT:
        reasons.append(f"согласие голосов {agreement} < {TRUST_AGREEMENT}")
    if not agent_vote:
        reasons.append("нет консенсуса агентов")
    elif not agent_agrees:
        reasons.append("голос расходится с консенсусом агентов")
    trusted = not reasons
    return {
        "trusted": trusted,
        "applied": trusted,  # применяется только доверенный вклад, одиночный — никогда
        "sessions": len(sessions),
        "up": up,
        "down": down,
        "majority": majority if total else "",
        "agreement": agreement,
        "agent_vote": agent_vote,
        "agent_agrees": agent_agrees,
        "reasons": reasons or ["доверенный вклад: независимые сессии и консенсус агентов"],
        "consensus": {"sessions": len(sessions), "agreement": agreement, "agent_vote": agent_vote},
    }


def append_feedback(votes: list[dict], path: str | Path = DEFAULT_PATH) -> int:
    """Дописывает голоса в хранилище. Best-effort: сбой не влияет на ответ интерфейсу."""
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with target.open("a", encoding="utf-8") as handle:
            for vote in votes:
                handle.write(json.dumps(vote, ensure_ascii=False) + "\n")
                written += 1
        return written
    except Exception as error:  # noqa: BLE001 - хранилище не должно ломать сценарий
        log_marker("M-GATEWAY", "append_feedback", "FAILED", error=type(error).__name__)
        return 0


def load_feedback(path: str | Path = DEFAULT_PATH) -> list[dict]:
    """Читает сохранённые голоса, пропуская повреждённые строки."""
    target = Path(path)
    if not target.is_file():
        return []
    votes: list[dict] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            votes.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return votes


def _consensus_by_target(votes: list[dict]) -> dict[str, dict]:
    """Считает консенсус агентов по каждому изображению: голос большинства среди источников агентов."""
    grouped: dict[str, dict[str, int]] = {}
    for vote in votes:
        if vote.get("source") not in {"agent", "moderator"}:
            continue
        key = vote.get("image_sha256") or f"slug:{vote.get('slug')}"
        bucket = grouped.setdefault(key, {"up": 0, "down": 0})
        bucket[str(vote.get("vote"))] = bucket.get(str(vote.get("vote")), 0) + 1
    return {key: {"vote": ("up" if counts["up"] >= counts["down"] else "down"),
                  "agent_votes": counts["up"] + counts["down"]}
            for key, counts in grouped.items()}


def feedback_summary(votes: list[dict], consensus_resolver=None) -> dict:
    """Агрегаты по голосам для сводки обучения.

    Если передан источник консенсуса (например каталог через gateway), используется он:
    иначе сводка и очередь расходились бы с решением эндпоинта приёма голоса.
    """
    unique = dedupe_votes(votes)
    up = sum(1 for vote in unique if vote.get("vote") == "up")
    down = sum(1 for vote in unique if vote.get("vote") == "down")
    reasons: dict[str, int] = {}
    for vote in unique:
        for reason in vote.get("reason") or []:
            reasons[reason] = reasons.get(reason, 0) + 1
    sessions = {str(vote.get("session_id")) for vote in unique}
    consensus = _consensus_by_target(unique)
    targets = {vote.get("image_sha256") or f"slug:{vote.get('slug')}" for vote in unique}
    trusted = queued = 0
    for target in targets:
        sample = next(vote for vote in unique
                      if (vote.get("image_sha256") or f"slug:{vote.get('slug')}") == target)
        resolved = None
        if consensus_resolver is not None:
            try:
                resolved = consensus_resolver(sample.get("slug") or "")
            except Exception:  # noqa: BLE001 - без источника консенсуса используем голоса агентов
                resolved = None
        verdict = evaluate_trust(unique, consensus=resolved or consensus.get(target),
                                 image_sha256=sample.get("image_sha256") or "",
                                 slug=sample.get("slug") or "")
        trusted += 1 if verdict["trusted"] else 0
        queued += 0 if verdict["trusted"] else 1
    return {
        "votes": len(votes),
        "unique_votes": len(unique),
        "sessions": len(sessions),
        "up": up,
        "down": down,
        "reasons": dict(sorted(reasons.items(), key=lambda item: -item[1])),
        "targets": len(targets),
        "trusted_targets": trusted,
        "queued_targets": queued,
        "consensus": consensus,
        "rules": {"min_sessions": MIN_TRUSTED_SESSIONS, "agreement": TRUST_AGREEMENT,
                  "rate_limit_per_minute": RATE_LIMIT_PER_MINUTE,
                  "note": "одиночный голос никогда не влияет; влияние только при согласии независимых сессий и консенсуса агентов"},
    }


def moderation_queue(votes: list[dict], limit: int = 50, consensus_resolver=None) -> list[dict]:
    """Очередь на ручную проверку: всё, что не получило доверия, с провенансом."""
    unique = dedupe_votes(votes)
    consensus = _consensus_by_target(unique)
    queued: list[dict] = []
    seen: set[str] = set()
    for vote in unique:
        target = vote.get("image_sha256") or f"slug:{vote.get('slug')}"
        if target in seen:
            continue
        seen.add(target)
        resolved = None
        if consensus_resolver is not None:
            try:
                resolved = consensus_resolver(vote.get("slug") or "")
            except Exception:  # noqa: BLE001
                resolved = None
        verdict = evaluate_trust(unique, consensus=resolved or consensus.get(target),
                                 image_sha256=vote.get("image_sha256") or "",
                                 slug=vote.get("slug") or "")
        if verdict["trusted"]:
            continue
        queued.append({"image_sha256": vote.get("image_sha256", ""), "slug": vote.get("slug", ""),
                       "sessions": verdict["sessions"], "up": verdict["up"], "down": verdict["down"],
                       "agreement": verdict["agreement"], "agent_vote": verdict["agent_vote"],
                       "needs_moderator": True,
                       "provenance": [{"session_id": item.get("session_id"),
                                       "vote": item.get("vote"), "reason": item.get("reason"),
                                       "source": item.get("source"), "timestamp": item.get("timestamp")}
                                      for item in unique
                                      if (item.get("image_sha256") or f"slug:{item.get('slug')}") == target]})
    return queued[:limit]
