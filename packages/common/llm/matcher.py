# START_MODULE_CONTRACT
# PURPOSE: LLM-сопоставление наблюдения этикетки с каталогом: выбор лучшего slug, вероятность и честный отказ
# SCOPE: MATCHER_INSTRUCTION, build_candidates_text, build_matcher_prompt, parse_match, match_candidates
# DEPENDS: json, packages.common.llm.router, packages.common.llm.structured
# LINKS: M-COMMON, M-VLM, M-GATEWAY
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# MATCHER_INSTRUCTION - инструкция матчера: выбрать вино из списка кандидатов или честно отказаться
# build_candidates_text - компактный нумерованный список кандидатов каталога для промпта
# build_matcher_prompt - полный промпт: наблюдение + кандидаты + требование строгого JSON
# parse_match - разбор ответа матчера по схеме с проверкой, что slug взят из списка
# match_candidates - вызов LLM-матчера с деградацией: slug None вместо выдумки
# _native_ollama_chat - нативный вызов Ollama /api/chat с think=false и format=json (обход reasoning)
# END_MODULE_MAP
"""LLM-сопоставление наблюдения с каталогом (C-074).

Классический текстовый поиск даёт пул кандидатов, но на разговорном чтении и неполных полях
финальный выбор ненадёжен. Матчер просит LLM сверить прочитанное с конкретным списком вин и
вернуть либо slug одного из них с вероятностью, либо честный отказ. Slug, которого нет в списке,
не принимается: модель не может «выдумать» позицию каталога.
"""
from __future__ import annotations

import json

from packages.common.llm.router import get_router
from packages.common.llm.structured import parse_structured

__all__ = ["MATCHER_INSTRUCTION", "build_candidates_text", "build_matcher_prompt", "match_candidates",
           "parse_match", "_native_ollama_chat"]

MATCHER_INSTRUCTION = (
    "Ты сопоставляешь этикетку вина, прочитанную с фото, со списком кандидатов каталога. "
    "Выбери ровно одного кандидата, который обозначает ТО ЖЕ вино (винодельня и название совпадают "
    "с учётом транслитерации и опечаток чтения). Если ни один кандидат не подходит — честно откажись. "
    "Строго запрещено выдумывать slug: можно вернуть только slug из списка или null. "
    "Ответ строго одним JSON-объектом без пояснений вокруг:\n"
    '{"slug": "<slug из списка или null>", "p_correct": <0..1>, "in_catalog": <true|false>, '
    '"reason": "<кратко почему>"}'
)

# Поля наблюдения, которые попадают в промпт: порядок важен для читаемости, пустые пропускаются.
_OBSERVATION_FIELDS = ("winery", "wine_name", "vintage", "grape", "sweetness", "color", "wine_type",
                       "region", "raw_text")


def _clean(value: object) -> str:
    """Однострочное текстовое значение для промпта."""
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False)
    return " ".join(str(value or "").split()).strip()[:200]


def build_candidates_text(candidates: list[dict], *, limit: int = 20) -> str:
    """Нумерованный список кандидатов: slug, название, винодельня, год, сорт, сладость."""
    lines: list[str] = []
    for index, item in enumerate(candidates[:max(1, limit)], start=1):
        slug = _clean(item.get("slug"))
        if not slug:
            continue
        parts = [_clean(item.get("name")) or "—"]
        for key, label in (("winery", "винодельня"), ("year", "год"), ("grape", "сорт"),
                           ("sweetness", "сладость")):
            value = _clean(item.get(key))
            if value:
                parts.append(f"{label}: {value}")
        lines.append(f"{index}. {slug} — " + "; ".join(parts))
    return "\n".join(lines)


def build_matcher_prompt(observation: dict, candidates: list[dict], *, limit: int = 20) -> str:
    """Полный промпт матчера: инструкция, наблюдение и список кандидатов."""
    observed = "; ".join(f"{name}: {_clean(observation.get(name))}"
                         for name in _OBSERVATION_FIELDS if _clean(observation.get(name)))
    if not observed:
        observed = "— (ничего не прочитано)"
    return (f"{MATCHER_INSTRUCTION}\n\n"
            f"Прочитано с этикетки:\n{observed}\n\n"
            f"Кандидаты каталога:\n{build_candidates_text(candidates, limit=limit)}\n")


def parse_match(text: str | None, candidates: list[dict]) -> dict:
    """Разбирает ответ матчера и принимает только slug из списка кандидатов.

    Выдуманный или отсутствующий slug сводится к отказу (slug=None), чтобы решение оставалось
    в границах каталога. Возвращает нормализованный результат без обращения к сети.
    """
    parsed = parse_structured(str(text or ""), "matcher")
    data = parsed.get("data") if parsed.get("valid") else {}
    data = data or {}
    allowed = {_clean(item.get("slug")) for item in candidates if item.get("slug")}
    slug = _clean(data.get("slug"))
    if not slug or slug not in allowed:
        slug = None
    p_correct = data.get("p_correct")
    try:
        p_correct = None if p_correct is None else max(0.0, min(1.0, float(p_correct)))
    except (TypeError, ValueError):
        p_correct = None
    in_catalog = data.get("in_catalog")
    if slug is None:
        in_catalog = False
    elif in_catalog is None:
        in_catalog = True
    return {"slug": slug, "p_correct": p_correct, "in_catalog": bool(in_catalog),
            "reason": _clean(data.get("reason")), "source": "llm" if parsed.get("valid") else "degraded"}


def _native_ollama_chat(prompt: str, model: str, *, max_tokens: int, timeout_s: float) -> str | None:
    """Нативный вызов Ollama /api/chat с think=false и format=json.

    OpenAI-совместимый путь не даёт отключить рассуждения: reasoning-модели (gemma4) тратят весь
    бюджет на внутренний монолог и возвращают пустой content при finish_reason=length. Нативный API
    принимает think=false и format=json, поэтому используется как основной путь для локального Ollama.
    Возвращает текст ответа или None, если вызов не удался.
    """
    import urllib.request

    from packages.common.config import settings

    base = str(getattr(settings, "ollama_base_url", "") or "http://127.0.0.1:11434").rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False,
               "think": False, "format": "json",
               "options": {"num_predict": max_tokens, "temperature": 0.0}}
    try:
        request = urllib.request.Request(f"{base}/api/chat", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
            data = json.loads(response.read().decode())
    except Exception as error:  # noqa: BLE001 - нативный путь не должен ронять матчер
        from packages.common.logging import log_marker

        log_marker("M-COMMON", "matcher_native", "FAILED", error=type(error).__name__)
        return None
    return str((data.get("message") or {}).get("content") or "").strip()


def match_candidates(observation: dict, candidates: list[dict], *, router=None,
                     role: str = "text_model", model: str | None = None, max_tokens: int = 256,
                     timeout_s: float | None = None, limit: int = 20) -> dict:
    """Просит LLM выбрать лучшее совпадение среди кандидатов каталога.

    Args:
        observation: поля, прочитанные с этикетки (winery, wine_name, raw_text, ...).
        candidates: кандидаты каталога (slug, name, winery, year, grape, sweetness).
        router: LLM-роутер; по умолчанию singleton из env. В тестах подменяется.
        role: роль маршрутизации (по умолчанию текстовая модель).
        model: явная модель, переопределяет первую ступень цепочки. Для локального Ollama-имени
            используется нативный путь (/api/chat, think=false), иначе — совместимый router.chat.
        max_tokens: лимит генерации (ответ короткий).
        timeout_s: жёсткий таймаут вызова.
        limit: максимум кандидатов в промпте.

    Returns:
        Словарь: slug (или None), p_correct, in_catalog, reason, source. При недоступной LLM —
        slug=None, source='degraded', без исключения наружу.
    """
    if not candidates:
        return {"slug": None, "p_correct": None, "in_catalog": False, "reason": "нет кандидатов",
                "source": "degraded"}
    prompt = build_matcher_prompt(observation, candidates, limit=limit)
    # Локальное Ollama-имя (без слэша провайдера) идёт нативным путём: только он отключает reasoning.
    if router is None and model and "/" not in str(model):
        text = _native_ollama_chat(prompt, str(model), max_tokens=max_tokens,
                                   timeout_s=timeout_s or 60.0)
        if text is None:
            return {"slug": None, "p_correct": None, "in_catalog": False,
                    "reason": "native ollama unavailable", "source": "degraded"}
        outcome = parse_match(text, candidates)
        outcome["model"] = str(model)
        outcome["provider"] = "ollama-native"
        return outcome
    active = router or get_router()
    try:
        result = active.chat([{"role": "user", "content": prompt}], role=role, model=model,
                             max_tokens=max_tokens, temperature=0.0,
                             response_format={"type": "json_object"}, extra={"think": False},
                             timeout_s=timeout_s)
    except Exception as exc:  # noqa: BLE001 - матчер не должен ронять пайплайн
        return {"slug": None, "p_correct": None, "in_catalog": False,
                "reason": f"llm unavailable: {type(exc).__name__}", "source": "degraded"}
    outcome = parse_match(getattr(result, "text", ""), candidates)
    outcome["model"] = getattr(result, "model", "") or ""
    outcome["provider"] = getattr(result, "provider", "") or ""
    return outcome
