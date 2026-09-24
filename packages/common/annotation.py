# START_MODULE_CONTRACT
# PURPOSE: Консенсус показаний нескольких ридеров этикетки и политика разбиения датасета
# SCOPE: READER_WEIGHTS, ACCEPT_AGREEMENT, ESCALATE_AGREEMENT, KEY_FIELDS, cluster_values,
#        field_consensus, adjudicate, judge_payload, assign_split, split_policy
# DEPENDS: packages.common.text_norm
# LINKS: M-EVAL, M-DATA
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# READER_WEIGHTS - веса ридеров: сильная модель важнее, затем OCR, qwen3-vl и слабая модель
# ACCEPT_AGREEMENT - порог согласия для принятия метки
# ESCALATE_AGREEMENT - порог, ниже которого случай уходит на ручную проверку
# KEY_FIELDS - поля, по которым считается согласие ридеров
# cluster_values - группирует показания ридеров по одному значению
# field_consensus - взвешенное голосование по одному полю
# adjudicate - вердикт по всем полям: принять или эскалировать, с обоснованием
# judge_payload - представление вердикта в форме схемы судьи
# assign_split - разбиение записи на train, eval или holdout
# split_policy - текстовое описание политики разбиения
# END_MODULE_MAP
"""Консенсус нескольких ридеров и защита датасета от утечки.

Модуль намеренно не знает ни о моделях, ни о сети: на вход идут показания ридеров и кандидаты
каталога, на выходе — решение и обоснование. Это позволяет проверять логику юнит-тестами без
запуска моделей и переиспользовать её в офлайн-скрипте.

Главный принцип: принимается только то, в чём ридеры согласны и что подтверждает каталог. Любое
расхождение, ничья или противоречие каталогу уходит на ручную проверку — лучше меньше меток, чем
ошибочные.
"""
from __future__ import annotations

from packages.common.text_norm import fold, similarity

__all__ = [
    "READER_WEIGHTS",
    "ACCEPT_AGREEMENT",
    "ESCALATE_AGREEMENT",
    "KEY_FIELDS",
    "cluster_values",
    "field_consensus",
    "adjudicate",
    "judge_payload",
    "assign_split",
    "split_policy",
]

# Сильная модель весит больше слабой: декоррелированные ошибки полезны, но разная точность важна.
# C-060: добавлен четвёртый ридер (qwen3-vl:2b) — вес меньше сильной модели, но выше слабой.
READER_WEIGHTS = {"vlm_primary": 0.4, "ocr_fusion": 0.25, "vlm_qwen": 0.2, "vlm_small": 0.15}
ACCEPT_AGREEMENT = 0.75
ESCALATE_AGREEMENT = 0.6
KEY_FIELDS = ("winery", "wine_name", "vintage", "grape", "color", "sweetness")
SAME_VALUE = 0.86
CATALOG_SUPPORT = 0.86


def cluster_values(values: list[tuple[str, str]]) -> list[dict]:
    """Группирует показания (ридер, значение) в кластеры похожих значений.

    Сравнение идёт через нормализацию: «Пино Нуар» и «пино-нуар» — одно значение, «Массандра» и
    «МАССАНДРА» — тоже. Пустые значения игнорируются: отсутствие показания это не голос.
    """
    clusters: list[dict] = []
    for reader, raw in values:
        text = str(raw or "").strip()
        if not text:
            continue
        for cluster in clusters:
            if similarity(text, cluster["value"]) >= SAME_VALUE or fold(text) == fold(cluster["value"]):
                cluster["readers"].append(reader)
                break
        else:
            clusters.append({"value": text, "readers": [reader]})
    return clusters


def field_consensus(readings: dict, field: str, weights: dict | None = None) -> dict:
    """Взвешенное голосование по одному полю.

    Returns:
        Словарь с победившим значением, его весом, полным весом голосов и признаком ничьей.
    """
    active = weights or READER_WEIGHTS
    votes = [(reader, (payload or {}).get(field))
             for reader, payload in readings.items() if reader in active]
    clusters = cluster_values([(reader, value) for reader, value in votes if value not in (None, "")])
    if not clusters:
        return {"field": field, "value": "", "weight": 0.0, "total": 0.0, "tie": False, "clusters": []}
    for cluster in clusters:
        cluster["weight"] = round(sum(active.get(reader, 0.0) for reader in cluster["readers"]), 4)
    clusters.sort(key=lambda item: (-item["weight"], -len(item["readers"])))
    total = round(sum(cluster["weight"] for cluster in clusters), 4)
    best = clusters[0]
    tie = len(clusters) > 1 and abs(clusters[1]["weight"] - best["weight"]) < 1e-9
    return {"field": field, "value": best["value"], "weight": best["weight"], "total": total,
            "tie": tie, "clusters": clusters, "readers": best["readers"]}


def _catalog_support(value: str, candidates: list[dict], field: str) -> float:
    """Насколько значение подтверждается каталогом: максимум похожести по полю кандидата."""
    if not value or not candidates:
        return 0.0
    best = 0.0
    for candidate in candidates:
        record = candidate.get("record") or candidate
        for key in (field, "name", "winery", "grape"):
            target = record.get(key)
            if target:
                best = max(best, similarity(value, str(target)))
    return round(best, 4)


def adjudicate(readings: dict, candidates: list[dict] | None = None,
               weights: dict | None = None) -> dict:
    """Выносит вердикт по показаниям ридеров и кандидатам каталога.

    Логика намеренно скептическая: сначала ищутся основания НЕ согласиться (ничья, расхождение,
    отсутствие поддержки каталога, конфликт винодельни и названия), и только при отсутствии таких
    оснований метка принимается.

    Returns:
        Словарь: verdict (accept или escalate), label, confidence, agreement, reasons, fields.
    """
    active = weights or READER_WEIGHTS
    candidates = candidates or []
    fields = {field: field_consensus(readings, field, active) for field in KEY_FIELDS}
    voted = {field: result for field, result in fields.items() if result["value"]}
    reasons: list[str] = []

    if not voted:
        return {"verdict": "escalate", "label": {}, "confidence": 0.0, "agreement": 0.0,
                "reasons": ["ни один ридер не дал полей"], "fields": fields}

    agreement = round(sum(result["weight"] for result in voted.values())
                      / max(sum(result["total"] for result in voted.values()), 1e-9), 4)
    ties = [field for field, result in voted.items() if result["tie"]]
    if ties:
        reasons.append(f"ничья по полям: {', '.join(ties)}")

    support = {field: _catalog_support(result["value"], candidates, field) for field, result in voted.items()}
    identity_support = max(support.get("winery", 0.0), support.get("wine_name", 0.0),
                           support.get("grape", 0.0))
    if identity_support < CATALOG_SUPPORT:
        reasons.append("каталог не подтверждает ни винодельню, ни название, ни сорт")

    # Конфликт идентичности: ридеры согласны по винодельне, но название указывает на другого
    # производителя — это признак того, что этикетка прочитана неверно.
    conflict = False
    name = voted.get("wine_name")
    if name and candidates:
        # Название вида «Мускатель Массандра» содержит имя винодельни и это нормально. Конфликтом
        # считается только случай, когда название ближе к винодельне каталога, чем к названию вина:
        # значит модель прочитала винодельню вместо названия.
        # Сравнение идёт на точное совпадение после нормализации: частичное совпадение здесь не
        # годится, потому что названия вида «Мускатель Массандра» содержат имя винодельни и это
        # нормально. Конфликт — только когда название И ЕСТЬ винодельня.
        folded_name = fold(name["value"])
        conflict = any(folded_name == fold(str((item.get("record") or item).get("winery") or ""))
                       for item in candidates
                       if (item.get("record") or item).get("winery"))
    if conflict:
        reasons.append("название указывает на другую винодельню")

    label = {field: result["value"] for field, result in voted.items()}
    # Уверенность — это согласие, взвешенное поддержкой каталога: «голый» ярлык не выдаём.
    confidence = round(min(agreement, max(identity_support, 0.5)) * 1.0, 4)
    if agreement < ESCALATE_AGREEMENT:
        reasons.append(f"согласие ниже порога ({agreement} < {ESCALATE_AGREEMENT})")
    if not reasons and agreement >= ACCEPT_AGREEMENT and identity_support >= CATALOG_SUPPORT:
        return {"verdict": "accept", "label": label, "confidence": confidence, "agreement": agreement,
                "reasons": [f"ридеры согласны, каталог подтверждает ({identity_support})"],
                "fields": fields, "support": support}
    if not reasons:
        reasons.append(f"согласие {agreement} ниже порога принятия {ACCEPT_AGREEMENT}")
    return {"verdict": "escalate", "label": label, "confidence": confidence, "agreement": agreement,
            "reasons": reasons, "fields": fields, "support": support}


def judge_payload(decision: dict) -> dict:
    """Представляет вердикт в форме схемы судьи (verdict, score, reasons)."""
    verdict = {"accept": "pass", "escalate": "uncertain"}.get(decision.get("verdict", ""), "uncertain")
    if decision.get("verdict") == "escalate" and any("другую винодельню" in reason
                                                     for reason in decision.get("reasons") or []):
        verdict = "fail"
    return {"verdict": verdict, "score": float(decision.get("confidence") or 0.0),
            "reasons": [str(reason)[:200] for reason in (decision.get("reasons") or [])][:5]}


def assign_split(origin: str, escalated: bool, base_split: str = "") -> str:
    """Определяет разбиение записи: train, eval или holdout.

    Политика защищает от утечки: спорные случаи и конфликты никогда не попадают в обучение,
    а производные (аугментации) наследуют разбиение источника, иначе копии оценочных фото
    просочились бы в обучение.
    """
    if escalated:
        return "holdout"
    if base_split:  # производное наследует разбиение источника целиком
        return base_split
    if origin in {"eval", "real_eval"}:
        return "eval"
    return "train" if origin in {"train", "catalog"} else "eval"


def split_policy() -> str:
    """Текстовое описание политики разбиения для отчёта и документации."""
    return ("Спорные случаи и конфликты — holdout и никогда train. Производные (аугментации) "
            "наследуют разбиение источника: аугментация оценочного фото остаётся оценочной, "
            "поэтому копии не попадают в обучение. Реальные фото оценочного набора помечаются "
            "eval независимо от уверенности консенсуса.")
