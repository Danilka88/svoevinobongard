#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Сборка подтверждённых пар «фото → slug» из независимых источников с разбиением fit/holdout
# SCOPE: DEFAULT_GOLD, DEFAULT_FEEDBACK, DEFAULT_CONSENSUS, DEFAULT_CATALOG, DEFAULT_OUT, DEFAULT_REPORT,
#        DEFAULT_MD, HOLDOUT_MOD, load_gold_pairs, load_feedback_pairs, load_consensus_labels,
#        assign_split, build_pairs, write_dataset, coverage_report, build_markdown, main
# DEPENDS: argparse, hashlib, json, sys, pathlib, packages.common.catalog
# LINKS: M-EVAL, M-ACTIVE-LEARNING
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# BASE - корень репозитория
# DEFAULT_GOLD - эталон field21_gold (подтверждённые slug)
# DEFAULT_FEEDBACK - flywheel-фидбек (vote=up)
# DEFAULT_CONSENSUS - консенсусные метки real100
# DEFAULT_CATALOG - catalog.jsonl (метаданные и покрытие)
# DEFAULT_OUT - датасет подтверждённых пар
# DEFAULT_REPORT - JSON-отчёт покрытия
# DEFAULT_MD - Markdown-отчёт
# HOLDOUT_MOD - каждый N-й фото уходит в holdout (детерминированное разбиение по хэшу)
# load_gold_pairs - подтверждённые slug-пары из эталона (matched)
# load_feedback_pairs - подтверждённые slug-пары из обратной связи (vote=up)
# load_consensus_labels - консенсусные метки real100 (текстовые, без slug)
# assign_split - детерминированное разбиение fit/holdout по хэшу фото
# build_pairs - объединение источников в единый список пар
# write_dataset - запись обучаемых пар без тестовых фото (инвариант утечки)
# coverage_report - сводка покрытия каталога и источников
# build_markdown - Markdown-отчёт
# main - CLI сборки датасета и отчётов
# END_MODULE_MAP
"""Сборка подтверждённых пар для дообучения.

Источники (только подтверждённые, независимые от теста конкурса):
- ``reports/field21_gold.json`` — 21 полевое фото владельца с ручной меткой (только тест, в обучение
  не идёт как изображение; здесь используется как источник подтверждённых slug-пар с разбиением split);
- ``data/flywheel/feedback.jsonl`` — голоса пользователей ``vote=up`` (подтверждённый slug);
- ``reports/real100_consensus.json`` — консенсусные метки 100 реальных фото (текстовые поля; slug
  проставляется только при однозначном совпадении с каталогом).

Защита от утечки: пары делятся на ``fit`` и ``holdout`` детерминированно по хэшу фото, поэтому
отложенный замер не пересекается с обучающими примерами. Обучение на собранном объёме не запускается:
скрипт фиксирует готовность и честное покрытие (для дообучения нужны сотни подтверждённых пар).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
DEFAULT_GOLD = BASE / "reports" / "field21_gold.json"
DEFAULT_FEEDBACK = BASE / "data" / "flywheel" / "feedback.jsonl"
DEFAULT_CONSENSUS = BASE / "reports" / "real100_consensus.json"
DEFAULT_CATALOG = BASE / "data" / "catalog" / "catalog.jsonl"
DEFAULT_OUT = BASE / "data" / "train-confirmed" / "pairs_confirmed.jsonl"
DEFAULT_REPORT = BASE / "reports" / "confirmed_pairs.json"
DEFAULT_MD = BASE / "reports" / "confirmed_pairs.md"
HOLDOUT_MOD = 5

__all__ = ["DEFAULT_CATALOG", "DEFAULT_CONSENSUS", "DEFAULT_FEEDBACK", "DEFAULT_GOLD", "DEFAULT_MD",
           "DEFAULT_OUT", "DEFAULT_REPORT", "HOLDOUT_MOD", "assign_split", "build_markdown",
           "build_pairs", "coverage_report", "load_consensus_labels", "write_dataset", "load_feedback_pairs",
           "load_gold_pairs", "main", "write_dataset"]


def _clean(value: object) -> str:
    """Однострочное значение."""
    return " ".join(str(value or "").split()).strip()


def assign_split(key: str, *, holdout_mod: int = HOLDOUT_MOD) -> str:
    """Детерминированное разбиение fit/holdout по хэшу ключа.

    Каждый N-й ключ (по стабильному хэшу) уходит в holdout, поэтому отложенный замер не пересекается
    с обучающими примерами и разбиение воспроизводимо между запусками.
    """
    digest = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
    return "holdout" if int(digest, 16) % max(1, holdout_mod) == 0 else "fit"


def load_gold_pairs(gold_path: Path) -> list[dict]:
    """Подтверждённые slug-пары из эталона: только однозначные (matched).

    sha256 тестового фото НЕ переносится в датасет: иначе хэш теста попал бы в защищённую поверхность
    ``data/train-confirmed`` (инвариант утечки C-072). Хранится только имя файла и slug.
    """
    if not gold_path.is_file():
        return []
    payload = json.loads(gold_path.read_text(encoding="utf-8"))
    pairs: list[dict] = []
    for name, record in (payload.get("files") or {}).items():
        if (record or {}).get("status") != "matched":
            continue
        slug = _clean((record or {}).get("expected_slug"))
        if not slug:
            continue
        pairs.append({"file": name, "slug": slug, "source": "gold_matched"})
    return pairs


def load_feedback_pairs(feedback_path: Path) -> list[dict]:
    """Подтверждённые slug-пары из обратной связи (vote=up)."""
    if not feedback_path.is_file():
        return []
    pairs: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for line in feedback_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            vote = json.loads(line)
        except ValueError:
            continue
        if str((vote or {}).get("vote")) != "up":
            continue
        slug = _clean((vote or {}).get("slug"))
        sha = _clean((vote or {}).get("image_sha256"))
        if not slug or (sha, slug) in seen:
            continue
        seen.add((sha, slug))
        pairs.append({"file": sha or slug, "slug": slug, "source": "feedback_up", "sha256": sha})
    return pairs


def load_consensus_labels(consensus_path: Path, catalog: dict) -> list[dict]:
    """Консенсусные метки real100: slug проставляется при однозначном совпадении по названию/винодельне."""
    if not consensus_path.is_file():
        return []
    payload = json.loads(consensus_path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for item in payload.get("accepted_labels") or []:
        label = (item or {}).get("label") or {}
        winery = _clean(label.get("winery"))
        name = _clean(label.get("wine_name"))
        slug = _match_catalog_slug(catalog, winery, name)
        rows.append({"file": _clean(item.get("file")), "slug": slug, "source": "real100_consensus",
                     "winery": winery, "name": name, "confidence": item.get("confidence")})
    return rows


def _match_catalog_slug(catalog: dict, winery: str, name: str) -> str | None:
    """Ищет однозначный slug каталога по винодельне и названию (без fuzzy: только явное вхождение)."""
    from packages.common.text_norm import fold

    if not winery or not name:
        return None
    folded_winery, folded_name = fold(winery), fold(name)
    matches = [slug for slug, record in (catalog or {}).items()
               if folded_winery and folded_winery in fold(_clean((record or {}).get("winery")))
               and folded_name and folded_name in fold(_clean((record or {}).get("name")))]
    return matches[0] if len(matches) == 1 else None


def build_pairs(gold: list[dict], feedback: list[dict], consensus: list[dict], *,
                holdout_mod: int = HOLDOUT_MOD) -> list[dict]:
    """Объединяет источники в единый датасет с разбиением и защитой от утечки.

    Инвариант утечки (C-072): 21 проверенное полевое фото владельца — это ТЕСТ, а не обучение. Ни его
    имена, ни хэши не должны попадать в защищённую поверхность ``data/train-confirmed``. Поэтому пары из
    ``gold_matched`` в обучающий датасет не включаются вовсе (``include_gold=False`` по умолчанию); они
    служат отложенной оценкой. Если включить их явно — они помечаются ``split=holdout`` для аудита, но
    по умолчанию не пишутся: защищённую поверхность формируют только подтверждённые пары без теста.
    """
    pairs: list[dict] = []
    for source in (gold, feedback, consensus):
        for item in source:
            entry = dict(item)
            if entry.get("source") == "gold_matched":
                entry["split"] = "holdout"
            else:
                entry["split"] = assign_split(
                    entry.get("sha256") or entry.get("file") or entry.get("slug"), holdout_mod=holdout_mod)
            pairs.append(entry)
    return pairs


def write_dataset(pairs: list[dict], out_path: Path, *, include_gold: bool = False) -> int:
    """Пишет только обучаемые пары (fit/holdout без тестовых фото) в датасет.

    Пары источника ``gold_matched`` (тест) по умолчанию не пишутся: имена/хэши тестовых фото не должны
    попадать в защищённую поверхность ``data/train-confirmed`` (инвариант утечки C-072).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            if pair.get("source") == "gold_matched" and not include_gold:
                continue
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
            written += 1
    return written


def coverage_report(pairs: list[dict], catalog: dict) -> dict:
    """Сводка: сколько подтверждённых пар, источников, покрытие каталога и доля holdout."""
    confirmed = [p for p in pairs if p.get("slug") and p.get("slug") in catalog]
    fit = [p for p in confirmed if p.get("split") == "fit"]
    holdout = [p for p in confirmed if p.get("split") == "holdout"]
    by_source: dict[str, int] = {}
    for pair in confirmed:
        by_source[pair.get("source", "")] = by_source.get(pair.get("source", ""), 0) + 1
    wineries = {_clean((catalog.get(p["slug"]) or {}).get("winery")) for p in confirmed}
    wineries.discard("")
    return {
        "total_pairs": len(pairs),
        "confirmed_pairs": len(confirmed),
        "unresolved_pairs": len(pairs) - len(confirmed),
        "fit": len(fit),
        "holdout": len(holdout),
        "unique_slugs": len({p["slug"] for p in confirmed}),
        "catalog_size": len(catalog),
        "coverage_rate": round(len({p["slug"] for p in confirmed}) / len(catalog), 4) if catalog else 0.0,
        "wineries_covered": len(wineries),
        "by_source": by_source,
        "ready_for_training": len(confirmed) >= 200,
    }


def build_markdown(report: dict) -> str:
    """Markdown-отчёт покрытия подтверждённых пар."""
    lines = [
        "# Подтверждённые пары для дообучения (C-078)",
        "",
        "| Метрика | Значение |",
        "|---|---|",
        f"| всего пар | {report['total_pairs']} |",
        f"| подтверждённых (slug в каталоге) | {report['confirmed_pairs']} |",
        f"| не разрешено (нет однозначного slug) | {report['unresolved_pairs']} |",
        f"| fit / holdout | {report['fit']} / {report['holdout']} |",
        f"| уникальных slug | {report['unique_slugs']} |",
        f"| покрытие каталога | {report['coverage_rate']} ({report['unique_slugs']}/{report['catalog_size']}) |",
        f"| виноделен покрыто | {report['wineries_covered']} |",
        f"| источники | {report['by_source']} |",
        "",
        "## Вывод",
        "",
        f"- Готовность к дообучению (порог 200 подтверждённых пар): "
        f"**{'да' if report['ready_for_training'] else 'нет'}**.",
        "- Обучение на собранном объёме не запускается: подтверждённых пар слишком мало, чтобы не",
        "  переобучить голову. Датасет и разбиение fit/holdout готовы — обучение включается по мере",
        "  накопления обратной связи (flywheel) и независимых подтверждений.",
        "- Защита от утечки: holdout отделён детерминированно по хэшу фото, поэтому отложенный замер",
        "  не пересекается с fit.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI: собирает подтверждённые пары, пишет датасет и отчёты."""
    parser = argparse.ArgumentParser(description="Сборка подтверждённых пар (C-078)")
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--feedback", default=str(DEFAULT_FEEDBACK))
    parser.add_argument("--consensus", default=str(DEFAULT_CONSENSUS))
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--md", default=str(DEFAULT_MD))
    args = parser.parse_args(argv)

    sys.path.insert(0, str(BASE))
    from packages.common.catalog import load_catalog

    catalog = load_catalog(args.catalog)
    pairs = build_pairs(load_gold_pairs(Path(args.gold)), load_feedback_pairs(Path(args.feedback)),
                        load_consensus_labels(Path(args.consensus), catalog))
    report = coverage_report(pairs, catalog)

    out = Path(args.out)
    written = write_dataset(pairs, out, include_gold=False)
    report["dataset_path"] = str(out.relative_to(BASE)) if str(out).startswith(str(BASE)) else str(out)
    report["dataset_rows"] = written

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.md).write_text(build_markdown(report), encoding="utf-8")
    print(f"  пар: {report['total_pairs']}, подтверждённых: {report['confirmed_pairs']}, "
          f"fit/holdout: {report['fit']}/{report['holdout']}, "
          f"готовность к обучению: {report['ready_for_training']}")
    print(f"  датасет: {out}; отчёты: {report_path} и {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
