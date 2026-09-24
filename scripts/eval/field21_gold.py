#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Эталонный маппинг 21 полевого фото к slug каталога и машинная проверка отсутствия утечки тестовых фото
# SCOPE: DEFAULT_FIELD_DIR, DEFAULT_MANIFEST, DEFAULT_CATALOG, GOLD_OUT, LEAK_OUT, BRAND_KEYS,
#        BRAND_MATCH_MIN, FOREIGN_MATCH_MIN, MATCH_GAP, AMBIG_WINDOW, CANDIDATE_LIMIT, PROTECTED_ROOTS, TEXT_SUFFIXES,
#        MANUAL_OVERRIDES, file_sha256, read_manifest, brand_keys, gated_candidates, map_label,
#        build_gold, gold_status_counts, render_gold_summary, load_gold, iter_protected_files,
#        leak_scan, clean_flywheel_leak, main
# DEPENDS: argparse, hashlib, json, re, pathlib, packages.common.catalog, packages.common.text_norm,
#          packages.common.text_rank
# LINKS: M-EVAL, M-DATA
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# DEFAULT_FIELD_DIR - каталог 21 проверенного полевого фото
# DEFAULT_MANIFEST - manifest.tsv с человеко-проверенными названиями
# DEFAULT_CATALOG - catalog.jsonl источник slug
# GOLD_OUT - путь отчёта эталонного маппинга
# LEAK_OUT - путь отчёта проверки утечки
# BRAND_KEYS - канонические бренды, по которым строится ворота винодельни
# BRAND_MATCH_MIN - минимальный скор совпадения внутри распознанного бренда
# FOREIGN_MATCH_MIN - повышенный порог, когда бренд не распознан (защита от ложных совпадений)
# MATCH_GAP - минимальный отрыв top-1 от top-2 для однозначного matched
# AMBIG_WINDOW - окно скора вокруг top-1 для набора приемлемых кандидатов
# CANDIDATE_LIMIT - сколько кандидатов хранить в отчёте
# PROTECTED_ROOTS - поверхности, куда тестовые фото не должны утекать
# TEXT_SUFFIXES - расширения текстовых артефактов для проверки хешей
# MANUAL_OVERRIDES - курируемые правки эталона с обоснованием
# file_sha256 - sha256 файла
# read_manifest - чтение manifest.tsv в пары (файл, человеко-проверенное название)
# brand_keys - канонические бренды, встречающиеся в строке
# gated_candidates - кандидаты каталога после ворот винодельни и окна скора
# map_label - статус фото: matched, ambiguous или not_found
# build_gold - эталонный маппинг всех фото с ручными правками
# gold_status_counts - сводка по статусам
# render_gold_summary - человекочитаемая сводка эталона
# load_gold - чтение сохранённого эталона
# iter_protected_files - текстовые артефакты защищённых поверхностей
# leak_scan - поиск хешей/имён тестовых фото в защищённых поверхностях и логе flywheel
# clean_flywheel_leak - удаление записей лога flywheel по хешам тестовых фото
# main - CLI: gold, leak-scan, clean
# END_MODULE_MAP
"""Эталонный маппинг 21 фото и инвариант утечки.

Человеко-проверенное название из ``manifest.tsv`` — слабая метка: владелец смотрел на
этикетку, но записывал название разговорно, без точного slug каталога. Поэтому маппинг
идёт через ``text_norm`` (транслит, алиасы, OCR-двойники) и текстовый поиск по формам
каталога, а результат честно помечается:

* ``matched`` — однозначный slug (отрыв top-1 от top-2 и достаточный скор);
* ``ambiguous`` — несколько равнозначных кандидатов серии/линейки (приводится набор);
* ``not_found`` — вина нет в каталоге (зарубежное или отсутствующий продукт).

Ручные правки (``MANUAL_OVERRIDES``) не «выдумывают» истину: каждая фиксирует, что именно
дал автоматический поиск и почему выбрано другое значение. Тот же модуль машинно проверяет,
что sha256 тестовых фото не встречается в галерее, индексе, обучающих данных и логе запросов.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from packages.common.catalog import load_catalog  # noqa: E402
from packages.common.text_norm import ALIASES, fold  # noqa: E402
from packages.common.text_rank import text_index, text_search  # noqa: E402

DEFAULT_FIELD_DIR = _ROOT / "data" / "samples" / "real_field"
DEFAULT_MANIFEST = DEFAULT_FIELD_DIR / "manifest.tsv"
DEFAULT_CATALOG = _ROOT / "data" / "catalog" / "catalog.jsonl"
GOLD_OUT = _ROOT / "reports" / "field21_gold.json"
LEAK_OUT = _ROOT / "reports" / "field21_leak.json"

# Ворота винодельни: сравниваем только внутри того же бренда. Список — подмножество
# курируемых ALIASES, где канонический ключ действительно бренд, а не сорт/тип/цвет/регион.
BRAND_KEYS = frozenset({
    "абрау дюрсо", "шато тамань", "фанагория", "новый свет", "высокий берег",
    "золотая балка", "лефкадия", "ведерников", "коктебель", "амфитрион", "мысхако",
    "геленджик", "кубань вино",
})
BRAND_MATCH_MIN = 0.45
FOREIGN_MATCH_MIN = 0.58
MATCH_GAP = 0.04
AMBIG_WINDOW = 0.05
CANDIDATE_LIMIT = 8

PROTECTED_ROOTS = (
    _ROOT / "data" / "index",
    _ROOT / "data" / "train",
    _ROOT / "data" / "train-confirmed",
    _ROOT / "data" / "annotation",
)
TEXT_SUFFIXES = frozenset({".json", ".jsonl", ".txt", ".tsv", ".csv", ".md", ".yaml", ".yml"})

# Курируемые правки. auto — что дал автоматический поиск, note — почему выбрано решение.
MANUAL_OVERRIDES: dict[str, dict] = {
    "Abrau Durso Reserve брюд.jpg": {
        "status": "matched",
        "slug": "abrau-dyurso-abrau-durso-reserve-brut-shardone-beloe-bryut-115",
        "note": "auto путает Reserve Brut и Brut Rose Reserve; в названии владельца нет «rose»",
    },
    "fanagory 5 element.jpg": {
        "status": "matched",
        "slug": "fanagoriya-5-elements-bryut-beloe-pino-gri-igristoe-bryut-beloe-125",
        "note": "auto ставит линейный брют выше; владелец прямо указал «5 element»",
    },
    "chateau tamagne semi-sweet полусладкое.jpg": {
        "status": "matched",
        "slug": "shato-taman-polusladkoe",
        "note": "Chateau Tamagne — латинское имя Шато Тамань (Кубань-Вино); полусладкое в каталоге",
    },
    "Лефкадия шардоне.jpg": {
        "status": "not_found",
        "note": "у Долины Лефкадия в каталоге нет Шардоне (Белое/Красное/Розе Мальбек/Темелион)",
    },
    "абрау дюрсо премиум.jpg": {
        "status": "not_found",
        "note": "линейки «Премиум» у Абрау-Дюрсо в каталоге нет; auto цепляется за бренд",
    },
    "Abrau Light.jpeg": {
        "status": "not_found",
        "note": "линейки Abrau Light в каталоге нет; auto находит только бренд, не продукт",
    },
}

__all__ = ["AMBIG_WINDOW", "BRAND_KEYS", "BRAND_MATCH_MIN", "CANDIDATE_LIMIT", "DEFAULT_CATALOG",
           "DEFAULT_FIELD_DIR", "DEFAULT_MANIFEST", "FOREIGN_MATCH_MIN", "GOLD_OUT", "LEAK_OUT",
           "MANUAL_OVERRIDES", "MATCH_GAP", "PROTECTED_ROOTS", "TEXT_SUFFIXES", "brand_keys",
           "build_gold", "clean_flywheel_leak", "file_sha256", "gated_candidates", "gold_status_counts",
           "iter_protected_files", "leak_scan", "load_gold", "main", "map_label", "read_manifest",
           "render_gold_summary"]


def file_sha256(path: str | Path) -> str:
    """Считает sha256 файла потоком (без чтения целиком в память)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: str | Path) -> list[tuple[str, str]]:
    """Читает manifest.tsv: первый столбец — файл, второй — человеко-проверенное название.

    Строки без второго столбца пропускаются: без метки фото не оценивается.
    """
    rows: list[tuple[str, str]] = []
    target = Path(path)
    if not target.is_file():
        return rows
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "\t" not in line:
            continue
        name, _, label = line.partition("\t")
        name, label = name.strip(), label.strip()
        if name and label:
            rows.append((name, label))
    return rows


def brand_keys(text: str | None) -> set[str]:
    """Возвращает канонические бренды, любой вариант написания которых есть в строке."""
    value = fold(text)
    found: set[str] = set()
    if not value:
        return found
    for canonical in BRAND_KEYS:
        variants = set(ALIASES.get(canonical, ())) | {canonical}
        if any((fold(variant) or "") and fold(variant) in value for variant in variants):
            found.add(canonical)
    return found


def gated_candidates(label: str, catalog: dict, index: dict, *, limit: int = CANDIDATE_LIMIT) -> list[dict]:
    """Кандидаты каталога внутри ворот винодельни, отфильтрованные окном скора.

    Если бренд в названии не распознан, ворота открыты (иначе зарубежные вина нельзя
    честно пометить как ``not_found``). Набор кандидатов ограничивается окном
    ``AMBIG_WINDOW`` вокруг top-1: это «приемлемые» позиции линейки, а не весь топ поиска.
    """
    hits = text_search({"wine_name": label, "winery": label}, catalog, limit=50, index=index)
    wanted = brand_keys(label)
    filtered: list[tuple[float, str]] = []
    for slug, score in hits:
        record = catalog.get(slug) or {}
        haystack = fold(f"{record.get('winery', '')} {record.get('name', '')}")
        if wanted:
            variants = {fold(v) for key in wanted for v in (set(ALIASES.get(key, ())) | {key})}
            if not any(variant and variant in haystack for variant in variants):
                continue
        filtered.append((float(score), slug))
    if not filtered:
        return []
    top = filtered[0][0]
    window = [(score, slug) for score, slug in filtered if score >= top - AMBIG_WINDOW]
    result: list[dict] = []
    for score, slug in window[:limit]:
        record = catalog.get(slug) or {}
        result.append({"slug": slug, "score": round(score, 4),
                       "winery": record.get("winery", ""), "name": record.get("name", "")})
    return result


def map_label(file_name: str, label: str, catalog: dict, index: dict) -> dict:
    """Определяет статус фото: matched, ambiguous или not_found.

    matched — есть скор выше порога и заметный отрыв; ambiguous — несколько равнозначных
    кандидатов (приводится набор приемлемых); not_found — ничего правдоподобного нет.
    """
    candidates = gated_candidates(label, catalog, index)
    threshold = BRAND_MATCH_MIN if brand_keys(label) else FOREIGN_MATCH_MIN
    record: dict = {
        "file": file_name,
        "human_label": label,
        "source": "auto",
        "candidates": candidates,
    }
    if not candidates or candidates[0]["score"] < threshold:
        record.update({"status": "not_found", "expected_slug": None, "score": 0.0})
        return record
    top, second = candidates[0]["score"], (candidates[1]["score"] if len(candidates) > 1 else 0.0)
    record["score"] = top
    if len(candidates) == 1 or (top - second) >= MATCH_GAP:
        record.update({"status": "matched", "expected_slug": candidates[0]["slug"]})
    else:
        record.update({"status": "ambiguous", "expected_slug": None})
    return record


def build_gold(field_dir: str | Path, manifest_path: str | Path, catalog: dict,
               overrides: dict | None = None) -> dict:
    """Строит эталонный маппинг всех фото и применяет курируемые правки.

    Итог: ``{"files": {file: record}, "overrides": {...}}``. Для каждого файла сохраняется
    и автоматический результат, и ручное решение (если оно есть), чтобы правку можно было
    проверить, а не принимать на веру.
    """
    directory = Path(field_dir)
    index = text_index(catalog)
    overrides = overrides if overrides is not None else MANUAL_OVERRIDES
    files: dict[str, dict] = {}
    for name, label in read_manifest(manifest_path):
        record = map_label(name, label, catalog, index)
        record["auto"] = {"status": record.get("status"), "expected_slug": record.get("expected_slug"),
                          "candidates": list(record.get("candidates") or [])}
        override = overrides.get(name)
        if override:
            record.update({
                "status": override["status"],
                "expected_slug": override.get("slug"),
                "source": "manual",
                "note": override.get("note", ""),
            })
        path = directory / name
        record["sha256"] = file_sha256(path) if path.is_file() else ""
        record["exists"] = path.is_file()
        files[name] = record
    return {"files": files, "overrides": {k: v for k, v in overrides.items() if k in files},
            "counts": gold_status_counts(files)}


def gold_status_counts(files: dict) -> dict:
    """Сводка по статусам эталона: matched, ambiguous, not_found и размер вычислимого набора."""
    files = files.get("files", files) if isinstance(files, dict) and "files" in files else files
    counts = {"matched": 0, "ambiguous": 0, "not_found": 0}
    for record in files.values():
        status = str(record.get("status") or "not_found")
        counts[status] = counts.get(status, 0) + 1
    counts["evaluable"] = counts.get("matched", 0) + counts.get("ambiguous", 0)
    counts["total"] = len(files)
    return counts


def render_gold_summary(gold: dict) -> str:
    """Человекочитаемая сводка эталона: счёт статусов, неоднозначные и ненайденные с кандидатами."""
    files = gold.get("files", {})
    counts = gold.get("counts") or gold_status_counts(files)
    lines = [
        "=== ЭТАЛОН 21 ФОТО ===",
        f"  всего {counts.get('total', 0)} · однозначных {counts.get('matched', 0)} · "
        f"неоднозначных {counts.get('ambiguous', 0)} · ненайденных {counts.get('not_found', 0)} · "
        f"вычислимых {counts.get('evaluable', 0)}",
    ]
    for status, title in (("matched", "ОДНОЗНАЧНЫЕ"), ("ambiguous", "НЕОДНОЗНАЧНЫЕ"), ("not_found", "НЕНАЙДЕННЫЕ")):
        rows = [(name, rec) for name, rec in files.items() if rec.get("status") == status]
        if not rows:
            continue
        lines.append(f"  --- {title} ---")
        for name, rec in rows:
            chosen = rec.get("expected_slug") or "—"
            lines.append(f"    {name} [{rec.get('source')}] -> {chosen}")
            if status == "ambiguous":
                for candidate in (rec.get("candidates") or [])[:5]:
                    lines.append(f"        · {candidate['slug']} ({candidate['score']})")
    return "\n".join(lines)


def load_gold(path: str | Path) -> dict:
    """Читает сохранённый эталон; отсутствие файла — пустой словарь."""
    target = Path(path)
    if not target.is_file():
        return {}
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def iter_protected_files(roots=PROTECTED_ROOTS, suffixes=TEXT_SUFFIXES):
    """Перебирает текстовые артефакты защищённых поверхностей (галерея, индекс, обучение)."""
    for root in roots:
        base = Path(root)
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in suffixes:
                yield path


def leak_scan(field_dir: str | Path, *, roots=PROTECTED_ROOTS, flywheel_path: str | Path | None = None,
              hashes: dict[str, str] | None = None) -> dict:
    """Ищет хеши и имена тестовых фото в галерее/индексе/обучении и в логе запросов flywheel.

    Возвращает отчёт с найденными утечками. Утечка в обучающих артефактах — это ошибка;
    утечка в логе запросов — материал для очистки (лог не должен питать обучение тестом).
    """
    directory = Path(field_dir)
    if hashes is None:
        hashes = {p.name: file_sha256(p) for p in sorted(directory.glob("*"))
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    hash_values = set(hashes.values())
    names = set(hashes)
    protected_hits: list[dict] = []
    checked = 0
    for path in iter_protected_files(roots):
        checked += 1
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for value in hash_values:
            if value in text:
                protected_hits.append({"surface": str(path), "kind": "hash", "value": value[:12]})
        for name in names:
            if name in text:
                protected_hits.append({"surface": str(path), "kind": "name", "value": name})
    flywheel_path = Path(flywheel_path) if flywheel_path else (_ROOT / "data" / "flywheel" / "queries.jsonl")
    flywheel_hits: list[str] = []
    flywheel_total = 0
    if flywheel_path.is_file():
        for line in flywheel_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            flywheel_total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(record.get("image_sha256") or "") in hash_values:
                flywheel_hits.append(str(record.get("image_sha256")))
    return {
        "field_photos": len(hashes),
        "protected_files_checked": checked,
        "protected_hits": protected_hits,
        "protected_clean": not protected_hits,
        "flywheel_path": str(flywheel_path),
        "flywheel_entries": flywheel_total,
        "flywheel_hits": len(flywheel_hits),
        "flywheel_clean": not flywheel_hits,
        "hashes": hashes,
    }


def clean_flywheel_leak(path: str | Path, hashes) -> int:
    """Удаляет из лога flywheel записи с хешами тестовых фото; возвращает число удалённых."""
    target = Path(path)
    if not target.is_file():
        return 0
    values = set(hashes.values()) if isinstance(hashes, dict) else set(hashes)
    kept: list[str] = []
    removed = 0
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            kept.append(line)
            continue
        if str(record.get("image_sha256") or "") in values:
            removed += 1
            continue
        kept.append(line)
    if removed:
        target.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return removed


def main(argv: list[str] | None = None) -> int:
    """CLI: построить эталон, проверить утечку (и при необходимости очистить лог flywheel)."""
    parser = argparse.ArgumentParser(description="Эталон 21 фото и инвариант утечки")
    parser.add_argument("--dir", default=str(DEFAULT_FIELD_DIR))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--out", default=str(GOLD_OUT))
    parser.add_argument("--leak-scan", action="store_true", help="проверить утечку вместо построения эталона")
    parser.add_argument("--clean", action="store_true", help="удалить тестовые хеши из лога flywheel")
    parser.add_argument("--leak-out", default=str(LEAK_OUT))
    args = parser.parse_args(argv)

    if args.leak_scan:
        report = leak_scan(args.dir)
        if args.clean and report["flywheel_hits"]:
            report["flywheel_removed"] = clean_flywheel_leak(report["flywheel_path"], report["hashes"])
            report["flywheel_hits_after"] = leak_scan(args.dir)["flywheel_hits"]
        out = Path(args.leak_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("=== ИНВАРИАНТ УТЕЧКИ ===")
        print(f"  тестовых фото: {report['field_photos']} · проверено артефактов: {report['protected_files_checked']}")
        print(f"  утечка в галерею/индекс/обучение: {len(report['protected_hits'])} (clean={report['protected_clean']})")
        print(f"  утечка в лог flywheel: {report['flywheel_hits']} из {report['flywheel_entries']}")
        if "flywheel_removed" in report:
            print(f"  удалено записей лога: {report['flywheel_removed']} · осталось утечек: {report['flywheel_hits_after']}")
        print(f"  отчёт: {out}")
        return 0

    catalog = load_catalog(args.catalog)
    gold = build_gold(args.dir, args.manifest, catalog)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")
    print(render_gold_summary(gold))
    print(f"  отчёт: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
