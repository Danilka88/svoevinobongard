# START_MODULE_CONTRACT
# PURPOSE: Верификация ETL каталога — реальный прогон и инварианты данных
# SCOPE: test_artifacts_exist, test_catalog_lines_and_unique_slugs, test_stats_invariants, test_every_record_has_required_fields
# DEPENDS: M-CATALOG (scripts/catalog/etl_catalog.py)
# LINKS: M-CATALOG
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# ROOT - корень проекта
# CATALOG - путь к catalog.jsonl
# STATS - путь к stats.json
# pytestmark - пропуск тестов, если ETL не прогонялся
# test_artifacts_exist - наличие трех артефактов ETL
# test_catalog_lines_and_unique_slugs - 2103 строки и 2103 уникальных slug
# test_stats_invariants - with_image 2037, without_image 66, shared 4
# test_every_record_has_required_fields - обязательные поля каждой записи
# END_MODULE_MAP
"""V-M-001: ETL каталога «Своё Вино» — реальный вывод и инварианты.

Тесты читают артефакты, произведённые `python3 scripts/catalog/etl_catalog.py`.
Если ETL не прогонялся — тест пропускается (skip), а не падает:
это integration-gate, а не unit-заглушка.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data" / "catalog" / "catalog.jsonl"
STATS = ROOT / "data" / "catalog" / "stats.json"

pytestmark = pytest.mark.skipif(
    not (CATALOG.exists() and STATS.exists()),
    reason="ETL не прогонялся: python3 scripts/catalog/etl_catalog.py",
)


def _load_stats() -> dict:
    return json.loads(STATS.read_text(encoding="utf-8"))


def test_artifacts_exist() -> None:
    assert CATALOG.is_file(), "catalog.jsonl отсутствует"
    assert STATS.is_file(), "stats.json отсутствует"
    assert (ROOT / "data" / "catalog" / "slug_to_image.json").is_file()


def test_catalog_lines_and_unique_slugs() -> None:
    lines = [json.loads(l) for l in CATALOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2103, f"catalog.jsonl должен содержать 2103 строки, содержит {len(lines)}"
    slugs = {r["slug"] for r in lines}
    assert len(slugs) == 2103, f"уникальных slug должно быть 2103, есть {len(slugs)}"


def test_stats_invariants() -> None:
    s = _load_stats()
    assert s["csv_rows"] == 4147
    assert s["unique_slugs"] == 2103
    assert s["with_image"] == 2037
    assert s["without_image"] == 66
    assert s["sitemap_entries"] == 2037
    assert s["shared_reference_images"] == 4
    assert s["with_image"] + s["without_image"] == s["unique_slugs"]


def test_every_record_has_required_fields() -> None:
    lines = [json.loads(l) for l in CATALOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    required = {"slug", "name", "category", "color", "region", "grape",
                "description", "winery", "year", "sweetness", "image", "has_image"}
    missing = [i for i, r in enumerate(lines) if not required.issubset(r)]
    assert not missing, f"записи без обязательных полей: индексы {missing[:5]}"
    assert all((r["image"] is not None) == r["has_image"] for r in lines)
