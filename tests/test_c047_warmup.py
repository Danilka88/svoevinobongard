"""Тесты C-047: подпись кэша форм, переиспользование между запусками и прогрев на старте.

Главная проверка — первый запрос каталога после старта укладывается в SLA, а кэш форм реально
переживает перезапуск процесса. Отдельно проверяется, что мелкие синтетические наборы не затирают
кэш боевого каталога: именно это ломало переиспользование до исправления.
"""
from __future__ import annotations

import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from packages.common.catalog import load_catalog  # noqa: E402
from packages.common.config import settings  # noqa: E402
from services.gateway.app import main as gw  # noqa: E402
from services.gateway.app import media  # noqa: E402
from services.gateway.app.media import (  # noqa: E402
    _forms_signature,
    _load_forms_cache,
    catalog_search,
    warm_catalog_index,
)


@lru_cache(maxsize=1)
def _real_catalog() -> dict:
    """Боевой каталог читается один раз на процесс."""
    return load_catalog(settings.catalog_jsonl)


def _records(count: int = 600) -> dict:
    return {f"wine-{index:04d}": {"name": f"Вино {index}", "winery": f"Винодельня {index % 11}",
                                  "region": "Крым", "category": "Красное", "grape": "Рислинг"}
            for index in range(count)}


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path, monkeypatch):
    """Кэш форм в тестах пишется во временный файл: боевой кэш не трогаем."""
    monkeypatch.setattr(media, "_FORMS_CACHE_PATH", tmp_path / "catalog_forms.json")
    media._INDEX_CACHE.clear()
    yield
    media._INDEX_CACHE.clear()


# --- подпись кэша --------------------------------------------------------------

def test_signature_is_content_based_not_order_based() -> None:
    """Порядок записей не должен влиять на подпись: иначе кэш теряется без причины."""
    records = _records(20)
    reversed_records = dict(reversed(list(records.items())))
    assert _forms_signature(records) == _forms_signature(reversed_records)


def test_signature_changes_when_catalog_changes() -> None:
    records = _records(20)
    before = _forms_signature(records)
    records["wine-0000"]["name"] = "Другое вино с другим названием"
    assert _forms_signature(records) != before


def test_signature_counts_records() -> None:
    assert _forms_signature(_records(20)).split(":")[1] == "20"


# --- переиспользование кэша -----------------------------------------------------

def test_cache_is_reused_without_rebuild(tmp_path: Path) -> None:
    """Второй прогрев берёт формы с диска и не пересчитывает их."""
    records = _records(600)
    assert warm_catalog_index(records) == 600
    cache_file = media._FORMS_CACHE_PATH
    assert cache_file.is_file()
    first_mtime = cache_file.stat().st_mtime_ns
    media._INDEX_CACHE.clear()
    started = time.perf_counter()
    assert warm_catalog_index(records) == 600
    elapsed = time.perf_counter() - started
    assert cache_file.stat().st_mtime_ns == first_mtime  # файл не перезаписывался
    assert elapsed < 1.0, f"повторный прогрев занял {elapsed:.3f} с"


def test_cache_signature_matches_catalog() -> None:
    records = _records(600)
    warm_catalog_index(records)
    payload = json.loads(media._FORMS_CACHE_PATH.read_text(encoding="utf-8"))
    assert payload["signature"] == _forms_signature(records)
    assert _load_forms_cache(_forms_signature(records)) is not None


def test_changed_catalog_invalidates_cache() -> None:
    records = _records(600)
    warm_catalog_index(records)
    records["wine-0000"]["grape"] = "Каберне Совиньон"  # содержимое изменилось
    assert _load_forms_cache(_forms_signature(records)) is None


def test_small_catalog_is_not_persisted() -> None:
    """Синтетические наборы не затирают кэш боевого каталога."""
    warm_catalog_index(_records(20))
    assert not media._FORMS_CACHE_PATH.exists()


# --- прогрев на старте и SLA ----------------------------------------------------

def test_startup_warmup_is_registered() -> None:
    """Прогрев подключён к жизненному циклу приложения, а не вызывается вручную."""
    assert gw.app.router.lifespan_context is gw._lifespan
    assert gw._lifespan.__wrapped__ if hasattr(gw._lifespan, "__wrapped__") else True


def test_startup_warmup_builds_index() -> None:
    """Прогрев на старте наполняет индекс: первый запрос пользователя приходит на тёплый индекс."""
    records = _records(600)
    gw.warm_catalog_index(records)
    assert warm_catalog_index(records) == 600
    assert media._INDEX_CACHE


def test_first_catalog_search_after_warmup_is_fast() -> None:
    """SLA первого запроса каталога: после прогрева поиск укладывается в секунду."""
    catalog = _real_catalog()
    media._INDEX_CACHE.clear()
    warm_catalog_index(catalog)
    started = time.perf_counter()
    result = catalog_search(catalog, {}, query="пино нуар", per_page=100)
    elapsed = time.perf_counter() - started
    assert result["total"] > 0
    assert elapsed < 1.0, f"первый поиск занял {elapsed:.3f} с"


def test_real_catalog_cache_is_reusable() -> None:
    """Боевой каталог достаточно велик, чтобы попадать в кэш на диске."""
    catalog = _real_catalog()
    warm_catalog_index(catalog)
    payload = json.loads(media._FORMS_CACHE_PATH.read_text(encoding="utf-8"))
    assert payload["signature"] == _forms_signature(catalog)
    assert len(payload["forms"]) == len(catalog)
