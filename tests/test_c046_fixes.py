"""Тесты исправлений C-046: скорость поиска, вырожденные изображения, CORS, разметка, отказы сети.

Первые четыре группы проверяют поведение бэкенда и разметки. Последняя группа — проверки на уровне
исходников интерфейса: поведение браузера (отсутствие необработанных отказов промиса) в питоновском
тесте воспроизвести нельзя, поэтому проверяется наличие защит в коде, а живая проверка выполняется
поведенческим тестированием в браузере.
"""
from __future__ import annotations

import io
import sys
import time
from functools import lru_cache
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from services.gateway.app import main as gw  # noqa: E402
from packages.common.catalog import load_catalog  # noqa: E402
from packages.common.config import settings  # noqa: E402
from packages.common.text_norm import fold, forms, similarity  # noqa: E402
from services.gateway.app.media import (  # noqa: E402
    _catalog_index,
    _prepared_similarity,
    catalog_search,
)

client = TestClient(gw.app)
WEB = _ROOT / "apps" / "web"


def _png(width: int, height: int) -> bytes:
    """Крошечное валидное изображение заданного размера."""
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 30, 40)).save(buffer, format="PNG")
    return buffer.getvalue()


def _big_records(count: int = 2000) -> dict:
    """Синтетический каталог размера боевого: на нём измеряется скорость поиска."""
    regions = ("Крым", "Кубань", "Долина Дона", "Ставрополье")
    records = {}
    for index in range(count):
        records[f"wine-{index:04d}"] = {
            "name": f"Вино номер {index} резерв",
            "winery": f"Винодельня {index % 97}",
            "region": regions[index % len(regions)],
            "category": "Красное" if index % 2 else "Белое",
            "grape": "Рислинг" if index % 3 else "Каберне Совиньон",
            "year": 2018 + index % 6,
        }
    records["massandra-muskatel"] = {"name": "Мускатель Массандра белый", "winery": "Массандра",
                                     "region": "Крым", "category": "Белое", "grape": "Мускатель",
                                     "year": 2023}
    return records


# --- BUG-001: скорость и корректность поиска ------------------------------------

def test_catalog_search_is_fast_on_full_catalog() -> None:
    """Поиск по каталогу размера боевого укладывается в SLA: было 13-16 с, стало меньше секунды."""
    records = _big_records()
    _catalog_index(records)  # прогрев: индекс строится один раз на процесс
    started = time.perf_counter()
    result = catalog_search(records, {}, query="массандра")
    elapsed = time.perf_counter() - started
    assert elapsed < 1.0, f"поиск занял {elapsed:.3f} с"
    assert result["total"] >= 1
    assert any(item["slug"] == "massandra-muskatel" for item in result["items"])


def test_catalog_search_returns_correct_record_and_paging() -> None:
    records = _big_records(300)
    found = catalog_search(records, {}, query="винодельня 5")
    assert found["total"] > 0
    page = catalog_search(records, {}, page=2, per_page=10)
    assert page["page"] == 2 and len(page["items"]) <= 10
    assert page["pages"] == max(1, -(-page["total"] // 10))


def test_catalog_search_filters_by_region_and_grape() -> None:
    records = _big_records(400)
    result = catalog_search(records, {}, region="Крым", grape="мускатель")
    assert result["total"] == 1
    assert result["items"][0]["slug"] == "massandra-muskatel"


def test_catalog_index_is_reused_between_calls() -> None:
    """Индекс строится один раз: повторный поиск не должен пересобирать формы."""
    records = _big_records(500)
    first = _catalog_index(records)
    assert _catalog_index(records) is first


# --- BUG-002: вырожденные изображения ------------------------------------------

@pytest.mark.parametrize("size", [(1, 1), (2, 2), (7, 7)])
def test_tiny_image_scan_is_degraded_not_503(size: tuple[int, int]) -> None:
    """Крошечное изображение даёт корректный отказ с причиной, а не падение поиска."""
    response = client.post("/v1/scan", files={"image": ("tiny.png", _png(*size), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["not_found"] is True
    assert body["degraded"] is True
    assert body["reason"] == "image_too_small"
    assert body["top5"] == []


def test_tiny_image_predict_returns_flat_slug() -> None:
    response = client.post("/v1/eval/predict", files={"image": ("tiny.png", _png(1, 1), "image/png")})
    assert response.status_code == 200
    assert response.json() == {"slug": None}


def test_tiny_image_does_not_reach_retrieval(monkeypatch) -> None:
    """Вырожденный кадр отсекается до обращения к поиску — иначе был бы 503."""
    async def forbidden(*args, **kwargs):
        raise AssertionError("retrieval не должен вызываться для вырожденного изображения")

    monkeypatch.setattr(gw, "run_scan", forbidden)
    assert client.post("/v1/scan", files={"image": ("t.png", _png(1, 1), "image/png")}).status_code == 200


def test_zero_bytes_and_non_image_still_rejected() -> None:
    assert client.post("/v1/scan", files={"image": ("e.png", b"", "image/png")}).status_code == 400
    assert client.post("/v1/scan", files={"image": ("n.png", b"not an image", "image/png")}).status_code == 400


def test_normal_image_is_not_treated_as_tiny() -> None:
    assert gw._image_too_small(_png(64, 64)) is False


# --- BUG-005: CORS -------------------------------------------------------------

def test_cors_preflight_returns_headers() -> None:
    response = client.options("/v1/scan", headers={"Origin": "http://127.0.0.1:3000",
                                                  "Access-Control-Request-Method": "POST"})
    assert response.status_code in {200, 204}
    assert response.headers.get("access-control-allow-origin") in {"http://127.0.0.1:3000", "*"}


def test_cors_origins_are_configurable() -> None:
    assert "http://127.0.0.1:3000" in gw._CORS_ORIGINS
    assert "http://localhost:3000" in gw._CORS_ORIGINS


def test_cors_header_on_actual_request() -> None:
    response = client.post("/v1/telemetry", json={"events": []}, headers={"Origin": "http://localhost:3000"})
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


# --- BUG-003, BUG-004, BUG-006: разметка и защиты интерфейса --------------------

def test_nuxt_config_declares_language() -> None:
    config = (WEB / "nuxt.config.ts").read_text(encoding="utf-8")
    assert "htmlAttrs" in config and "lang: 'ru'" in config


def test_layout_has_no_h1() -> None:
    """Заголовок оболочки не должен быть первого уровня: на странице остаётся один h1."""
    layout = (WEB / "layouts" / "default.vue").read_text(encoding="utf-8")
    assert "<h1" not in layout


def test_telemetry_plugin_ignores_network_rejections() -> None:
    plugin = (WEB / "plugins" / "telemetry.client.ts").read_text(encoding="utf-8")
    assert "unhandledrejection" in plugin
    assert "preventDefault" in plugin  # сетевые отказы не превращаются в ошибку скрипта


def test_gateway_client_has_safe_fallbacks() -> None:
    """Отказ gateway даёт состояние экрана, а не необработанный отказ промиса."""
    client_source = (WEB / "composables" / "useGateway.ts").read_text(encoding="utf-8")
    assert client_source.count("catch {") >= 6
    assert "unavailable: true" in client_source


# --- эквивалентность прежней логике на боевом каталоге --------------------------

@lru_cache(maxsize=1)
def _real_catalog() -> dict:
    """Боевой каталог читается один раз на процесс: иначе каждый тест строит формы заново."""
    return load_catalog(settings.catalog_jsonl)


@pytest.mark.parametrize("query", ["пино нуар", "захарин"])
def test_prepared_similarity_matches_original_decisions(query: str) -> None:
    """Решение по каждой записи совпадает с прежним вызовом similarity.

    Сравнение идёт по выборке: решение принимается по записи независимо, поэтому проверка
    эквивалентности не требует прогонять все 2103 записи на каждый тест.
    """
    catalog = _real_catalog()
    index = _catalog_index(catalog)
    query_forms = forms(query)
    for slug in list(catalog)[:250]:
        original = similarity(query, index[slug]["haystack"]) >= 0.6
        prepared = _prepared_similarity(query_forms, index[slug]["forms"]) >= 0.6
        assert prepared == original, f"расхождение на {slug}"


def test_search_total_matches_previous_logic_on_real_catalog() -> None:
    """Итоговый отбор по боевому каталогу совпадает с прежней формулой."""
    catalog = _real_catalog()
    index = _catalog_index(catalog)
    query = "пино нуар"
    query_folded = fold(query)
    expected = sum(1 for slug in catalog
                   if query_folded in index[slug]["haystack"]
                   or similarity(query, index[slug]["haystack"]) >= 0.6)
    result = catalog_search(catalog, {}, query=query, per_page=100)
    assert result["total"] == expected


def test_search_is_fast_on_real_catalog() -> None:
    """SLA поиска на боевом каталоге: было 13-16 с, требуется меньше секунды."""
    catalog = _real_catalog()
    _catalog_index(catalog)  # прогрев индекса форм
    started = time.perf_counter()
    result = catalog_search(catalog, {}, query="пино нуар")
    elapsed = time.perf_counter() - started
    assert result["total"] > 0
    assert elapsed < 1.0, f"поиск занял {elapsed:.3f} с"
