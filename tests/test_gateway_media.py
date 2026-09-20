# START_MODULE_CONTRACT
# PURPOSE: Верификация медиа-обогащения и UI-эндпоинтов gateway: поля, метрики демо, браузер каталога
# SCOPE: test_media_index_has_entries, test_media_for_absolute_urls, test_demo_metrics_missing_file_ok, test_demo_metrics_reads_reports, test_catalog_search_filters, test_catalog_pagination_and_sort, test_card_schema_has_media_fields, test_sommelier_schema_additive_fields
# DEPENDS: services.gateway.app.media, packages.contracts.schemas, pytest
# LINKS: M-GATEWAY, M-CATALOG, V-M-GATEWAY
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# RECORDS - тестовые записи каталога для браузера
# MEDIA - тестовые медиа-записи
# test_media_index_has_entries - медиа-индекс читается и содержит позиции с картинками
# test_media_for_absolute_urls - URL в медиа абсолютные
# test_demo_metrics_missing_file_ok - отсутствие отчёта не является ошибкой
# test_demo_metrics_reads_reports - метрики собираются из отчётов в сводку дашборда
# test_catalog_search_filters - поиск и фильтры браузера каталога работают через нормализацию
# test_catalog_pagination_and_sort - пагинация и сортировка считаются корректно
# test_card_schema_has_media_fields - схема карточки содержит аддитивные медиа-поля
# test_sommelier_schema_additive_fields - схема сомелье содержит карточки блюд, аналоги и заметку
# END_MODULE_MAP
"""V-M-GATEWAY: медиа-обогащение и UI-эндпоинты."""
from __future__ import annotations

import json

from packages.contracts.schemas import Card, SommelierResponse
from services.gateway.app.media import catalog_search, demo_metrics, load_media, media_for

__all__ = [
    "MEDIA",
    "RECORDS",
    "test_card_schema_has_media_fields",
    "test_catalog_pagination_and_sort",
    "test_catalog_search_filters",
    "test_demo_metrics_missing_file_ok",
    "test_demo_metrics_reads_reports",
    "test_media_for_absolute_urls",
    "test_media_index_has_entries",
    "test_sommelier_schema_additive_fields",
]

RECORDS = {
    "shato-taman-polusladkoe": {"slug": "shato-taman-polusladkoe", "name": "Шато Тамань Полусладкое",
                                "winery": "Кубань-Вино", "region": "Кубань", "category": "Белое",
                                "grape": "Рислинг", "year": 2022},
    "abrau-dyurso-bryut": {"slug": "abrau-dyurso-bryut", "name": "Абрау Дюрсо Брют",
                           "winery": "Абрау-Дюрсо", "region": "Кубань", "category": "Белое",
                           "grape": "Шардоне", "year": 2021},
    "kokur-suhoe": {"slug": "kokur-suhoe", "name": "Кокур Сухое", "winery": "Массандра",
                    "region": "Крым", "category": "Белое", "grape": "Кокур", "year": 2020},
}
MEDIA = {
    "shato-taman-polusladkoe": {"image_url": "https://api.vino-svoe.ru/uploads/a.webp",
                                "category_gradient": "linear-gradient(180deg, #fff 0%, #000 100%)",
                                "dish_images": [{"name": "Сыры", "url": "https://api.vino-svoe.ru/uploads/cheese.webp"}]},
}


def test_media_index_has_entries() -> None:
    """Медиа-индекс читается из артефакта и содержит позиции с картинками."""
    index = load_media()
    assert index, "медиа-индекс не должен быть пустым"
    assert all(str(entry.get("slug")) for entry in list(index.values())[:50])


def test_media_for_absolute_urls() -> None:
    """URL в медиа абсолютные, а не относительные пути API."""
    entry = media_for("a-gordienko-m-nikolaev-pino-nuar-krasnoe-suhoe-135")
    assert entry.get("image_url", "").startswith("http")
    assert entry.get("region_geojson_url", "").startswith("http")


def test_demo_metrics_missing_file_ok(tmp_path) -> None:
    """Отсутствие отчётов не является ошибкой."""
    result = demo_metrics(tmp_path)
    assert result["sources"]["final"] == "missing"
    assert result["dashboard"]["proxy_acc_at_1"] is None


def test_demo_metrics_reads_reports(tmp_path) -> None:
    """Метрики собираются из отчётов в сводку дашборда."""
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "final.json").write_text(json.dumps({
        "quality": {"acc_at_1": 0.775, "acc_at_5": 0.97},
        "latency_ms": {"p50": 130.0, "p95": 160.0}, "sla_ms": 3000.0, "min_acc_at_1": 0.75,
    }), encoding="utf-8")
    (tmp_path / "reports" / "real_field.json").write_text(json.dumps({
        "identity_at_1_russian": 0.5263, "identity_at_5_russian": 0.6316, "cases": 23,
        "latency_ms": {"p95": 1091.4},
    }), encoding="utf-8")
    dashboard = demo_metrics(tmp_path)["dashboard"]
    assert dashboard["proxy_acc_at_1"] == 0.775
    assert dashboard["real_identity_at_1"] == 0.5263
    assert dashboard["sla_ms"] == 3000.0


def test_catalog_search_filters() -> None:
    """Поиск и фильтры работают через нормализацию текста."""
    page = catalog_search(RECORDS, MEDIA, query="рисл")
    assert [item["slug"] for item in page["items"]] == ["shato-taman-polusladkoe"]
    assert catalog_search(RECORDS, MEDIA, region="крым")["total"] == 1
    assert catalog_search(RECORDS, MEDIA, winery="абрау")["total"] == 1
    assert page["items"][0]["image_url"].startswith("http")


def test_catalog_pagination_and_sort() -> None:
    """Пагинация и сортировка считаются корректно."""
    page = catalog_search(RECORDS, MEDIA, per_page=2, page=2, sort="winery")
    assert page["total"] == 3 and page["pages"] == 2 and len(page["items"]) == 1
    first_page = catalog_search(RECORDS, MEDIA, sort="year", per_page=3)
    assert first_page["items"][0]["year"] == 2020, "сортировка по году по возрастанию"


def test_card_schema_has_media_fields() -> None:
    """Схема карточки содержит аддитивные медиа-поля."""
    card = Card()
    for field in ("category_gradient", "region_image_url", "region_geojson_url",
                  "grape_images", "dish_images"):
        assert hasattr(card, field), f"в карточке нет {field}"


def test_sommelier_schema_additive_fields() -> None:
    """Схема сомелье содержит карточки блюд, аналоги и заметку, не ломая pairing."""
    response = SommelierResponse(slug="a")
    assert response.pairing == [] and response.pairing_cards == []
    assert response.alternatives == [] and response.sensory_note == ""
