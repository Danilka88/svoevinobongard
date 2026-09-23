"""Тесты C-058: нормализация адресов изображений на границе ответа gateway.

Первая часть проверяет правила на уровне помощников (что проксируется, что нет), вторая —
интеграционно перебирает все медиа-эндпоинты и падает при любом сыром изображении портала.
Это и есть защита от забвения: новый эндпоинт с сырым адресом сломает тест, а не интерфейс.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from packages.common.official_api import (  # noqa: E402
    MEDIA_SIZE_CARD,
    MEDIA_SIZE_HERO,
    image_proxy_url,
    is_image_path,
    proxy_images_in,
    resize_proxy_url,
)
from services.gateway.app import main as gw  # noqa: E402

client = TestClient(gw.app)
SLUG = "nebo-rossii-glera"
IMAGE = "nebo_rossii_glera_ad8495191f.webp"
GEOJSON = "krym_3a7a0211ba.geojson"


def _raw_images(node: object, found: list[str] | None = None) -> list[str]:
    """Собирает все сырые адреса изображений портала в структуре ответа."""
    found = [] if found is None else found
    if isinstance(node, dict):
        for value in node.values():
            _raw_images(value, found)
    elif isinstance(node, list):
        for value in node:
            _raw_images(value, found)
    elif isinstance(node, str) and "/uploads/" in node and "/v1/img/str-api/" not in node:
        if is_image_path(node):
            found.append(node)
    return found


# --- правила нормализации -------------------------------------------------------

def test_image_paths_are_recognised() -> None:
    for path in ("/uploads/a.webp", "https://api.vino-svoe.ru/uploads/a.JPG", "/uploads/a.png"):
        assert is_image_path(path) is True
    for path in ("/uploads/krym.geojson", "/uploads/data.json", "/media/logo.svg", ""):
        assert is_image_path(path) is False


def test_image_gets_proxy_url() -> None:
    assert image_proxy_url(f"/uploads/{IMAGE}") == (
        f"https://api.vino-svoe.ru/v1/img/str-api/{MEDIA_SIZE_CARD}/{MEDIA_SIZE_CARD}/resize/uploads/{IMAGE}")


def test_geojson_is_never_proxied() -> None:
    """Данные не изображение: прокси отдаёт на них 404, поэтому путь остаётся как есть."""
    for raw in (f"/uploads/{GEOJSON}", f"https://api.vino-svoe.ru/uploads/{GEOJSON}"):
        assert image_proxy_url(raw) == raw


def test_already_proxied_and_local_and_foreign_untouched() -> None:
    proxied = f"https://api.vino-svoe.ru/v1/img/str-api/600/600/resize/uploads/{IMAGE}"
    for raw in (proxied, "/media/placeholder.svg", "https://cdn.example.com/uploads/a.webp"):
        assert image_proxy_url(raw) == raw


def test_nested_structures_are_normalised() -> None:
    payload = {"card": {"image_url": f"/uploads/{IMAGE}"},
               "items": [{"image_url": f"https://api.vino-svoe.ru/uploads/{IMAGE}"}],
               "region": {"geojson_url": f"/uploads/{GEOJSON}"},
               "note": "без картинки", "count": 3}
    result = proxy_images_in(payload)
    assert "/v1/img/str-api/" in result["card"]["image_url"]
    assert "/v1/img/str-api/" in result["items"][0]["image_url"]
    assert result["region"]["geojson_url"] == f"/uploads/{GEOJSON}"  # данные не тронуты
    assert result["note"] == "без картинки" and result["count"] == 3


def test_resize_helper_changes_only_size() -> None:
    proxied = f"https://api.vino-svoe.ru/v1/img/str-api/{MEDIA_SIZE_CARD}/{MEDIA_SIZE_CARD}/resize/uploads/{IMAGE}"
    resized = resize_proxy_url(proxied, MEDIA_SIZE_HERO)
    assert f"/{MEDIA_SIZE_HERO}/{MEDIA_SIZE_HERO}/resize/" in resized
    assert IMAGE in resized
    assert resize_proxy_url("https://cdn.example.com/a.webp", MEDIA_SIZE_HERO) == "https://cdn.example.com/a.webp"


# --- интеграционно: все медиа-эндпоинты -----------------------------------------

def test_scan_card_is_normalised_by_boundary(monkeypatch) -> None:
    """Скан — главный сценарий: карточка приходит с сырым адресом и нормализуется на границе."""
    from packages.contracts.schemas import Card, ScanResponse

    async def fake_scan(*args, **kwargs):
        card = Card(name="Небо России Глера", winery="Небо России",
                    image_url=f"https://api.vino-svoe.ru/uploads/{IMAGE}",
                    region_image_url=f"https://api.vino-svoe.ru/uploads/{IMAGE}")
        return ScanResponse(slug=SLUG, confidence=0.5, card=card, top5=[])

    monkeypatch.setattr(gw, "run_scan", fake_scan)
    body = client.post("/v1/scan", files={"image": ("q.jpg", b"\xff\xd8\xff" + b"0" * 64, "image/jpeg")}).json()
    assert _raw_images(body) == []
    assert "/v1/img/str-api/" in json.dumps(body, ensure_ascii=False)


@pytest.mark.parametrize("path", [
    f"/v1/similar/{SLUG}",
    f"/v1/enotourism/{SLUG}",
    f"/v1/wine/{SLUG}?with_enotourism=true",
    "/v1/catalog?per_page=5",
])
def test_media_endpoints_have_no_raw_images(path: str) -> None:
    """Защита от забвения: любой сырой адрес изображения в любом эндпоинте ломает тест."""
    response = client.get(path)
    assert response.status_code == 200, f"{path}: {response.status_code}"
    body = response.json()
    raw = _raw_images(body)
    assert raw == [], f"{path}: сырые изображения {raw[:2]}"
    assert "/v1/img/str-api/" in json.dumps(body, ensure_ascii=False), f"{path}: нет прокси-адресов"


def test_sommelier_and_wine_keep_geojson_untouched() -> None:
    """Файлы данных не должны превращаться в прокси-адреса изображений."""
    body = client.get(f"/v1/wine/{SLUG}?with_enotourism=true").json()
    text = json.dumps(body, ensure_ascii=False)
    # Все адреса .geojson обязаны остаться данными, а не превратиться в прокси изображений.
    for match in re.findall(r'"(https?://[^"]*\.geojson)"', text):
        assert "/v1/img/str-api/" not in match
    assert _raw_images({"a": "https://api.vino-svoe.ru/uploads/krym.geojson"}) == []


def test_catalog_endpoint_returns_proxy_images() -> None:
    body = client.get("/v1/catalog?per_page=5&sort=name").json()
    images = [item.get("image_url", "") for item in body.get("items", [])]
    assert images and all("/v1/img/str-api/" in src for src in images if src)
