"""Тесты C-050: прокси-адреса картинок портала и устойчивое отображение.

Сырые адреса портала отдают 404, рабочий путь — прокси изображений. Тесты проверяют построение
адреса, неизменность локальных и сторонних ссылок, состав производного файла медиа и то, что
интерфейс использует единую обёртку изображения.
"""
from __future__ import annotations

import json
from pathlib import Path

from packages.common.official_api import (  # noqa: E402
    MEDIA_SIZE_CARD,
    MEDIA_SIZE_HERO,
    image_proxy_url,
    proxy_images_in,
)
from services.gateway.app.media import catalog_search  # noqa: E402

WEB = Path(__file__).resolve().parents[1] / "apps" / "web"
MEDIA_FILE = Path(__file__).resolve().parents[1] / "data" / "catalog" / "catalog_media.json"
FILE = "a_gordienko_m_nikolaev_pino_nuar_krasnoe_suhoe_135_660d8a5012.webp"


def test_proxy_url_from_relative_path() -> None:
    url = image_proxy_url(f"/uploads/{FILE}")
    assert url == f"https://api.vino-svoe.ru/v1/img/str-api/{MEDIA_SIZE_CARD}/{MEDIA_SIZE_CARD}/resize/uploads/{FILE}"


def test_proxy_url_from_full_portal_url() -> None:
    """В кэше API адреса полные — помощник обязан понимать и их."""
    url = image_proxy_url(f"https://api.vino-svoe.ru/uploads/{FILE}")
    assert url.startswith("https://api.vino-svoe.ru/v1/img/str-api/")
    assert url.endswith(f"/resize/uploads/{FILE}")


def test_proxy_size_is_configurable() -> None:
    url = image_proxy_url(f"/uploads/{FILE}", size=MEDIA_SIZE_HERO)
    assert f"/{MEDIA_SIZE_HERO}/{MEDIA_SIZE_HERO}/resize/" in url


def test_local_and_foreign_urls_are_untouched() -> None:
    """Локальные ассеты и чужие домены переписывать нельзя."""
    for raw in ("/media/logo-svoe-vino.svg", "https://example.com/uploads/x.webp",
                "https://cdn.other.ru/pic.png", ""):
        assert image_proxy_url(raw) == raw


def test_proxy_images_in_recurses() -> None:
    payload = {"image": {"url": f"/uploads/{FILE}"},
               "grapes": [{"url": f"/uploads/{FILE}"}],
               "note": "без картинки", "count": 3}
    result = proxy_images_in(payload)
    assert "/v1/img/str-api/" in result["image"]["url"]
    assert "/v1/img/str-api/" in result["grapes"][0]["url"]
    assert result["note"] == "без картинки" and result["count"] == 3


def test_media_file_uses_proxy_urls() -> None:
    """Производный файл медиа не должен содержать сырых адресов портала."""
    records = json.loads(MEDIA_FILE.read_text(encoding="utf-8"))
    assert len(records) > 2000
    # Прокси-адрес сам содержит /resize/uploads/, поэтому «сырым» считается адрес без прокси-префикса.
    proxy = sum(1 for record in records.values() if "/v1/img/str-api/" in str(record.get("image_url")))
    raw = sum(1 for record in records.values()
              if "/uploads/" in str(record.get("image_url"))
              and "/v1/img/str-api/" not in str(record.get("image_url")))
    assert proxy > 2000, f"прокси-адресов {proxy}"
    assert raw == 0, f"сырых адресов осталось {raw}"


def test_catalog_search_returns_proxy_url() -> None:
    """Даже при сыром адресе в производном файле выдача каталога отдаёт рабочий адрес."""
    records = {"wine-1": {"name": "Вино", "winery": "Винодельня", "region": "Крым",
                          "category": "Красное", "grape": "Рислинг"}}
    media = {"wine-1": {"image_url": f"https://api.vino-svoe.ru/uploads/{FILE}"}}
    result = catalog_search(records, media, query="вино")
    assert "/v1/img/str-api/" in result["items"][0]["image_url"]


def test_ui_uses_single_image_wrapper() -> None:
    """Сырых изображений вне обёртки быть не должно."""
    for path in list((WEB / "components").glob("*.vue")) + list((WEB / "pages").glob("*.vue")):
        if path.name == "AppImage.vue":
            continue
        assert "<img " not in path.read_text(encoding="utf-8"), f"{path.name}: сырое изображение"


def test_image_wrapper_is_robust() -> None:
    component = (WEB / "components" / "AppImage.vue").read_text(encoding="utf-8")
    assert 'loading="lazy"' in component
    assert 'decoding="async"' in component
    assert 'referrerpolicy="no-referrer"' in component
    assert "@error" in component and "fallback" in component
    assert (WEB / "public" / "media" / "placeholder.svg").is_file()
