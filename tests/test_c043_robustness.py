# START_MODULE_CONTRACT
# PURPOSE: Устойчивость эндпоинтов к плохим входам и понятный отказ
# SCOPE: test_scan_response_has_new_fields_with_defaults, test_eval_predict_flat_contract_unchanged, test_empty_file_scan_400, test_empty_file_predict_400, test_non_image_scan_400, test_non_image_predict_400, test_oversized_file_rejected, test_scan_internal_error_is_degraded_not_500, test_scan_retrieval_unavailable_503, test_predict_internal_error_never_500, test_predict_retrieval_unavailable_503, test_no_label_gray_photo_returns_not_found_with_reason, test_empty_retrieval_returns_no_candidates_reason, test_multiple_bottles_low_scores_stay_uncertain, test_blurred_and_glared_photo_do_not_crash, test_ocr_unavailable_sets_reason_and_does_not_fail, test_recognized_fields_filled_from_ocr_and_empty_without, test_collect_recognized_fields_merges_and_normalizes, test_resolve_scan_reason_matrix
# DEPENDS: packages.contracts.schemas (ScanResponse), services.gateway.app (main), services.gateway.app.pipeline (_collect_recognized_fields, _resolve_scan_reason, run_scan)
# LINKS: M-GATEWAY, M-CONTRACTS
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# client - константа теста
# test_scan_response_has_new_fields_with_defaults - проверяет: scan response has new fields with defaults
# test_eval_predict_flat_contract_unchanged - Плоский контракт проверяем на подставном пайплайне: тест не должен зависеть от сети
# test_empty_file_scan_400 - проверяет: empty file scan 400
# test_empty_file_predict_400 - проверяет: empty file predict 400
# test_non_image_scan_400 - проверяет: non image scan 400
# test_non_image_predict_400 - проверяет: non image predict 400
# test_oversized_file_rejected - проверяет: oversized file rejected
# test_scan_internal_error_is_degraded_not_500 - проверяет: scan internal error is degraded not 500
# test_scan_retrieval_unavailable_503 - проверяет: scan retrieval unavailable 503
# test_predict_internal_error_never_500 - проверяет: predict internal error never 500
# test_predict_retrieval_unavailable_503 - проверяет: predict retrieval unavailable 503
# test_no_label_gray_photo_returns_not_found_with_reason - Однотонная сцена без этикетки: система не падает и говорит, что не нашла
# test_empty_retrieval_returns_no_candidates_reason - Каталог не дал ни одного кандидата: отказ без исключения
# test_multiple_bottles_low_scores_stay_uncertain - Несколько бутылок дают близкие слабые скоры: ответ остаётся отказом, но с кандидатами
# test_blurred_and_glared_photo_do_not_crash - Сильный блюр и блик: пайплайн отрабатывает до конца без исключений
# test_ocr_unavailable_sets_reason_and_does_not_fail - OCR-сервис недоступен: причина фиксируется, запрос не падает
# test_recognized_fields_filled_from_ocr_and_empty_without - Поля с этикетки заполняются при сигнале OCR и остаются пустыми без него
# test_collect_recognized_fields_merges_and_normalizes - проверяет: collect recognized fields merges and normalizes
# test_resolve_scan_reason_matrix - проверяет: resolve scan reason matrix
# END_MODULE_MAP
"""Гарантии устойчивости C-043: плохие входы не роняют сервис, отказ объясняется.

Тесты делятся на три группы:
- контракт: новые поля ответа и плоский контракт организатора;
- входы: пустой файл, не-изображение, слишком большой файл;
- деградация: недоступный OCR, недоступный retrieval, внутренняя ошибка.

Сценарии «не-вино, блюр, блики, несколько бутылок» проверяются на настоящем пайплайне
с подставными клиентами сервисов: сеть заменяется, логика решения остаётся боевой.
"""
from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageFilter

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from packages.contracts.schemas import ScanResponse  # noqa: E402
from services.gateway.app import main as gw  # noqa: E402
from services.gateway.app.pipeline import (  # noqa: E402
    _collect_recognized_fields,
    _resolve_scan_reason,
    run_scan,


)

__all__ = [
    "client",
    "test_scan_response_has_new_fields_with_defaults",
    "test_eval_predict_flat_contract_unchanged",
    "test_empty_file_scan_400",
    "test_empty_file_predict_400",
    "test_non_image_scan_400",
    "test_non_image_predict_400",
    "test_oversized_file_rejected",
    "test_scan_internal_error_is_degraded_not_500",
    "test_scan_retrieval_unavailable_503",
    "test_predict_internal_error_never_500",
    "test_predict_retrieval_unavailable_503",
    "test_no_label_gray_photo_returns_not_found_with_reason",
    "test_empty_retrieval_returns_no_candidates_reason",
    "test_multiple_bottles_low_scores_stay_uncertain",
    "test_blurred_and_glared_photo_do_not_crash",
    "test_ocr_unavailable_sets_reason_and_does_not_fail",
    "test_recognized_fields_filled_from_ocr_and_empty_without",
    "test_collect_recognized_fields_merges_and_normalizes",
    "test_resolve_scan_reason_matrix",
]

client = TestClient(gw.app)


def _jpeg(image: Image.Image) -> bytes:
    """Кодирует изображение в JPEG."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _noise(size: tuple[int, int] = (320, 320)) -> bytes:
    """Случайный шум: заведомо «не этикетка», но валидное изображение."""
    rng = np.random.default_rng(0)
    return _jpeg(Image.fromarray(rng.integers(0, 255, (*size, 3), dtype=np.uint8)))


def _gray(size: tuple[int, int] = (320, 320)) -> bytes:
    """Однотонная серая картинка: пустая сцена без этикетки."""
    rng = np.random.default_rng(1)
    data = np.full((*size, 3), 128, dtype=np.uint8)
    data[:20, :20] = rng.integers(0, 255, (20, 20, 3), dtype=np.uint8)
    return _jpeg(Image.fromarray(data))


class _FakeClients:
    """Подставные клиенты сервисов: сеть не участвует, логика решения боевая."""

    def __init__(self, items: list[dict] | None = None, ocr: dict | None = None,
                 ocr_available: bool = True, normalize: bool = True) -> None:
        self.items = items if items is not None else []
        self.ocr_result = ocr
        self.ocr_available = ocr_available
        self.normalize_available = normalize
        self.ocr_calls = 0

    async def normalize(self, image_bytes: bytes, filename: str = "query.jpg") -> dict | None:
        """Подготовка изображения: по умолчанию недоступна — работаем по исходному кадру."""
        if not self.normalize_available:
            return None
        return {"method": "identity", "crops": []}

    async def search(self, image_bytes: bytes, filename: str = "query.png") -> dict | None:
        """Поиск по индексу: отдаёт подготовленный список кандидатов."""
        return {"top_k": list(self.items)}

    async def ocr(self, image_bytes: bytes) -> dict | None:
        """Распознавание текста: None означает недоступность сервиса."""
        self.ocr_calls += 1
        return self.ocr_result if self.ocr_available else None

    async def rerank(self, items: list[dict], fields: dict) -> dict | None:
        """Переранжирование: в тестах не меняет порядок."""
        return {"items": items}

    async def extract(self, image_bytes: bytes, filename: str = "query.png") -> dict | None:
        """VLM выключен в настройках, поэтому всегда недоступен."""
        return None


def _run(image_bytes: bytes, clients: _FakeClients, store):
    """Прогоняет пайплайн на подставных клиентах."""
    return asyncio.run(run_scan(image_bytes, store, clients=clients, filename="q.jpg"))


# --- контракт -----------------------------------------------------------------

def test_scan_response_has_new_fields_with_defaults() -> None:
    response = ScanResponse(slug=None)
    assert response.recognized_fields == {}
    assert response.reason == ""
    assert response.degraded is False


def test_eval_predict_flat_contract_unchanged(monkeypatch) -> None:
    """Плоский контракт проверяем на подставном пайплайне: тест не должен зависеть от сети."""
    async def fake_scan(*args, **kwargs):
        return ScanResponse(slug="lesnaya-proseka", confidence=0.9, not_found=False)

    monkeypatch.setattr(gw, "run_scan", fake_scan)
    body = client.post("/v1/eval/predict", files={"image": ("q.jpg", _noise(), "image/jpeg")}).json()
    assert set(body.keys()) == {"slug"}
    assert body["slug"] == "lesnaya-proseka"


# --- входы --------------------------------------------------------------------

def test_empty_file_scan_400() -> None:
    response = client.post("/v1/scan", files={"image": ("empty.jpg", b"", "image/jpeg")})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_empty_file_predict_400() -> None:
    response = client.post("/v1/eval/predict", files={"image": ("empty.jpg", b"", "image/jpeg")})
    assert response.status_code == 400


def test_non_image_scan_400() -> None:
    response = client.post("/v1/scan", files={"image": ("fake.jpg", b"this is not an image", "image/jpeg")})
    assert response.status_code == 400
    assert "image" in response.json()["detail"].lower()


def test_non_image_predict_400() -> None:
    response = client.post("/v1/eval/predict", files={"image": ("fake.png", b"plain text", "image/png")})
    assert response.status_code == 400


def test_oversized_file_rejected(monkeypatch) -> None:
    monkeypatch.setattr(gw, "_MAX_IMAGE_BYTES", 1024)
    payload = b"\xff\xd8\xff" + b"0" * 4096
    assert client.post("/v1/scan", files={"image": ("big.jpg", payload, "image/jpeg")}).status_code == 400
    assert client.post("/v1/eval/predict", files={"image": ("big.jpg", payload, "image/jpeg")}).status_code == 400


# --- деградация ---------------------------------------------------------------

def test_scan_internal_error_is_degraded_not_500(monkeypatch) -> None:
    async def broken(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(gw, "run_scan", broken)
    response = client.post("/v1/scan", files={"image": ("q.jpg", _noise(), "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert body["degraded"] is True
    assert body["reason"] == "internal_error"
    assert body["not_found"] is True
    assert body["top5"] == []


def test_scan_retrieval_unavailable_503(monkeypatch) -> None:
    async def unavailable(*args, **kwargs):
        raise RuntimeError("retrieval service unavailable")

    monkeypatch.setattr(gw, "run_scan", unavailable)
    response = client.post("/v1/scan", files={"image": ("q.jpg", _noise(), "image/jpeg")})
    assert response.status_code == 503
    assert "retrieval" in response.json()["detail"]


def test_predict_internal_error_never_500(monkeypatch) -> None:
    async def broken(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(gw, "run_scan", broken)
    response = client.post("/v1/eval/predict", files={"image": ("q.jpg", _noise(), "image/jpeg")})
    assert response.status_code == 200
    assert response.json() == {"slug": None}


def test_predict_retrieval_unavailable_503(monkeypatch) -> None:
    async def unavailable(*args, **kwargs):
        raise RuntimeError("retrieval service unavailable")

    monkeypatch.setattr(gw, "run_scan", unavailable)
    response = client.post("/v1/eval/predict", files={"image": ("q.jpg", _noise(), "image/jpeg")})
    assert response.status_code == 503


# --- настоящий пайплайн с подставными клиентами --------------------------------

def test_no_label_gray_photo_returns_not_found_with_reason() -> None:
    """Однотонная сцена без этикетки: система не падает и говорит, что не нашла."""
    store = gw.get_store()
    clients = _FakeClients(items=[{"slug": "nebo-rossii-glera", "score": 0.21, "name": "Глера", "winery": "Небо России"}],
                           ocr={"ok": False, "reason": "no text"})
    response = _run(_gray(), clients, store)
    assert response.not_found is True
    assert response.slug  # система всё равно возвращает лучшую догадку, а не пустоту
    assert response.reason == "no_catalog_match"
    assert response.top5  # кандидаты остаются, чтобы UI мог их показать


def test_empty_retrieval_returns_no_candidates_reason() -> None:
    """Каталог не дал ни одного кандидата: отказ без исключения."""
    store = gw.get_store()
    response = _run(_noise(), _FakeClients(items=[]), store)
    assert response.not_found is True
    assert response.reason == "no_candidates"
    assert response.top5 == []
    assert response.recognized_fields == {}


def test_multiple_bottles_low_scores_stay_uncertain() -> None:
    """Несколько бутылок дают близкие слабые скоры: ответ остаётся отказом, но с кандидатами."""
    store = gw.get_store()
    items = [{"slug": "nebo-rossii-glera", "score": 0.30, "name": "Глера", "winery": "Небо России"},
             {"slug": "lesnaya-proseka", "score": 0.29, "name": "Лесная просека", "winery": "Лесная просека"},
             {"slug": "valeriy-zaharin-avtohtonnoe-vino-kryma-ot-valeriya-zaharina-aleatiko-kefesiya-rozovoe-suhoe-11",
              "score": 0.28, "name": "Алеатико", "winery": "Валерий Захарин"}]
    response = _run(_noise(), _FakeClients(items=items), store)
    assert response.not_found is True
    assert len(response.top5) == 3
    assert response.confident is False


def test_blurred_and_glared_photo_do_not_crash() -> None:
    """Сильный блюр и блик: пайплайн отрабатывает до конца без исключений."""
    store = gw.get_store()
    image = Image.open(io.BytesIO(_gray())).filter(ImageFilter.GaussianBlur(radius=14))
    glare = np.asarray(image).copy()
    glare[:80, :80] = 255  # засветка в углу
    payload = _jpeg(Image.fromarray(glare))
    response = _run(payload, _FakeClients(items=[{"slug": "lesnaya-proseka", "score": 0.24,
                                                 "name": "Лесная просека", "winery": "Лесная просека"}],
                                          ocr={"ok": False, "reason": "no text"}), store)
    assert response.not_found is True
    assert response.reason in {"no_catalog_match", "low_confidence"}


def test_ocr_unavailable_sets_reason_and_does_not_fail() -> None:
    """OCR-сервис недоступен: причина фиксируется, запрос не падает."""
    store = gw.get_store()
    clients = _FakeClients(items=[{"slug": "nebo-rossii-glera", "score": 0.22,
                                   "name": "Глера", "winery": "Небо России"}], ocr_available=False)
    response = _run(_gray(), clients, store)
    assert clients.ocr_calls == 1  # OCR вызывался и вернул недоступность
    assert response.reason == "ocr_unavailable"
    assert response.not_found is True


def test_recognized_fields_filled_from_ocr_and_empty_without() -> None:
    """Поля с этикетки заполняются при сигнале OCR и остаются пустыми без него."""
    store = gw.get_store()
    ocr = {"ok": True, "fields": {"winery": "Массандра", "wine_name": "Мускатель белый",
                                  "vintage": "2023", "grape": "Мускатель", "sweetness": "сладкое"}}
    clients = _FakeClients(items=[{"slug": "massandra-muskatel-belyy-belye-sorta-vin", "score": 0.30,
                                   "name": "Мускатель белый", "winery": "Массандра"}],
                           ocr=ocr)
    with_ocr = _run(_gray(), clients, store)
    assert with_ocr.recognized_fields.get("winery") == "Массандра"
    assert str(with_ocr.recognized_fields.get("vintage")) in {"2023", "23"}
    empty = _run(_gray(), _FakeClients(items=[{"slug": "lesnaya-proseka", "score": 0.2,
                                               "name": "Лесная просека", "winery": "Лесная просека"}],
                                       ocr={"ok": False, "fields": {}}), store)
    assert empty.recognized_fields == {}


# --- чистые помощники ----------------------------------------------------------

def test_collect_recognized_fields_merges_and_normalizes() -> None:
    ocr = {"ok": True, "fields": {"winery": "Массандра", "vintage": "2023", "extra": "мусор", "region": "  "}}
    vlm = {"ok": True, "fields": {"wine_name": "Мускатель", "region": "Крым", "vintage": "2024"}}
    merged = _collect_recognized_fields(ocr, vlm)
    assert merged["winery"] == "Массандра"
    assert merged["wine_name"] == "Мускатель"
    assert merged["region"] == "Крым"  # VLM уточняет OCR
    assert merged["vintage"] == 2024  # значение приводится к числу
    assert "extra" not in merged
    assert _collect_recognized_fields(None, None) == {}


@pytest.mark.parametrize(("not_found", "confident", "fields", "candidates", "expected"), [
    (True, False, {}, 0, "no_candidates"),
    (True, False, {}, 3, "no_catalog_match"),
    (True, False, {"winery": "Массандра"}, 3, "no_catalog_match"),
    (False, False, {"winery": "Массандра"}, 3, "low_confidence"),
    (False, True, {}, 3, ""),
])
def test_resolve_scan_reason_matrix(not_found, confident, fields, candidates, expected) -> None:
    assert _resolve_scan_reason(not_found, confident, fields, candidates) == expected
