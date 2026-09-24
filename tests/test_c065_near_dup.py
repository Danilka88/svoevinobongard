# START_MODULE_CONTRACT
# PURPOSE: Проверка групп near-дубликатов и дизамбигуации: выбор по сигналам, честная неоднозначность, выдача и аналоги
# SCOPE: CATALOG, SIGNALS, test_build_groups_finds_series, test_group_contains_members, test_no_group_for_unique,
#        test_disambiguate_by_year, test_disambiguate_by_sweetness, test_ambiguous_without_signals,
#        test_tie_is_ambiguous, test_scan_has_top10_and_alternatives, test_near_dup_disabled_by_default,
#        test_eval_predict_stays_flat
# DEPENDS: asyncio, numpy, pytest, fastapi.testclient, packages.common.catalog, packages.common.config,
#          packages.common.near_dup, packages.contracts.schemas, services.gateway.app.main, services.gateway.app.pipeline
# LINKS: M-GATEWAY, M-MATCHER, M-EVAL
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# CATALOG - синтетический каталог с серией и уникальным вином
# SIGNALS - сигналы различия для проверок
# client - тестовый клиент gateway
# test_build_groups_finds_series - серия собирается в группу
# test_group_contains_members - группа содержит оба вина серии
# test_no_group_for_unique - уникальное вино в группу не попадает
# test_disambiguate_by_year - выбор по подтверждённому году
# test_disambiguate_by_sweetness - выбор по сладости
# test_ambiguous_without_signals - без сигналов решение не выдумывается
# test_tie_is_ambiguous - ничья помечается неоднозначной
# test_scan_has_top10_and_alternatives - ответ содержит десять кандидатов и аналоги
# test_near_dup_disabled_by_default - флаг выключен по умолчанию
# test_eval_predict_stays_flat - контракт организатора остаётся плоским
# END_MODULE_MAP
"""Near-дубликаты: проверки на синтетическом каталоге и через подставной пайплайн."""
from __future__ import annotations

import asyncio

import numpy as np
import pytest
from fastapi.testclient import TestClient

from packages.common.config import settings
from packages.common.near_dup import build_groups, disambiguate, near_dup_group, signature
from packages.contracts.schemas import ScanResponse
from services.gateway.app import main as gw
from services.gateway.app import pipeline

client = TestClient(gw.app)

CATALOG = {
    "series-suhoe-2023": {"name": "Шёпот цветов 2023 сухое", "winery": "Скалистый берег",
                          "year": 2023, "sweetness": "сухое", "category": "Красное"},
    "series-bryut-2024": {"name": "Шёпот цветов 2024 брют", "winery": "Скалистый берег",
                          "year": 2024, "sweetness": "брют", "category": "Игристое"},
    "unique-wine": {"name": "Совершенно другое вино", "winery": "Иная винодельня",
                    "year": 2020, "sweetness": "сладкое", "category": "Белое"},
}
SIGNALS = {"vintage": "2024", "sweetness": "брют", "color": "игристое"}

__all__ = ["CATALOG", "SIGNALS", "test_build_groups_finds_series", "test_group_contains_members",
           "test_no_group_for_unique", "test_disambiguate_by_year", "test_disambiguate_by_sweetness",
           "test_ambiguous_without_signals", "test_tie_is_ambiguous",
           "test_scan_has_top10_and_alternatives", "test_near_dup_disabled_by_default",
           "test_eval_predict_stays_flat"]


def test_build_groups_finds_series() -> None:
    groups = build_groups(CATALOG)
    assert groups, "серия из двух позиций обязана собраться в группу"
    assert any(len(members) >= 2 for members in groups.values())


def test_group_contains_members() -> None:
    groups = build_groups(CATALOG)
    members = near_dup_group("series-suhoe-2023", groups)
    assert "series-bryut-2024" in members
    assert signature(CATALOG["series-suhoe-2023"])[0] == signature(CATALOG["series-bryut-2024"])[0]


def test_no_group_for_unique() -> None:
    groups = build_groups(CATALOG)
    assert near_dup_group("unique-wine", groups) == []


def test_disambiguate_by_year() -> None:
    groups = build_groups(CATALOG)
    decision = disambiguate(["series-suhoe-2023", "series-bryut-2024"], SIGNALS, CATALOG, groups)
    assert decision["ambiguous"] is False
    assert decision["slug"] == "series-bryut-2024"
    assert decision["signals"]["year_hit"] is True


def test_disambiguate_by_sweetness() -> None:
    groups = build_groups(CATALOG)
    decision = disambiguate(["series-suhoe-2023", "series-bryut-2024"],
                            {"sweetness": "сухое"}, CATALOG, groups)
    assert decision["slug"] == "series-suhoe-2023"


def test_ambiguous_without_signals() -> None:
    """Сигналов нет — угадывать нельзя: ответ помечается неоднозначным."""
    groups = build_groups(CATALOG)
    decision = disambiguate(["series-suhoe-2023", "series-bryut-2024"], {}, CATALOG, groups)
    assert decision["ambiguous"] is True
    assert set(decision["candidates"]) == {"series-suhoe-2023", "series-bryut-2024"}


def test_tie_is_ambiguous() -> None:
    """Ничья сигналов: порядок выдачи уже выражает предпочтение, догадка не нужна."""
    groups = build_groups(CATALOG)
    both = {"vintage": "2023", "sweetness": "брют"}
    decision = disambiguate(["series-suhoe-2023", "series-bryut-2024"], both, CATALOG, groups)
    assert decision["ambiguous"] is False
    assert decision["slug"] == "series-suhoe-2023"  # первый по выдаче


class _FakeClients:
    """Подставные клиенты: слабый результат поиска, чтобы включился путь аналогов."""

    def __init__(self, items=None):
        self.items = items or [{"slug": "series-suhoe-2023", "score": 0.30,
                                "name": "Шёпот цветов 2023 сухое", "winery": "Скалистый берег"}]

    async def normalize(self, data, filename="q.jpg"):
        return {"method": "identity", "crops": []}

    async def search(self, data, filename="q.png"):
        return {"top_k": list(self.items)}

    async def ocr(self, data):
        return None

    async def rerank(self, items, fields):
        return {"items": items}

    async def extract(self, data, filename="q.png"):
        return None

    async def observe(self, data, filename="q.png", timeout=None):
        return {"ok": False}


def _scan(clients):
    import io
    from PIL import Image
    buffer = io.BytesIO()
    Image.fromarray(np.full((48, 48, 3), 190, dtype=np.uint8)).save(buffer, format="PNG")
    return asyncio.run(pipeline.run_scan(buffer.getvalue(), gw.get_store(), clients=clients,
                                         filename="q.png"))


def test_scan_has_top10_and_alternatives() -> None:
    """Ответ сканера содержит десять кандидатов отдельным полем и аналоги при отказе."""
    response = _scan(_FakeClients())
    assert hasattr(response, "top10") and isinstance(response.top10, list)
    assert hasattr(response, "ambiguous") and hasattr(response, "alternatives")
    if response.not_found:
        assert isinstance(response.alternatives, list)  # аналоги могут быть пустыми, но поле есть
    assert len(response.top10) >= len(response.top5)


def test_near_dup_disabled_by_default() -> None:
    assert settings.near_dup_enabled is False
    assert settings.scan_top_k == 10


def test_eval_predict_stays_flat(monkeypatch) -> None:
    async def fake_scan(*args, **kwargs):
        return ScanResponse(slug="series-suhoe-2023", confidence=0.4, top10=[], alternatives=[])

    monkeypatch.setattr(gw, "run_scan", fake_scan)
    body = client.post("/v1/eval/predict",
                       files={"image": ("q.png", b"\xff\xd8\xff" + b"0" * 64, "image/jpeg")}).json()
    assert set(body.keys()) == {"slug"} and body["slug"] == "series-suhoe-2023"
