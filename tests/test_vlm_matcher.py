# START_MODULE_CONTRACT
# PURPOSE: Верификация VLM-извлечения (ремонт JSON, finish_reason, деградация) и matcher-фьюжна с дизамбигуацией
# SCOPE: test_parse_fields_clean, test_parse_fields_repairs_fences_and_noise, test_parse_fields_clamps_confidence, test_extract_fields_degrades_on_error, test_extract_fields_rejects_empty_content, test_vintage_hit, test_sweetness_and_color_hit, test_rerank_prefers_vintage_match, test_rerank_degrades_without_fields
# DEPENDS: services.vlm.app.extract, services.vlm.app.prompt, services.matcher.app.merge, packages.common.catalog
# LINKS: M-VLM, M-MATCHER
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# test_parse_fields_clean - чистый JSON разбирается в поля
# test_parse_fields_repairs_fences_and_noise - markdown-обертка и мусор вокруг JSON ремонтируются
# test_parse_fields_clamps_confidence - confidence приводится к диапазону 0..1
# test_extract_fields_degrades_on_error - отказ LLM дает ok=false с причиной, без исключения
# test_extract_fields_rejects_empty_content - пустой content при finish_reason=length не считается валидным
# test_extract_fields_ok_and_json_object - успешное извлечение и передача reasoning_effort=none
# test_prompt_encodes_image - изображение кодируется в data URL, сообщения собираются корректно
# test_vintage_hit - год с этикетки сопоставляется с годом каталога, названием и слагом
# test_sweetness_and_color_hit - сладость и цвет сопоставляются с метаданными каталога
# test_rerank_prefers_vintage_match - при равном базовом скоре выигрывает кандидат с верным годом
# test_rerank_degrades_without_fields - без полей VLM порядок сохраняется
# END_MODULE_MAP
"""V-M-VLM и V-M-MATCHER: поля этикетки и фьюжн."""
from __future__ import annotations

import numpy as np
import pytest

from packages.common.catalog import CatalogStore
from packages.common.llm.adapters import BaseAdapter
from packages.common.llm.errors import ProviderError
from packages.common.llm.config import ProviderConfig
from services.matcher.app.merge import rerank_candidates, sweetness_hit, vintage_hit
from services.vlm.app.extract import extract_fields, parse_fields
from services.vlm.app.prompt import build_messages, encode_image_data_url

__all__ = [
    "test_parse_fields_clean",
    "test_parse_fields_repairs_fences_and_noise",
    "test_parse_fields_clamps_confidence",
    "test_extract_fields_degrades_on_error",
    "test_extract_fields_rejects_empty_content",
    "test_extract_fields_ok_and_json_object",
    "test_prompt_encodes_image",
    "test_vintage_hit",
    "test_sweetness_and_color_hit",
    "test_rerank_prefers_vintage_match",
    "test_rerank_degrades_without_fields",
]

_RECORDS = [
    {"slug": "kokur-suhoe-2025", "name": "Кокур Сухое, 2025", "winery": "Валерий Захарьин",
     "category": "Белое", "color": "белое", "sweetness": "сухое", "year": 2025,
     "region": "Крым", "grape": "Кокур"},
    {"slug": "kokur-2024", "name": "Кокур, 2024", "winery": "Валерий Захарьин",
     "category": "Белое", "color": "белое", "sweetness": "полусладкое", "year": 2024,
     "region": "Крым", "grape": "Кокур"},
]


class _StubLLM(BaseAdapter):
    """Адаптер с заданным ответом или ошибкой."""

    def __init__(self, text: str = "", error: Exception | None = None, finish: str = "stop") -> None:
        super().__init__(ProviderConfig(name="stub", base_url="http://stub", api_key="s", local=True))
        self.text = text
        self.error = error
        self.finish = finish
        self.calls = 0

    def chat(self, model: str, messages: list[dict], *, max_tokens: int = 512,
             temperature: float = 0.0, response_format: dict | None = None,
             extra: dict | None = None, timeout_s: float | None = None):
        from packages.common.llm.adapters import ChatResult

        self.calls += 1
        self.last_extra = extra
        if self.error:
            raise self.error
        return ChatResult(text=self.text, model=model, provider=self.name, total_tokens=10,
                          latency_ms=5.0, finish_reason=self.finish)


class _StubRouter:
    """Роутер, отдающий заранее созданный адаптер."""

    def __init__(self, adapter: BaseAdapter) -> None:
        self.adapter = adapter

    def chat(self, messages, **kwargs):
        return self.adapter.chat(kwargs.get("model") or "stub", messages,
                                 max_tokens=kwargs.get("max_tokens", 128),
                                 temperature=0.0,
                                 response_format=kwargs.get("response_format"),
                                 extra=kwargs.get("extra"),
                                 timeout_s=kwargs.get("timeout_s"))


def _store() -> CatalogStore:
    return CatalogStore({record["slug"]: record for record in _RECORDS})


def test_parse_fields_clean() -> None:
    fields = parse_fields('{"winery":"Кокур","wine_name":"Сухое","vintage":"2025",'
                          '"sweetness":"сухое","region":"Крым","grape":"Кокур","confidence":0.8}')
    assert fields["winery"] == "Кокур"
    assert fields["vintage"] == "2025"
    assert fields["confidence"] == pytest.approx(0.8)


def test_parse_fields_repairs_fences_and_noise() -> None:
    text = 'Думал долго.\n```json\n{"winery": "Захарьин", "vintage": 2025, "grape": null}\n```\nГотово.'
    fields = parse_fields(text)
    assert fields["winery"] == "Захарьин"
    assert fields["vintage"] == "2025"
    assert "grape" not in fields
    assert parse_fields("совсем не json") == {}


def test_parse_fields_clamps_confidence() -> None:
    assert parse_fields('{"confidence": 5}')["confidence"] == 1.0
    assert parse_fields('{"confidence": -3}')["confidence"] == 0.0
    assert "confidence" not in parse_fields('{"confidence": "много"}')


def test_extract_fields_degrades_on_error() -> None:
    router = _StubRouter(_StubLLM(error=ProviderError("stub", "m", "boom", retryable=True)))
    image = np.full((64, 64, 3), 200, dtype=np.uint8)
    fields = extract_fields([image], router)
    assert fields.ok is False
    assert "ProviderError" in fields.reason


def test_extract_fields_rejects_empty_content() -> None:
    router = _StubRouter(_StubLLM(text="", finish="length"))
    fields = extract_fields([np.full((64, 64, 3), 200, dtype=np.uint8)], router)
    assert fields.ok is False
    assert fields.reason == "empty content"
    assert fields.finish_reason == "length"


def test_extract_fields_ok_and_json_object() -> None:
    router = _StubRouter(_StubLLM(text='{"winery":"Захарьин","vintage":"2025","confidence":0.7}'))
    fields = extract_fields([np.full((64, 64, 3), 200, dtype=np.uint8)], router)
    assert fields.ok is True
    assert fields.vintage == "2025"
    assert fields.as_dict()["winery"] == "Захарьин"
    assert router.adapter.last_extra["reasoning_effort"] == "none"
    assert router.adapter.last_extra["options"]["num_ctx"] == 8192


def test_prompt_encodes_image() -> None:
    url = encode_image_data_url(np.full((900, 600, 3), 128, dtype=np.uint8), max_side=384)
    assert url.startswith("data:image/jpeg;base64,")
    messages = build_messages([np.full((64, 64, 3), 128, dtype=np.uint8)])
    assert messages[0]["content"][0]["type"] == "text"
    assert messages[0]["content"][1]["type"] == "image_url"


def test_vintage_hit() -> None:
    assert vintage_hit("2025", _RECORDS[0]) is True
    assert vintage_hit("2024", _RECORDS[0]) is False
    assert vintage_hit("", _RECORDS[0]) is False


def test_sweetness_and_color_hit() -> None:
    assert sweetness_hit("белое сухое", _RECORDS[0]) is True
    assert sweetness_hit("белое полусладкое", _RECORDS[0]) is False
    assert sweetness_hit("белое полусладкое", _RECORDS[1]) is True
    assert sweetness_hit("", _RECORDS[0]) is False


def test_rerank_prefers_vintage_match() -> None:
    candidates = [
        {"slug": "kokur-2024", "score": 0.70},
        {"slug": "kokur-suhoe-2025", "score": 0.69},
    ]
    fields = {"winery": "Валерий Захарьин", "wine_name": "Кокур Сухое", "vintage": "2025",
              "wine_type": "белое сухое"}
    reranked = rerank_candidates(candidates, fields, _store())
    assert reranked[0].slug == "kokur-suhoe-2025"
    assert reranked[0].vintage_match is True
    assert reranked[0].sweetness_match is True
    assert reranked[1].vintage_match is False


def test_rerank_degrades_without_fields() -> None:
    candidates = [{"slug": "kokur-2024", "score": 0.70}, {"slug": "kokur-suhoe-2025", "score": 0.69}]
    reranked = rerank_candidates(candidates, {}, _store())
    assert [item.slug for item in reranked] == ["kokur-2024", "kokur-suhoe-2025"]
    assert all(item.text_score == 0.0 for item in reranked)
