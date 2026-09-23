"""Тесты C-053: консенсус ридеров и политика разбиения — на моках, без запуска моделей.

Проверяется чистая логика: единогласие, согласие двух из трёх с поддержкой каталога, несовместимые
показания, ничья, конфликт винодельни и названия, а также защита датасета от утечки.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from packages.common.annotation import (  # noqa: E402
    ACCEPT_AGREEMENT,
    READER_WEIGHTS,
    adjudicate,
    assign_split,
    cluster_values,
    field_consensus,
    judge_payload,
    split_policy,
)

# Кандидаты каталога: запись, которую подтверждают ридеры
MASSANDRA = {"slug": "massandra-muskatel", "name": "Мускатель Массандра белый",
             "winery": "Массандра", "grape": "Мускатель", "year": 2023}

UNANIMOUS = {
    "vlm_primary": {"winery": "Массандра", "wine_name": "Мускатель Массандра белый", "vintage": 2023,
                    "grape": "Мускатель", "sweetness": "сладкое", "color": "белое"},
    "ocr_fusion": {"winery": "Массандра", "wine_name": "МУСКАТЕЛЬ МАССАНДРА БЕЛЫЙ", "vintage": "2023",
                   "sweetness": "сладкое"},
    "vlm_small": {"winery": "Массандра", "wine_name": "Мускатель Массандра", "vintage": 2023,
                  "grape": "Мускатель", "color": "белое"},
}


def test_cluster_values_merges_similar_and_skips_empty() -> None:
    clusters = cluster_values([("a", "Пино Нуар"), ("b", "пино-нуар"), ("c", ""), ("d", "Рислинг")])
    assert len(clusters) == 2
    assert clusters[0]["readers"] == ["a", "b"]


def test_field_consensus_weights_readers() -> None:
    result = field_consensus({"vlm_primary": {"winery": "Массандра"},
                              "ocr_fusion": {"winery": "Массандра"},
                              "vlm_small": {"winery": "Массандра"}}, "winery")
    assert result["value"] == "Массандра"
    assert result["weight"] == pytest.approx(sum(READER_WEIGHTS.values()), abs=1e-6)
    assert result["tie"] is False


def test_unanimous_readers_are_accepted() -> None:
    decision = adjudicate(UNANIMOUS, [{"record": MASSANDRA}])
    assert decision["verdict"] == "accept"
    assert decision["label"]["winery"] == "Массандра"
    assert decision["confidence"] > 0.5
    assert decision["reasons"]


def test_two_of_three_with_catalog_support_are_accepted() -> None:
    """Слабая модель ошибается, но сильная и OCR согласны, и каталог это подтверждает."""
    readings = {
        "vlm_primary": {"winery": "Массандра", "wine_name": "Мускатель Массандра белый", "vintage": 2023},
        "ocr_fusion": {"winery": "Массандра", "wine_name": "МУСКАТЕЛЬ МАССАНДРА", "vintage": "2023"},
        "vlm_small": {"winery": "Инкерман", "wine_name": "Рислинг", "vintage": 2021},
    }
    decision = adjudicate(readings, [{"record": MASSANDRA}])
    assert decision["verdict"] == "accept"
    assert decision["agreement"] >= ACCEPT_AGREEMENT


def test_incompatible_readings_are_escalated() -> None:
    readings = {
        "vlm_primary": {"winery": "Массандра", "wine_name": "Мускатель", "vintage": 2023},
        "ocr_fusion": {"winery": "Инкерман", "wine_name": "Каберне", "vintage": 2019},
        "vlm_small": {"winery": "Абрау-Дюрсо", "wine_name": "Брют", "vintage": 2021},
    }
    decision = adjudicate(readings, [{"record": MASSANDRA}])
    assert decision["verdict"] == "escalate"
    assert decision["agreement"] < ACCEPT_AGREEMENT
    assert decision["reasons"]


def test_tie_is_escalated_even_with_catalog_match() -> None:
    """Ничья между двумя сильными показаниями не разрешается автоматически."""
    # Ничья возможна только при равном весе голосов: 0.5 против 0.3 + 0.2.
    readings = {
        "vlm_primary": {"winery": "Массандра"},
        "ocr_fusion": {"winery": "Инкерман"},
        "vlm_small": {"winery": "Инкерман"},
    }
    decision = adjudicate(readings, [{"record": MASSANDRA}])
    assert decision["verdict"] == "escalate"
    assert any("ничья" in reason for reason in decision["reasons"])


def test_catalog_contradiction_blocks_acceptance() -> None:
    """Ридеры согласны между собой, но каталог такого вина не знает — принимает человек."""
    readings = {"vlm_primary": {"winery": "Неизвестная винодельня", "wine_name": "Странное вино"},
                "ocr_fusion": {"winery": "Неизвестная винодельня", "wine_name": "Странное вино"},
                "vlm_small": {"winery": "Неизвестная винодельня", "wine_name": "Странное вино"}}
    decision = adjudicate(readings, [{"record": MASSANDRA}])
    assert decision["verdict"] == "escalate"
    assert any("каталог не подтверждает" in reason for reason in decision["reasons"])


def test_name_pointing_to_other_winery_is_conflict() -> None:
    """Название совпало с винодельней каталога: значит этикетка прочитана неверно."""
    readings = {"vlm_primary": {"winery": "Массандра", "wine_name": "Массандра"},
                "ocr_fusion": {"winery": "Массандра", "wine_name": "Массандра"},
                "vlm_small": {"winery": "Массандра", "wine_name": "Массандра"}}
    decision = adjudicate(readings, [{"record": MASSANDRA}])
    assert any("другую винодельню" in reason for reason in decision["reasons"])
    assert judge_payload(decision)["verdict"] == "fail"


def test_empty_readings_are_escalated() -> None:
    decision = adjudicate({}, [])
    assert decision["verdict"] == "escalate"
    assert decision["confidence"] == 0.0


def test_judge_payload_matches_judge_schema() -> None:
    """Вердикт представляется в форме схемы судьи и проходит её проверку."""
    from packages.common.llm.structured import parse_structured
    import json

    accepted = judge_payload(adjudicate(UNANIMOUS, [{"record": MASSANDRA}]))
    assert accepted["verdict"] == "pass" and 0.0 <= accepted["score"] <= 1.0
    validated = parse_structured(json.dumps(accepted, ensure_ascii=False), "judge")
    assert validated["valid"] is True, validated["errors"]


def test_weights_are_configurable() -> None:
    """Веса ридеров настраиваются: с другим весом OCR меняется победитель."""
    readings = {"vlm_primary": {"winery": "Массандра"}, "ocr_fusion": {"winery": "Инкерман"}}
    default = field_consensus(readings, "winery")
    ocr_heavy = field_consensus(readings, "winery", {"vlm_primary": 0.1, "ocr_fusion": 0.9})
    assert default["value"] == "Массандра"
    assert ocr_heavy["value"] == "Инкерман"


# --- политика разбиения и защита от утечки --------------------------------------

def test_accepted_real_eval_photo_stays_eval() -> None:
    """Фото оценочного набора не попадают в обучение даже при полном согласии."""
    assert assign_split("real_eval", escalated=False) == "eval"


def test_escalated_case_goes_to_holdout() -> None:
    assert assign_split("real_eval", escalated=True) == "holdout"
    assert assign_split("train", escalated=True) == "holdout"


def test_synthetic_inherits_source_split() -> None:
    """Аугментация оценочного фото остаётся оценочной: иначе копии просочились бы в обучение."""
    assert assign_split("synthetic", escalated=False, base_split="eval") == "eval"
    assert assign_split("synthetic", escalated=False, base_split="holdout") == "holdout"
    assert assign_split("synthetic", escalated=False, base_split="train") == "train"


def test_train_only_from_train_origins() -> None:
    assert assign_split("train", escalated=False) == "train"
    assert assign_split("catalog", escalated=False) == "train"
    assert assign_split("real_eval", escalated=False) == "eval"


def test_split_policy_is_documented() -> None:
    policy = split_policy()
    assert "holdout" in policy and "наследуют" in policy and "train" in policy
