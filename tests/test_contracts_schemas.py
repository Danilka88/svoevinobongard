# START_MODULE_CONTRACT
# PURPOSE: Верификация контрактов API: rich-ответ, плоский контракт организатора, контракт ошибки
# SCOPE: test_scan_response_defaults, test_eval_predict_flat_contract, test_error_contract
# DEPENDS: packages.contracts.schemas
# LINKS: M-CONTRACTS
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# test_scan_response_defaults - ScanResponse содержит все поля контракта
# test_eval_predict_flat_contract - EvalPredictResponse сериализуется только в slug
# test_error_contract - ErrorResponse требует error и detail
# END_MODULE_MAP
"""V-M-CONTRACTS: единые схемы API."""
from __future__ import annotations

from packages.contracts.schemas import (
    Card,
    EvalPredictResponse,
    ErrorResponse,
    ScanResponse,
    TopKItem,
)

__all__ = [
    "test_scan_response_defaults",
    "test_eval_predict_flat_contract",
    "test_error_contract",
]


def test_scan_response_defaults() -> None:
    response = ScanResponse(slug="kokur-suhoe-2025", confidence=0.87,
                            top5=[TopKItem(slug="kokur-suhoe-2025", score=0.87)],
                            card=Card(name="Кокур Сухое"))
    payload = response.model_dump()
    for field in ("slug", "confidence", "f1_top1", "f1_top5", "margin", "path",
                  "latency_ms", "top5", "not_found", "card"):
        assert field in payload, field
    assert payload["path"] == "retrieval"
    assert payload["top5"][0]["slug"] == "kokur-suhoe-2025"


def test_eval_predict_flat_contract() -> None:
    assert EvalPredictResponse(slug="kokur-suhoe-2025").model_dump() == {"slug": "kokur-suhoe-2025"}
    assert EvalPredictResponse(slug=None).model_dump() == {"slug": None}


def test_error_contract() -> None:
    error = ErrorResponse(error="not_found", detail="unknown slug")
    assert error.model_dump() == {"error": "not_found", "detail": "unknown slug"}
