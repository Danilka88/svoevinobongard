# START_MODULE_CONTRACT
# PURPOSE: Тесты C-076 контракта витрины: аддитивное поле normalization в ScanResponse
# SCOPE: test_scan_response_has_normalization, test_normalization_defaults_empty,
#        test_scan_response_accepts_normalization
# DEPENDS: packages.contracts.schemas
# LINKS: M-CONTRACTS, M-GATEWAY
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# test_scan_response_has_normalization - поле normalization присутствует в схеме ответа
# test_normalization_defaults_empty - по умолчанию поле пустое (обратная совместимость)
# test_scan_response_accepts_normalization - поле принимает результат нормализации
# END_MODULE_MAP
"""Тесты контракта витрины C-076."""
from __future__ import annotations

from packages.contracts.schemas import ScanResponse

__all__ = ["test_normalization_defaults_empty", "test_scan_response_accepts_normalization",
           "test_scan_response_has_normalization"]


def test_scan_response_has_normalization() -> None:
    """Поле normalization объявлено в схеме (аддитивно, контракт не ломается)."""
    assert "normalization" in ScanResponse.model_fields


def test_normalization_defaults_empty() -> None:
    """По умолчанию поле пустое."""
    response = ScanResponse(slug=None)
    assert response.normalization == {}


def test_scan_response_accepts_normalization() -> None:
    """Ответ принимает результат нормализации чтения под каталог."""
    response = ScanResponse(slug=None, normalization={"winery": "Кубань-Вино", "brand": "Шато Тамань",
                                                      "source": "llm"})
    assert response.normalization["winery"] == "Кубань-Вино"
    assert response.normalization["source"] == "llm"
