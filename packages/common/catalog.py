# START_MODULE_CONTRACT
# PURPOSE: Read-only доступ к каталогу: чтение catalog.jsonl и метаданные позиции по slug
# SCOPE: load_catalog, load_official_v3, CatalogStore
# DEPENDS: none (stdlib json)
# LINKS: M-COMMON
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# load_catalog - чтение catalog.jsonl в словарь slug -> запись
# load_official_v3 - загрузка каталога, обогащённого официальным API (C-017)
# CatalogStore - read-only доступ к каталогу: get, card, summary, slugs_with_image
# END_MODULE_MAP
"""Read-only каталог для сервисов M1.

Источник — ``data/catalog/catalog.jsonl`` (2103 slug), созданный ETL C-001.
Размещено в shared-пакете, потому что каталог читают И retrieval (метаданные
выдачи), И gateway (карточка вина): сервисы не импортируют друг друга, но оба
могут использовать общую библиотеку.

Prod-цель — PostgreSQL + pgvector (см. ``services/catalog``); при недоступном
Docker в M1 используется файловый каталог + FAISS/numpy-индекс.
"""
from __future__ import annotations

import json
from pathlib import Path

from packages.common.logging import log_marker

__all__ = ["load_catalog", "load_official_v3", "CatalogStore"]

CARD_FIELDS = (
    "name", "winery", "region", "category", "color",
    "grape", "description", "year", "sweetness", "image",
)


def load_catalog(path: str | Path) -> dict[str, dict]:
    """Читает catalog.jsonl и возвращает отображение slug → запись.

    Raises:
        FileNotFoundError: если файл каталога отсутствует.
    """
    catalog_path = Path(path)
    if not catalog_path.is_file():
        raise FileNotFoundError(f"catalog not found: {catalog_path}")
    records: dict[str, dict] = {}
    with catalog_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            slug = record.get("slug")
            if slug:
                records[slug] = record
    return records


class CatalogStore:
    """Read-only доступ к метаданным каталога."""

    def __init__(self, records: dict[str, dict]) -> None:
        self._records = records

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "CatalogStore":
        """Создает store из файла catalog.jsonl."""
        return cls(load_catalog(path))

    def __len__(self) -> int:
        return len(self._records)

    def get(self, slug: str) -> dict | None:
        """Возвращает запись каталога или None."""
        return self._records.get(slug)

    def card(self, slug: str) -> dict:
        """Возвращает поля карточки для slug (пустой dict, если slug неизвестен)."""
        record = self._records.get(slug)
        if not record:
            return {}
        return {field: record.get(field) for field in CARD_FIELDS}

    def summary(self, slug: str) -> dict:
        """Короткая сводка позиции для ответа поиска."""
        record = self._records.get(slug) or {}
        return {
            "slug": slug,
            "name": record.get("name", ""),
            "winery": record.get("winery", ""),
            "region": record.get("region", ""),
            "year": record.get("year"),
        }

    def slugs_with_image(self) -> list[str]:
        """Список slug, у которых есть эталонное фото."""
        return [slug for slug, record in self._records.items() if record.get("image")]

    def slug_to_record(self) -> dict[str, dict]:
        """Отображение slug → запись (read-only, без копирования)."""
        return self._records


def load_official_v3(path: str | Path | None = None) -> dict[str, dict]:
    """Загружает обогащённый официальными данными каталог (C-017) как индекс slug -> запись.

    Отсутствие файла — не ошибка: карточка просто останется на данных CSV.
    """
    target = Path(path or "data/catalog/catalog_enriched_v3.jsonl")
    if not target.is_file():
        log_marker("M-CATALOG", "load_v3", "SKIP", reason="file-missing", path=str(target))
        return {}
    index: dict[str, dict] = {}
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            slug = str(record.get("slug") or "")
            if slug:
                index[slug] = record
    log_marker("M-CATALOG", "load_v3", "DONE", slugs=len(index))
    return index
