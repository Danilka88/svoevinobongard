# services/catalog

Хранение каталога и доступ к метаданным.

- ETL: `scripts/catalog/etl_catalog.py` (CSV + sitemap → `catalog.jsonl`).
- Загрузка в PostgreSQL (+ pgvector) для векторного поиска.
- Источник истины — Postgres; Strapi (ориентир ТЗ) — опциональный слой совместимости.

Статус: ETL готов (M0); загрузка в БД — M1.
