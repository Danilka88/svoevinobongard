# scripts/catalog

`etl_catalog.py` — собирает каталог из материалов конкурса.

```bash
# путь к материалам конкурса определяется автоматически (см. --help)
python3 scripts/catalog/etl_catalog.py
# при необходимости:
#   --materials "/путь/к/Датасет" --out data/catalog
```

Выход (`data/catalog/`, gitignored):
- `catalog.jsonl` — по объекту на slug;
- `slug_to_image.json` — индекс slug → файл эталонного фото;
- `stats.json` — сводка.

Источник фото — `wines_sitemap_*.xml` (авторитетный), метаданные — CSV.
