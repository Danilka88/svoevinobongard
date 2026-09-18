# 🍷 Своё Вино — сканер российских вин по фото этикетки

Сервис распознаёт вино по фотографии этикетки и отдаёт карточку вина с платформы
**«Своё Вино»**: `{"slug": "..."}` + уверенность + топ-5 кандидатов.

> Конкурс: РСХБ / ЛЦТ. Задача: найти карточку вина в каталоге (~2100 позиций) по «полевому»
> фото этикетки. Целевая точность 90–100%, SLA < 3 сек.

## Идея

Гибридный пайплайн + самообучение:

```
фото → нормализация → Tier-1 retrieval (SigLIP 2 / DINOv2 + pgvector + геоверификация)
     → если неуверенно → VLM-fallback (gemma4:12b) → матч по метаданным
     → ответ {"slug", confidence, top5}
     → лог → активное обучение → дообучение Tier-1
```

Tier-1 дешёвый и быстрый и со временем «впитывает» знания VLM (data flywheel),
поэтому сервис становится точнее без роста стоимости и латентности.

## Статус

Ранняя стадия (M0): каталог, ETL и эвалюатор. Пайплайн моделей — в разработке.

## Структура

```
apps/web/                 # Nuxt 3 — мобильный UI (отделяемый модуль)
services/
  gateway/                # FastAPI — публичный API и оркестрация
  normalize/              # детекция этикетки, выпрямление, кропы
  retrieval/              # энкодеры + pgvector/FAISS + геоверификация
  vlm/                    # OpenAI-совместимый коннектор (ollama/lmstudio/openrouter/…)
  matcher/                # фьюжн скоров и решение
  catalog/                # ETL и хранение каталога
  active-learning/        # логи, ревью, аугментация, дообучение
packages/{contracts,common}/
scripts/{catalog,eval}/   # ETL каталога и оценка F1
deploy/yandex/            # манифесты Yandex Cloud Functions / Workflow
data/                     # датасеты и фото (gitignored)
```

Подробности: `ARCHITECTURE.md`.

## Требования

- Docker + Docker Compose
- Python 3.11+ (для скриптов ETL/оценки)
- Ollama (локальный VLM) — опционально для fallback
- PostgreSQL + pgvector — в docker-compose

## Быстрый старт (M0 — каталог и оценка)

```bash
# 1. Каталог: CSV + sitemap → data/catalog/ (путь к материалам по умолчанию)
python3 scripts/catalog/etl_catalog.py

# 2. Оценка на своих данных (если есть ground truth)
python3 scripts/eval/evaluate.py \
  --images data/samples/field \
  --manifest data/samples/field/manifest.tsv \
  --ground-truth data/samples/field/ground_truth.tsv \
  --endpoint http://127.0.0.1:8080/v1/scan
```

## Переменные окружения

См. `.env.example`. Ключевые: `VLM_PROVIDER`, `VLM_BASE_URL`, `VLM_API_KEY`, `VLM_MODEL`,
`VLM_MODEL_FAST`, `DATABASE_URL`, `GATEWAY_PORT`.

## Ограничения

- Каталог — из материалов конкурса; для целей вне конкурса использовать нельзя.
- Приватный тест — у организатора; публичный набор полевых фото неполный.
- Локальный VLM ограничен железом (Mac M4/32GB): основная модель `gemma4:12b`.
