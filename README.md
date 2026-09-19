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

M1 (Tier-1) реализован: нормализация, энкодер SigLIP 2 so400m, обученная голова домен-адаптации,
геоверификация Top-K (ORB + RANSAC) и калиброванные пороги решения. Proxy-качество на синтетических
«полевых» фото: acc@1 0.685, acc@5 0.790 (seed 42); полевой прогон 31 фото — 0 ошибок, p95 358 мс.
VLM-fallback (C-004), UI (C-005), flywheel (C-006) и деплой (C-007) — в разработке.

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

## Быстрый старт (M1 — Tier-1 пайплайн)

```bash
# 0. Окружение
python3 -m venv --system-site-packages .venv
.venv/bin/pip install torch torchvision transformers opencv-python-headless uvicorn python-multipart

# 1. Каталог: CSV + sitemap → data/catalog/
python3 scripts/catalog/etl_catalog.py

# 2. Индекс: эмбеддинги эталонов + признаки геоверификации (+ голова, если обучена)
PYTHONPATH=. .venv/bin/python scripts/catalog/build_index.py \
  --encoder siglip2 --model google/siglip2-so400m-patch16-naflex \
  --head models/head.pt --features --out data/index

# 3. Домен-адаптация (опционально): пары и обучение головы
PYTHONPATH=. .venv/bin/python scripts/train/synthetic_pairs.py --limit 700 --per-slug 2 --out data/train
PYTHONPATH=. .venv/bin/python scripts/train/train_head.py --pairs data/train/pairs.jsonl \
  --embeddings data/train/embeddings.npy --epochs 12 --out models/head.pt

# 4. Сервисы (retrieval прогревается на старте — см. «Прогрев и SLA»)
PYTHONPATH=. .venv/bin/python -m uvicorn services.normalize.app.main:app --port 8081 &
PYTHONPATH=. .venv/bin/python -m uvicorn services.retrieval.app.main:app --port 8082 &
PYTHONPATH=. .venv/bin/python -m uvicorn services.gateway.app.main:app  --port 8080 &

# 5. Оценка: калибровка порогов и baseline-отчёт
PYTHONPATH=. .venv/bin/python scripts/eval/calibrate.py --index data/index --samples 200 --normalize --out reports/calibration.json
PYTHONPATH=. .venv/bin/python scripts/eval/baseline_report.py --index data/index --samples 200 --normalize --geo --out reports/baseline.json
python3 scripts/eval/evaluate.py --images data/samples/field \
  --endpoint http://127.0.0.1:8080/v1/eval/predict --output reports/predictions.field.jsonl
```

## Прогрев и SLA

Холодный старт (загрузка энкодера, головы и индекса) занимает ~3–6 секунд, что превышает целевой
SLA 3 секунды. Поэтому retrieval **прогревается при старте**: `WARMUP_ON_START=1` (по умолчанию)
загружает модель и прогоняет пробный поиск до того, как сервер начнёт принимать запросы.
Дополнительно доступны явные эндпоинты:

```bash
curl -s -X POST http://127.0.0.1:8082/warmup   # только retrieval
curl -s -X POST http://127.0.0.1:8080/v1/warmup # gateway → retrieval
```

Замер: первый `/v1/scan` после старта — 955 мс (до прогрева было 6137 мс); полевой прогон
31 фото — p50 235 мс, p95 358 мс, max 502 мс. Отключается переменной `WARMUP_ON_START=0`.

## Геоверификация и калибровка

Top-K переранжируется локальными признаками: ORB (по умолчанию) или SIFT, BFMatcher с ratio-test
и RANSAC-гомография; вклад регулируется `GEO_WEIGHT`, число верифицируемых кандидатов — `GEO_TOPK`.
Пороги `SCORE_TAU`, `MARGIN_DELTA` и `CONFIDENCE_TEMPERATURE` подбираются `scripts/eval/calibrate.py`
на синтетическом proxy-наборе и фиксируются в `.env.example`. `confidence` — softmax по top-K
(калиброванная вероятность), `margin` — отрыв top1 от top2.

## Запуск через Docker (весь стек)

```bash
make images                 # один раз: базовые образы svoe-base и svoe-ml-base
cp .env.example .env
docker compose up --build -d
make warm                   # прогрев retrieval: первый запрос не платит за загрузку модели
# gateway http://127.0.0.1:8080  ·  UI http://127.0.0.1:3000
docker compose config       # валидация конфигурации
```

Локальные LLM/VLM (Ollama :11434, LM Studio :1234) работают **на хосте**; из контейнеров они
доступны как `host.docker.internal` (в compose настроен `extra_hosts`). Веса моделей и данные
монтируются volume-ами (`./data`, `./models`), а не запекаются в образ.

## Единые команды (Makefile)

| Команда | Что делает |
|---|---|
| `make setup` | venv + python-зависимости + `bun install` |
| `make etl` | каталог: CSV + sitemap → `data/catalog` (2103 slug) |
| `make index` | векторный индекс + признаки геоверификации (+ голова, если обучена) |
| `make warm` | прогрев retrieval до первого запроса (SLA) |
| `make eval` | baseline-отчёт (Tier-1 + гео) и полевой прогон через gateway |
| `make test` / `make lint` | pytest и `grace lint` |
| `make up` / `make down` / `make logs` | весь стек в Docker |
| `make web-dev` / `make web-build` / `make web-preview` | Nuxt: dev, сборка, запуск сборки |

Материалы для сдачи: `docs/DEMO.md` (сценарий видео ≤7 мин) и `docs/LIMITATIONS.md`
(честные ограничения и замеры).

## Переменные окружения

См. `.env.example`. Ключевые: `ENCODER`, `ENCODER_MODEL`, `ENCODER_DEVICE`, `VECTOR_BACKEND`, `TOP_K`,
`INDEX_DIR`, `GEO_WEIGHT`, `GEO_TOPK`, `GEO_METHOD`, `SCORE_TAU`, `MARGIN_DELTA`,
`CONFIDENCE_TEMPERATURE`, `WARMUP_ON_START`, `NORMALIZE_URL`, `RETRIEVAL_URL`, `DATABASE_URL`,
`VLM_PROVIDER`, `VLM_MODEL`.

## Ограничения

- Каталог — из материалов конкурса; для целей вне конкурса использовать нельзя.
- Приватный тест — у организатора; публичный набор полевых фото неполный.
- Локальный VLM ограничен железом (Mac M4/32GB): основная модель `gemma4:12b`.
- Пороги решения откалиброваны на синтетических «полевых» фото; при переносе на реальные фото
  приватного набора требуется повторная калибровка (`scripts/eval/calibrate.py`).
- `faiss` на macOS arm64 не используется: его libomp конфликтует с libomp от torch (`OMP Error #15`),
  поэтому векторный поиск идёт numpy-бэкендом.
