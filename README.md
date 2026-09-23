# 🍷 Своё Вино — сканер российских вин по фото этикетки

Компьютерное зрение + гибридный retrieval: **фото этикетки → `{"slug": "<вино>"}`** с уверенностью,
топ-5 кандидатов и карточкой вина с платформы «Своё Вино» — за **≈160 мс** (SLA 3 с), без единого
облачного вызова на инференсе.

> **Конкурс:** РСХБ / ЛЦТ. **Задача:** найти карточку вина в каталоге (2103 позиции) по «полевому»
> фото этикетки. **Цель:** 90–100% top-1, SLA < 3 с (жёсткий лимит скрипта организатора — 10 с).

---

## До / После

**До:** покупатель у полки с десятками российских вин не может найти их в международных приложениях;
поиск вручную по названию — минуты и ошибки; интеграция с каталогом требует ручной сверки этикетки.

**После:** одно фото → карточка именно этого вина (винодельня, регион, сорт, крепость, температура
подачи, гастросочетания, описание и изображение) за доли секунды. Система **сама сообщает
`not_found`**, когда не уверена (83% случаев на реальных фото), вместо неверного ответа.

---

## О продукте

- **Работает на реальных фото, а не только на синтетике.** 21 магазинное фото: identity-match@1
  **0.53**, @5 **0.58** по российским винам, корректный отказ для не-российских — **100%**
  (`docs/METRICS.md`, C-025). Синтетический proxy (in-distribution): acc@1 **0.775**, acc@5 **0.970**.
- **Трёхуровневый гибрид по цене и скорости.** Дешёвый Tier-1 (эмбеддинг + геоверификация) →
  OCR-сигнал (сотни мс) → VLM-арбитр — только при неуверенности. Каждый уровень деградирует
  корректно: без OCR/VLM система отдаёт результат Tier-1 (`services/gateway/app/pipeline.py:222`).
- **Data flywheel (самообучение).** Логи запросов без PII → очередь ревью (HITL) → подтверждённые
  метки → экспорт датасета → дообучение головы → реестр версий → реиндекс. `make flywheel-*`.
- **Карточка = официальные данные портала.** Обогащение из `api.vino-svoe.ru/v1` для **2035 из 2103**
  позиций: крепость, температура подачи, гастрономические сочетания, сорта, регион, винодельня со
  slug, изображение (C-017/C-018).
- **Мультипровайдерный LLM Router.** Ollama · LM Studio · OpenRouter · YandexGPT · custom — одна
  конфигурация, fallback-цепочка, circuit breaker и cost-velocity breaker (C-008).
- **Полностью локальный инференс.** Нет per-request стоимости и зависимости от сети: `LLM_LOCAL_ONLY=1`
  исключает облако из цепочки. Облако остаётся опциональным fallback.
- **Модульная архитектура без связности.** `services/*` не импортируют друг друга — только HTTP/JSON
  через `gateway`; UI (`Nuxt 3`) можно удалить без влияния на ML (`ARCHITECTURE.md`).
- **Готовность к облаку.** Манифесты Serverless Containers, Workflow-ретриггер, Lockbox,
  Object Storage, Managed PostgreSQL + pgvector (`deploy/yandex/`).
- **Только permissive-лицензии.** SigLIP 2, DINOv2, RapidOCR, OpenCV, FastAPI, Nuxt, PyTorch —
  Apache-2.0/MIT. AGPL (Ultralytics) не используется. Чистый clean-room относительно `old/`.
- **Инженерная дисциплина.** GRACE 4: 101 governed-файл, контракты, assertion-гейты, 33 завершённых
  change с машино-проверямыми критериями; `make gate` — один коммит-гейт качества и SLA.

---

## Скрытые возможности (есть в коде — неочевидны снаружи)

| Возможность | Где | Почему это преимущество |
|---|---|---|
| **LLM Router** с 5 провайдерами, fallback-цепочкой, per-provider circuit breaker и cost-velocity breaker | `packages/common/llm/router.py`, `config/providers.yaml` | Смена модели/провайдера — только через YAML/env, без правок кода; отказ провайдера не роняет фичу |
| **4 OCR-движка**: `rapidocr` (дефолт), `cyrillic` (PP-OCRv5), `fusion` (лат+кир), `paddleocr_vl` | `packages/common/ocr*.py`, `services/ocr/` | Быстрый онлайн-сигнал (250–290 мс) + лучший офлайн-ридер для обогащения |
| **Плугабельные энкодеры**: `siglip2` / `dinov2` / `clip`; бэкенды `numpy` / `faiss` / `pgvector` | `services/retrieval/app/encoder.py`, `.env.example` | Смена представления и векторного стора без переписывания сервиса |
| **Геоверификация ORB/SIFT + RANSAC** с проверкой всех видов позиции | `services/retrieval/app/geo.py` | Разводит near-duplicate серии (порог/вес калибруются: `GEO_WEIGHT`, `GEO_TOPK`) |
| **Мультивью-галерея** (6 синтетических «полевых» видов на позицию → 12 222 вектора) | `scripts/catalog/build_index.py --views` | Устойчивость к ракурсу/масштабу без дообучения модели |
| **Транслитерация лат↔кир** для сопоставления названий | `packages/common/text_norm.py`, `text_rank.py` | `chateau tamagne` ↔ `шато тамань`; 331/400 записей улучшились, 0 ухудшились (C-027) |
| **Текстовый фьюжн `conditional`** (только на near-dup пару в визуальном top-2) | `services/matcher/app/textfusion.py` | 32/150 срабатываний, 0 ухудшений; флаг `TEXT_FUSION_MODE` |
| **Офлайн-обогащение моделью `gemma4:26b-mlx`** (137 годов, 102 сладости для 228 slug) | `scripts/kb/enrich_vlm.py` | Улучшает каталог там, где онлайн-VLM не укладывается в SLA |
| **Энотуризм**: 9 туров, 11 регионов, 180 статей + сомелье и аналоги | `services/gateway/app/enotourism.py`, `recommend.py` | Retention-блоки сверх основного сценария; сомелье работает даже без LLM (детерминированный откат) |
| **Flywheel HITL**: SQLite-логи, очередь ревью, экспорт датасета, реестр версий, ретрейн | `services/active_learning/`, `scripts/flywheel/cli.py` | Замкнутый цикл улучшения качества на подтверждённых данных |
| **Прогрев на старте** (`WARMUP_ON_START`) и P&S-тайлы по всему кадру | `services/retrieval/app/main.py`, `packages/common/preprocess.py` | Холодный старт ~6 с не попадает в первый запрос: 6137 мс → **955 мс** |
| **Yandex Cloud deployment** (Serverless Containers + Workflow + Lockbox + pgvector) | `deploy/yandex/` | Архитектура готова к внедрению, а не только к демо |

---

## Схема работы

**Онлайн-путь (инференс):**

```
фото
  │
  ▼
normalize ── EXIF, full-frame (без кропа), letterbox + P&S-тайлы (4–6)
  │
  ▼
Tier-1 retrieval ── энкодер SigLIP 2 so400m → обученная голова
  │                 → kNN (numpy/FAISS/pgvector) → Top-K
  ▼
геоверификация ── ORB/SIFT + BFMatcher(ratio-test) + RANSAC по GEO_TOPK
  │               → score + GEO_WEIGHT · geo_score
  ▼
решение ── confidence (softmax, калиброванная T) · margin = top1 − top2
  │
  ├─ уверенно ─────────────────────────────► ответ
  └─ неуверенно → OCR-сигнал (rapidocr, ~250 мс)
                     └─ всё ещё неуверенно → VLM-fallback (gemma4, при включении)
  │
  ▼
{"slug", confidence, f1_top1, f1_top5, margin, path, latency_ms, top5, card}
  │
  ▼
flywheel-лог (без PII) → ревью → дообучение → новый индекс
```

**Границы слоёв:**

```
apps/web (Nuxt 3, mobile-first)
        │  HTTP/JSON — единственный контракт
        ▼
services/gateway (FastAPI)  ── /v1/scan · /v1/eval/predict · /v1/similar
        │                          /v1/sommelier · /v1/enotourism · /v1/health
        ├── normalize   :8081   EXIF, детекция, выпрямление, TTA
        ├── retrieval   :8082   энкодер + индекс + геоверификация
        ├── vlm         :8083   OpenAI-совместимый коннектор (fallback)
        ├── matcher     :8084   фьюжн скоров и текстовых полей
        └── ocr         :8085   RapidOCR / PP-OCRv5
                     │
                     ▼
        PostgreSQL + pgvector (прод) / numpy-индекс (dev)
```

**Офлайн (flywheel):**

```
логи → фильтр (низкий margin/ошибки) → очередь ревью (человек)
     → подтверждённые метки + аугментация (Albumentations)
     → дообучение Tier-1 → оценка F1@1/@5 → реестр версии → реиндекс
```

**Деплой (Yandex Cloud):** каждый сервис — Serverless Container; оркестрация и триггеры
переобучения — Workflow; изображения/веса/отчёты — Object Storage; каталог — Managed PostgreSQL +
pgvector; секреты — Lockbox; образы — Container Registry (`deploy/yandex/README.md`).

---

## Технологический стек

#### Tier-1 retrieval (SigLIP 2 so400m + обученная голова)

- **Почему:** мультиязычный image-text энкодер Apache-2.0, устойчив к домен-разрыву; голова
  домен-адаптации дообучается на синтетических «полевых» парах.
- **Реализация:** `services/retrieval/app/encoder.py`, `head.py`, `search.py`; альтернативы
  `dinov2`, `clip`.
- **Время:** сборка индекса 6 видов ≈60 мин, 12 222 вектора / 12.4 МБ; онлайн p95 ≈160 мс.

#### Геоверификация (OpenCV ORB/SIFT + RANSAC)

- **Почему:** дёшево разводит near-duplicate серии, где косинусный скор близок.
- **Реализация:** `services/retrieval/app/geo.py`; `GEO_WEIGHT`, `GEO_TOPK`, `GEO_METHOD`.

#### OCR-сигнал (RapidOCR; опционально PP-OCRv5 cyrillic / PaddleOCR-VL)

- **Почему:** дешёвая замена VLM в онлайн-пути; RapidOCR — ONNX, Apache-2.0, ~250–290 мс.
- **Реализация:** `services/ocr/`, `packages/common/ocr*.py`; `OCR_ENGINE=rapidocr|cyrillic|fusion|paddleocr_vl`.
- **Размен:** `paddleocr_vl` читает лучше всех, но ~186–234 с/кадр — только офлайн.

#### LLM Router (OpenAI-совместимые провайдеры)

- **Почему:** один адаптер закрывает Ollama, LM Studio, OpenRouter, YandexGPT, custom; провайдер-специфика
  не проникает в код.
- **Реализация:** `packages/common/llm/adapters.py`, `router.py`, `circuit.py`, `config/providers.yaml`.
- **Управление:** `LLM_LOCAL_ONLY=1`; cost breaker: `cost_plan_tokens_per_min`, `cost_multiplier`.

#### API (FastAPI) + UI (Nuxt 3)

- **Почему:** gateway — единственная публичная точка; UI отделяем.
- **Реализация:** `services/gateway/app/main.py`, `apps/web/`.

#### Данные (PostgreSQL + pgvector / numpy)

- **Почему:** pgvector — прод-цель и масштабирование; numpy-бэкенд работает без Docker и без
  конфликта libomp на macOS arm64.
- **Реализация:** `VECTOR_BACKEND=numpy|faiss|pgvector`.

---

## Экономическая эффективность

Ключевая экономика: **инференс полностью локальный** — нет стоимости за скан и нет риска
rate-limit. Облачные LLM вызываются только как опциональный fallback.

| Компонент | Вызовов на скан | Время | Стоимость (локально) |
|---|---|---|---|
| Нормализация + P&S-тайлы | 1 | ~10–30 мс | 0 ₽ |
| Tier-1 retrieval (SigLIP + kNN) | 1 | p50 ~157 мс | 0 ₽ |
| Геоверификация ORB+RANSAC | ≤ `GEO_TOPK` (10) | входит в p95 ~176 мс | 0 ₽ |
| OCR-сигнал (только при неуверенности) | 0–1 | ~250–290 мс | 0 ₽ |
| VLM-fallback | 0 (выключен по умолчанию) | 3–34 с при включении | 0 ₽ локально |
| **Итого p95** | — | **176 мс** при SLA 3000 мс | **0 ₽ / скан** |

| Сценарий | Объём | Время | Стоимость |
|---|---|---|---|
| Офлайн-обогащение каталога (`gemma4:26b-mlx`) | 228 near-dup slug | ~1.4 ч, p50 22.4 с | 0 ₽ |
| Сборка индекса (6 видов) | 2103 позиции → 12 222 вектора | ≈60 мин | 0 ₽ |
| Облачный fallback (опция) | по требованию | зависит от провайдера | по тарифу провайдера |

Против коммерческих vision-API: **нет per-image billing**, нет утечки данных за периметр, нет
зависимости от доступности внешнего сервиса — принципиально для банковского/ритейл-контура (РСХБ).

---

## Примеры результатов

### 1. Скан: «плоский» контракт организатора

```bash
curl -s -X POST -F image=@data/samples/eval_queries/019c68d0.jpg \
  http://127.0.0.1:8080/v1/eval/predict
```

```json
{ "slug": "nebo-rossii-glera" }
```

### 2. Скан: rich-ответ для UI

```bash
curl -s -X POST -F image=@label.jpg http://127.0.0.1:8080/v1/scan | jq
```

```json
{
  "slug": "kokur-suhoe-2025",
  "confidence": 0.94,
  "f1_top1": 0.94,
  "f1_top5": 0.81,
  "margin": 0.27,
  "path": "retrieval",
  "latency_ms": 158.4,
  "top5": [
    { "slug": "kokur-suhoe-2025", "score": 0.87, "name": "Кокур сухое", "winery": "…", "geo_score": 0.42, "inliers": 57 }
  ],
  "not_found": false,
  "confident": true,
  "ocr_used": false,
  "degraded": false,
  "card": {
    "name": "Кокур сухое",
    "winery": "…", "region": "Крым",
    "color": "белое", "grape": "Кокур",
    "alcohol_pct": 12.0, "temperature": "10–12 °C",
    "dishes": ["…"], "api_match": true
  }
}
```

> Значения приведены как **пример структуры ответа**; актуальные метрики — в `docs/METRICS.md`
> и `reports/final.json`.

### 3. Похожие вина и энотуризм

```bash
curl -s "http://127.0.0.1:8080/v1/similar/kokur-suhoe-2025?limit=6"
curl -s "http://127.0.0.1:8080/v1/enotourism/kokur-suhoe-2025"
```

---

## Быстрый старт

### Локально, без Docker (numpy dev-бэкенд)

```bash
# 0. Окружение
make setup                       # venv + python-зависимости + bun install

# 1. Каталог: CSV + sitemap → data/catalog/ (2103 slug)
make etl

# 2. Индекс: эмбеддинги эталонов + признаки геоверификации (+ голова)
make index

# 3. Сервисы
PYTHONPATH=. .venv/bin/python -m uvicorn services.normalize.app.main:app --port 8081 &
PYTHONPATH=. .venv/bin/python -m uvicorn services.retrieval.app.main:app --port 8082 &
PYTHONPATH=. .venv/bin/python -m uvicorn services.gateway.app.main:app  --port 8080 &
make warm                        # прогрев retrieval: первый запрос не платит за загрузку модели

# 4. Проверка
curl -s http://127.0.0.1:8080/v1/health
```

### Весь стек в Docker

```bash
make images                      # один раз: базовые образы svoe-base и svoe-ml-base
cp .env.example .env
docker compose up --build -d
make warm
# gateway http://127.0.0.1:8080  ·  UI http://127.0.0.1:3000
```

### UI отдельно

```bash
make web-dev                     # Nuxt dev на http://127.0.0.1:3000
make web-build && make web-preview
```

---

## API

Все эндпоинты публикует `gateway` (единственная точка входа). Контракты — `packages/contracts/schemas.py`.

### `POST /v1/scan` — основной сценарий

- **Request:** `multipart/form-data`, поле `image` (JPG/PNG/WebP, ≤ `MAX_UPLOAD_MB`).
- **Медиа:** картинки отдаются рабочими адресами портала через прокси изображений
  (`https://api.vino-svoe.ru/v1/img/str-api/{w}/{h}/resize/uploads/<file>.webp`, 600 для карточек и 1200 для крупных).
  Сырой путь `/uploads/...` отдаёт 404, поэтому проверять загрузку нужно **GET**-запросом: HEAD на прокси отвечает 404.
- **Response:** `slug`, `confidence`, `f1_top1`, `f1_top5`, `margin`, `path` (`retrieval` / `ocr_fallback` /
  `vlm_fallback` / `text_first`), `latency_ms`, `top5[]`, `not_found`, `confident`, `vlm_used`,
  `ocr_used`, `degraded`, `card`.
- **Status:** `200`; `503` — retrieval недоступен.

### `POST /v1/eval/predict` — контракт организатора

- **Request:** `multipart/form-data`, поле `image`.
- **Response:** строго плоский `{"slug": "..."}` (или `{"slug": null}`).
- **Status:** `200`; `400` — пустое изображение; `503` — пайплайн недоступен.

### `GET /v1/similar/{slug}?limit=6`

- **Response:** `base_slug`, `items[]` (slug, name, winery, region, grape, reason, image_url, source).
- Источник: официальные похожие портала, иначе — метаданные каталога.

### `POST /v1/sommelier`

- **Request:** `{"slug": "...", "answers": {"dish": "...", "budget": "..."}}`.
- **Response:** `recommendations[]`, `pairing[]`, `official_dishes[]`, `source`, `note`.
- **Отказоустойчивость:** при недоступной LLM — детерминированный подбор по каталогу.

### `GET /v1/enotourism/{slug}`

- **Response:** `region` (факт, события), `tours[]`, `articles[]` из официального API.

### `GET /v1/wine/{slug}?with_enotourism=true`

- **Response:** полная карточка вина (`card`) с медиа портала; при `with_enotourism=true` — ещё и блок энотуризма.
- **Status:** `200`; `404` — slug неизвестен.

### `GET /v1/catalog?q=&region=&category=&grape=&winery=&page=&per_page=&sort=`

- **Response:** `total`, `pages`, `items[]`, `facets` (счётчики по региону, категории, сорту и винодельне).
- Поиск ускорен предвычисленным индексом форм (C-046/C-047): индекс прогревается на старте gateway, поэтому первый запрос каталога укладывается в SLA.

### `GET /v1/demo/metrics`

- **Response:** дашборд метрик (прокси-замеры, реальные фото, порог гейта) для экрана `/metrics`.

### `POST /v1/telemetry` · `GET /v1/telemetry/summary`

- Приём **обезличенных** событий интерфейса (просмотры, действия, ошибки, голосования) и сводка по ним:
  сессии, маршруты, действия, ошибки, сценарии, задержки p50/p95 и `llm_validation`
  (`exact` / `repaired` / `retried` / `failed` / `truncated`) — сколько ответов моделей пришлось восстанавливать.
- Приватность: только белый список полей, никаких изображений и свободного текста; сессия — `X-Session-Id`.

### `POST /v1/feedback`

- **Request:** `{"vote": "up|down", "reason": ["wrong_wine" | "wrong_year" | "wrong_grape" | "bad_photo" | "other"], "slug": "...", "image_sha256": "..."}`.
- **Response:** `accepted`, `trusted`, `applied`, `sessions`, `up`/`down`, `agreement`, `agent_vote`, `reasons`.
- **Правило доверия:** одиночный голос **никогда не влияет**. Влияние (`trusted`, `applied`) возможно только при **≥3 независимых сессиях И согласии с консенсусом агентов**; иначе голос уходит в очередь модератора. Действуют дедупликация (сессия + изображение) и ограничение частоты.

### `GET /v1/learning/summary` · `GET /v1/learning/queue` · `GET /v1/learning/dataset` · `POST /v1/learning/decision`

- `summary` — голоса, 👍/👎, распределение причин, доверенные вклады, очередь, решения и пороги правил.
- `queue` — очередь модератора с провенансом (сессии, голоса, причины, время).
- `dataset` — счётчики датасета разметки по разбиениям (`train`/`eval`/`holdout`) и решения модератора.
- `decision` — `{"decision": "accept|reject", "slug" | "image_sha256": "...", "reason": "..."}`; фиксируется провенанс «кто-когда-почему». **Модератор условный (демо): реальной аутентификации нет.**
- Витрина механизма — страница `/learning` (раздел «Обучение»).

### `GET /v1/health` · `POST /v1/warmup`

- `health` — живость и список эндпоинтов; `warmup` — прогрев retrieval (SLA первого запроса).

---

## Установка и конфигурация

- Docker + Docker Compose; Python 3.11+ (для ETL/оценки); Node/Bun (UI); PostgreSQL + pgvector — в compose.
- **Зависимости Python — `requirements.txt`** (Python 3.14): `fastapi`, `uvicorn`, `httpx`, `pydantic`, `PyYAML`,
  `jsonschema` (валидация структурированного вывода моделей), `numpy`, `rapidfuzz`, `Pillow`,
  `opencv-python-headless`; тяжёлые `torch`/`transformers` — опционально и закомментированы.
  Установка: `python -m pip install -r requirements.txt`.
- Все переменные — в `.env.example` (скопировать в `.env`, **не коммитить**).

Ключевые группы настроек:

| Группа | Переменные |
|---|---|
| Энкодер / индекс | `ENCODER`, `ENCODER_MODEL`, `ENCODER_DEVICE`, `VECTOR_BACKEND`, `TOP_K`, `INDEX_DIR`, `VIEWS` |
| Пороги решения | `SCORE_TAU`, `MARGIN_DELTA`, `CONFIDENCE_TEMPERATURE` |
| Геоверификация | `GEO_WEIGHT`, `GEO_TOPK`, `GEO_MIN_MATCHES`, `GEO_METHOD` |
| Полный кадр | `NORMALIZE_CROP`, `PAS_TILES`, `PAS_TILES_UNCERTAIN` |
| OCR | `OCR_ENABLED`, `OCR_ENGINE`, `OCR_MAX_SIDE`, `OCR_TIMEOUT_S` |
| VLM | `VLM_FALLBACK_ENABLED`, `VLM_PROVIDER`, `VLM_MODEL`, `VLM_REASONING_EFFORT` |
| LLM Router | `LLM_PROVIDERS_FILE`, `LLM_LOCAL_ONLY`, ключи провайдеров |
| Текстовый фьюжн | `TEXT_FUSION_MODE`, `TEXT_FUSION_WEIGHT` |
| SLA | `WARMUP_ON_START`, `SERVICE_TIMEOUT_S` |

Провайдеры и цепочки — `config/providers.yaml` (секреты только через env). Актуальные дефолты:
`SCORE_TAU=0.7615`, `MARGIN_DELTA=0.0058`, `GEO_WEIGHT=0.7`, `VIEWS=6`, `OCR_ENGINE=rapidocr`,
`TEXT_FUSION_MODE=conditional`, `VLM_FALLBACK_ENABLED=0`, `NORMALIZE_CROP=0`.

---

## Тестирование

```bash
make test     # pytest
make lint     # grace lint — грамматика и семантическая разметка артефактов
make eval     # синтетический baseline: acc@1/@5, near-dup, latency
make gate     # финальный гейт: lint + тесты + eval (acc@1 ≥ 0.75, p95 ≤ 3000 мс) + живой контракт
```

| Метрика | Значение | Источник |
|---|---|---|
| Тестов собрано | **195** (26 файлов) | `pytest --collect-only` |
| syntax/grammar lint | 0 errors (101 governed-файл) | `grace lint` |
| acc@1 (in-distribution proxy, n=200, seed 42) | **0.775** | `reports/final.json` |
| acc@5 | **0.970** | `reports/final.json` |
| near-duplicate acc@1 | **0.7619** | `reports/final.json` |
| latency p50 / p95 | **156.9 / 176.5 мс** | `reports/final.json` |
| identity@1 / @5 (21 реальное фото, RU) | **0.526 / 0.632** | `reports/real_field.json` (C-025; @5 поднят транслитом, C-027) |
| корректный `not_found` на реальных фото | **82.6%** | `docs/METRICS.md` |
| отказ для не-российских вин | **100%** | `docs/METRICS.md` |

**Честная оговорка.** Прокси-числа **in-distribution**: галерея и запросы строятся одним семейством
аугментаций. На отложенном суровом OOD-семействе acc@1 падает до ≈0.06, поэтому прокси-числа нельзя
выдавать за полевое качество. Полевого набора в комплекте нет (2 официальных контрольных фото) —
подробности в `docs/LIMITATIONS.md`.

---

## Ограничения (кратко)

- Полевых фото нет; официальный комплект — 2 контрольных фото. Полевое качество измерит организатор.
- Узкое место — **представление**: у 6 из 19 реальных российских фото правильной позиции нет даже в
  top-100 из 2103 (C-026). Дешёвые эвристики домена исчерпаны.
- VLM-fallback выключен по умолчанию: локальные модели не читают этикетки в бюджет SLA (3–34 с).
- `faiss` недоступен на macOS arm64 (конфликт libomp с torch) — используется numpy-бэкенд.
- PaddleOCR-VL — офлайн-инструмент (~186–234 с/кадр), не онлайн-путь.

Полный список: `docs/LIMITATIONS.md`.

---

## План развития

**Сделано (C-001…C-033, 33 change):**

- ✅ **M0** — скелет репозитория, ETL каталога (2103 slug), self-evaluator.
- ✅ **M1** — Tier-1 retrieval (SigLIP2-so400m + обученная голова), full-frame вход + P&S,
  мультивью-галерея (6 видов), OOD-валидация.
- ✅ **M2** — геоверификация ORB+RANSAC, калибровка порогов, SLA-контроль, quality/SLA-гейт `make gate`.
- ✅ **M3** — VLM-fallback + matcher fusion, LLM Router, OCR (rapidocr/cyrillic/fusion/paddleocr_vl),
  текстовый KB + условный фьюжн, транслит лат↔кир, RRF text-first (флаг off), cross-encoder (флаг off).
- ✅ **M4** — Nuxt 3 UI: карточка, сомелье, аналоги, энотуризм; паритет карточки с порталом.
- ✅ **M5** — flywheel: SQLite-логи, очередь ревью, HITL, экспорт датасета, реестр версий;
  офлайн-обогащение 26b.
- ✅ **M6** — Docker/compose, Makefile, Yandex-манифесты, демо-документация, синк официального API
  (2035/2103), гигиена артефактов.

**В плане:**

- 🔲 10–20 реальных полевых фото → повторная калибровка порогов и галереи на реальном распределении.
- 🔲 Смена подхода к представлению: дообучение на подтверждённых парах через flywheel
  (это уже обучение — отдельное решение владельца).
- 🔲 Доменный/облачный реранкер (Qwen3-VL-Reranker) — кандидат, когда появится ключ.
- 🔲 pgvector-бэкенд: загрузчик каталога в PostgreSQL (прод-контур).
- 🔲 Кириллический OCR-движок, читающий цифры (год), и нейросетевой детектор этикетки (D-FINE/RT-DETRv4).

---

## Структура

```
apps/web/                 # Nuxt 3 — мобильный UI (отделяемый модуль)
services/
  gateway/                # FastAPI — публичный API и оркестрация
  normalize/              # детекция этикетки, выпрямление, P&S-тайлы
  retrieval/              # энкодеры + numpy/FAISS/pgvector + геоверификация
  vlm/                    # OpenAI-совместимый коннектор (fallback)
  matcher/                # фьюжн скоров и текстовых полей
  ocr/                    # RapidOCR / PP-OCRv5
  catalog/                # ETL и хранение каталога
  active_learning/        # логи, ревью, аугментация, дообучение
packages/{contracts,common}/
scripts/{catalog,eval,kb,train,flywheel}/
config/providers.yaml     # LLM-провайдеры и цепочки
deploy/{docker,yandex}/   # базовые образы, манифесты Yandex Cloud
docs/{METRICS,LIMITATIONS,DEMO,DATA}.md
.grace/                   # GRACE 4: граф, verification, changes
```

Подробности: `ARCHITECTURE.md`, `AGENTS.md`.

## Лицензии

Все обязательные компоненты — permissive (Apache-2.0 / MIT). AGPL-компоненты не используются.
Материалы и данные конкурса находятся вне репозитория (`data/`, `models/` — gitignored).

## Быстрый старт полного демо (gateway + UI)

```bash
# 1. каталог и индекс (один раз; сборка галереи 6 видов занимает ~50 минут)
make etl && make index

# 2. стек сервисов: normalize, retrieval, matcher, gateway
.venv/bin/python -m uvicorn services.normalize.app.main:app --port 8081 &
.venv/bin/python -m uvicorn services.retrieval.app.main:app --port 8082 &
.venv/bin/python -m uvicorn services.matcher.app.main:app   --port 8084 &
.venv/bin/python -m uvicorn services.gateway.app.main:app   --port 8080 &
curl -s http://127.0.0.1:8080/v1/warmup          # прогрев retrieval, ~40 с

# 3. UI: dev-режим или собранная сборка
cd apps/web && bun install && bun run dev        # http://127.0.0.1:3000
# либо прод-сборка:
# cd apps/web && bun run build
# NUXT_PUBLIC_API_BASE=http://127.0.0.1:8080 PORT=3000 node .output/server/index.mjs

# 4. проверка готовности
make gate                                        # lint + тесты + eval + контракт организатора
```
UI общается только с gateway; экраны: Сканер `/`, Обучение `/learning`, Сомелье `/sommelier`,
Каталог `/catalog`, Метрики `/metrics`, Как это работает `/how` (в меню «Ещё»).
Навигация: пять основных пунктов в таб-баре и лист «Ещё» на мобильном, полный список на десктопе.
