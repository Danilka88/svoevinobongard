# ARCHITECTURE

## Обзор

```
apps/web (Nuxt 3, mobile-first)
        │  HTTP/JSON — единственный контракт
        ▼
services/gateway (FastAPI)  ── /v1/eval/predict  /v1/scan  /v1/similar  /v1/sommelier
        │
        ├── services/normalize     — EXIF, детекция этикетки, выпрямление, TTA
        ├── services/retrieval     — энкодер (SigLIP 2/DINOv2) + pgvector/FAISS + геоверификация
        ├── services/vlm           — OpenAI-совместимый коннектор (fallback)
        ├── services/matcher       — фьюжн скоров, margin, решение
        ├── services/active_learning — логи → ревью → аугментация → дообучение
        └─────────────────────────────────────────────
                     PostgreSQL + pgvector (catalog)
```

**Границы слоёв**
- `gateway` — единственная публичная точка. UI не обращается к ML-сервисам напрямую.
- Модули общаются по HTTP/JSON, не импортируют друг друга → независимый деплой.
- UI можно удалить без влияния на ML.

## Пайплайн (онлайн)

Нормализация → Tier-1 CV → near-dup/OCR-сигнал → решение; observer, VLM-fallback и калиброванный
гейт — опциональные слои за флагами.

1. **Прогрев (при старте):** `retrieval` загружает энкодер, голову, индекс и признаки эталонов и
   прогоняет пробный поиск (`WARMUP_ON_START=1`, эндпоинты `/warmup` и `/v1/warmup`), чтобы первый
   пользовательский запрос не платил за холодный старт.
2. `POST /v1/scan` (или `/v1/eval/predict`) принимает `image` (multipart).
3. **Нормализация** (`services/normalize`): EXIF-ориентация, ресайз, **полный кадр без кропа этикетки**
   (`NORMALIZE_CROP=0`) + letterbox и pan-and-scan тайлы (`PAS_TILES=4/6`). Это база и для retrieval, и
   для VLM: овальные/цилиндрические этикетки, блики и ракурсы ломают детекцию и выпрямление.
   Детектор этикетки — OpenCV-эвристика (нейросетевой D-FINE/RT-DETRv4 не подключён).
4. **Tier-1 CV** (`services/retrieval`): эмбеддинг **SigLIP 2 so400m** (naFlex) → проекционная голова
   домен-адаптации → kNN по numpy-бэкенду (pgvector/FAISS — прод-цель) → Top-K; затем
   **геоверификация ORB + BFMatcher(ratio-test) + RANSAC** по верхним `GEO_TOPK` кандидатам →
   фьюжн `score + GEO_WEIGHT · geo_score` и переранжирование.
5. **Near-dup / OCR-сигнал:** группы near-дубликатов (серии одного винодела) и дешёвый OCR-сигнал
   (RapidOCR, ~250 мс) подключаются при неуверенном Tier-1; оба — за флагами
   (`NEAR_DUP_DISAMBIGUATION=0`, `OCR_ENABLED=1`, `TEXT_FUSION_MODE=conditional`).
6. **Решение:** `confidence` — softmax по top-K с калиброванной температурой; `margin` — отрыв top1 от
   top2; `confident = (score ≥ SCORE_TAU) and (margin ≥ MARGIN_DELTA)`. Калиброванный **`p_correct`-гейт**
   (`CALIBRATION_ENABLED=0`, консервативная поправка) уточняет уверенность, не меняя дефолтного решения.
7. **Observer (опциональный слой, `OBSERVER_ENABLED=0`).** При неуверенности свободный текст наблюдения
   этикетки ищется **по всему каталогу** функцией `text_search` (`packages/common/text_rank.py`) и
   сливается с визуальным top-K (`path="observer"`). Слой живёт на VLM-сервисе (`POST /observe`), имеет
   жёсткий таймаут (`OBSERVER_TIMEOUT_S=6.5`) и по умолчанию **выключен**: зачётный скрипт
   последовательный с лимитом 10 с/запрос, а точность наблюдений не доказана (нет ground truth).
   Включается как «quality mode» для живого демо, а не для зачётного пути.
8. **VLM-fallback** (C-004): коннектор `services/vlm` может быть вызван при неуверенности; по умолчанию
   выключен (`VLM_FALLBACK_ENABLED=0`) — локальные модели не укладываются в бюджет SLA.
9. **«Не найдено» → аналоги:** без уверенного ответа система отдаёт `not_found` и ближайшие
   кандидаты/аналоги (`GET /v1/similar/{slug}`), а не пустой экран или неверный ответ.
10. Ответ: `slug`, `confidence`, `f1_top1`, `f1_top5`, `margin`, `path`, `latency_ms`, `top5`, `card`.
11. Лог запроса → flywheel (`services/active_learning`).

## Пайплайн (офлайн, flywheel)

```
логи → фильтр (низкий margin/ошибки) → очередь ревью (человек)
     → подтверждённые метки + синтетическая аугментация (Albumentations)
     → дообучение Tier-1 (contrastive / LoRA) → оценка F1@1/@5
     → реестр модели (MLflow/DVC) → деплой новой версии
```

## Стек

- Backend: FastAPI, Python 3.14 (`requirements.txt`)
- CV: PyTorch + transformers (MPS/ONNX), OpenCV (numpy-бэкенд индекса)
- Модели: SigLIP 2 so400m / DINOv2 (Apache-2.0), RapidOCR и PaddleOCR-VL (Apache-2.0),
  VLM `gemma4` (локально) / любые OpenAI-совместимые.
  Нейросетевой детектор этикетки (D-FINE/RT-DETRv4) — кандидат, не подключён.
- БД: PostgreSQL + pgvector (прод-цель); numpy-бэкенд для локальной разработки
- UI: Nuxt 3
- Инфра: Docker Compose (локально), Yandex Cloud Functions/Workflow/Object Storage (деплой)

## Конфигурация VLM

Единый OpenAI-совместимый коннектор, смена провайдера только через env:

```
VLM_PROVIDER=ollama | lmstudio | openrouter | yandex | custom
VLM_BASE_URL=...   VLM_API_KEY=...   VLM_MODEL=gemma4:12b   VLM_MODEL_FAST=gemma4:e4b
```

## Схема развёртывания

```
                 ┌──────────────────────────────────────────────┐
   браузер ─────▶│  web (Nuxt 3 / Nitro)  :3000                 │
                 └───────────────────┬──────────────────────────┘
                                     │ HTTP (NUXT_PUBLIC_API_BASE)
                 ┌───────────────────▼──────────────────────────┐
   скрипт  ─────▶│  gateway (FastAPI)     :8080                 │  ← единственный публичный вход
   организатора  └──┬────────┬────────┬────────┬────────┬───────┘
                     │        │        │        │        │
              normalize  retrieval   vlm    matcher    ocr        (по HTTP/JSON, без импортов)
               :8081      :8082    :8083    :8084    :8085
                     │        │        │        │        │
                     └────────┴───┬────┴────────┴────────┘
                                  ▼
                     PostgreSQL + pgvector (прод) / numpy-индекс (dev)
                     Локальные LLM: Ollama :11434 НА ХОСТЕ (host.docker.internal)
```

**Локально:** `docker compose up` поднимает postgres + 6 сервисов + web; базовые образы
собираются `make images`. Работает и без Docker — numpy dev-бэкенд (`make setup && make index`).

**Yandex Cloud (манифесты в `deploy/yandex/`):** каждый сервис — Serverless Container;
оркестрация и триггеры переобучения — Workflow; изображения/веса/отчёты — Object Storage;
каталог — Managed PostgreSQL + pgvector; секреты — Lockbox; образы — Container Registry.

**Границы слоёв (не нарушаются):**
- `web` обращается только к `gateway`; удаление UI не влияет на ML.
- `services/*` не импортируют друг друга — только HTTP/JSON; общий код живёт в `packages/*`.
- Локальные LLM/VLM не переносятся в образы: провайдеры подключаются через `config/providers.yaml`
  и env, поэтому локальный запуск и облако используют один код.

## Нефункциональные требования

- SLA < 3 сек (цель), жёсткий лимит скрипта 10 сек. Замер M1: полевой прогон p50 235 мс, p95 358 мс,
  первый запрос после прогрева 955 мс.
- Прогрев обязателен: холодный старт ~6 с выше SLA, поэтому модель греется при старте сервиса.
- Пороги решения калибруются на proxy-наборе (`scripts/eval/calibrate.py`) и переносятся через env.
- Устойчивость: при недоступности VLM — отдаём результат Tier-1; при недоступности геоверификации —
  работаем по эмбеддингам (`GEO_WEIGHT=0`).
- Воспроизводимость: pin версий, `.env.example`, docker-compose, README.

## Что добавилось после M1 (C-012…C-023)

**Вход.** Кадр целиком, без кропа этикетки (`NORMALIZE_CROP=0`): овальные и цилиндрические этикетки,
блики и ракурсы ломают детекцию. Вместо кропа — letterbox-масштабы и P&S-тайлы (`PAS_TILES=4`,
`PAS_TILES_UNCERTAIN=6`), покрывающие весь кадр без знания положения этикетки.

**Галерея.** `build_index.py --views N` индексирует эталон плюс N−1 синтетических полевых видов
(детерминированный seed через crc32); slug повторяется на каждый вектор, поэтому поиск возвращает позицию.
Дефолт — 6 видов (12 222 вектора, 12.4 МБ, сборка ≈60 мин). Геоверификация ORB+RANSAC хранит признаки
по одному набору на slug.

**Текстовый сигнал.** OCR-движки: `rapidocr` (латиница и цифры, дефолт), `cyrillic` (PP-OCRv5 mobile,
кириллица), `fusion` (оба, построчное слияние), `paddleocr_vl` (лучшее чтение, ≈3 мин на кадр — офлайн).
Режимы фьюжна `off | conditional | always`; дефолт `conditional` — включение только на near-dup пару
в визуальном top-2 (32–40 срабатываний из 150, 0 ухудшений).

**Данные.** Официальный API `api.vino-svoe.ru/v1` синхронизируется в кэш (`data/api_cache/`, 9 МБ) и даёт
`catalog_enriched_v3.jsonl` (2035 из 2103 позиций): крепость, температура подачи, гастросочетания, сорта,
регион, винодельня со slug, изображение. Энотуризм: 9 туров, 11 регионов, 180 статей.
Офлайн-обогащение моделью 26b заполняет год и сладость для 228 near-dup позиций.

**Границы не изменились.** UI общается только с gateway; `services/*` не импортируют друг друга —
общий код (каталог, препроцессинг, OCR, клиент API) живёт в `packages/*`. VLM/LLM доступны только офлайн.
