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
        ├── services/active-learning — логи → ревью → аугментация → дообучение
        └─────────────────────────────────────────────
                     PostgreSQL + pgvector (catalog)
```

**Границы слоёв**
- `gateway` — единственная публичная точка. UI не обращается к ML-сервисам напрямую.
- Модули общаются по HTTP/JSON, не импортируют друг друга → независимый деплой.
- UI можно удалить без влияния на ML.

## Пайплайн (онлайн)

1. **Прогрев (при старте):** `retrieval` загружает энкодер, голову, индекс и признаки эталонов и
   прогоняет пробный поиск (`WARMUP_ON_START=1`, эндпоинты `/warmup` и `/v1/warmup`), чтобы первый
   пользовательский запрос не платил за холодный старт.
2. `POST /v1/scan` (или `/v1/eval/predict`) принимает `image`.
3. **Нормализация:** EXIF-ориентация, ресайз, детекция этикетки/бутылки (D-FINE/RT-DETRv4),
   выпрямление перспективы, TTA-кропы. При неудаче детектора — кропы OpenCV-эвристикой.
4. **Tier-1 retrieval:** эмбеддинг SigLIP 2 so400m (naFlex) → проекционная голова домен-адаптации →
   kNN по numpy-бэкенду (pgvector/FAISS — прод-цель) → Top-K.
5. **Геоверификация:** ORB (или SIFT) + BFMatcher с ratio-test + RANSAC-гомография по верхним
   `GEO_TOPK` кандидатам → фьюжн `score + GEO_WEIGHT · geo_score` и переранжирование.
6. **Решение:** `confidence` — softmax по top-K с калиброванной температурой; `margin` — отрыв
   top1 от top2; `confident = (score ≥ SCORE_TAU) and (margin ≥ MARGIN_DELTA)`.
   Ветвление на **VLM-fallback** (`gemma4:12b`) при неуверенности подключает change C-004.
7. Ответ: `slug`, `confidence`, `f1_top1`, `f1_top5`, `margin`, `path`, `latency_ms`, `top5`, `card`.
8. Лог запроса → `active-learning`.

## Пайплайн (офлайн, flywheel)

```
логи → фильтр (низкий margin/ошибки) → очередь ревью (человек)
     → подтверждённые метки + синтетическая аугментация (Albumentations)
     → дообучение Tier-1 (contrastive / LoRA) → оценка F1@1/@5
     → реестр модели (MLflow/DVC) → деплой новой версии
```

## Стек

- Backend: FastAPI, Python 3.11+
- CV: PyTorch + transformers/timm (MPS/ONNX), OpenCV, Kornia
- Модели: SigLIP 2 / DINOv2 (Apache-2.0), D-FINE/RT-DETRv4 (Apache-2.0),
  PaddleOCR (Apache-2.0), VLM `gemma4:12b` (локально) / любые OpenAI-совместимые
- БД: PostgreSQL + pgvector; FAISS для локальной разработки
- UI: Nuxt 3
- Инфра: Docker Compose (локально), Yandex Cloud Functions/Workflow/Object Storage (деплой)

## Конфигурация VLM

Единый OpenAI-совместимый коннектор, смена провайдера только через env:

```
VLM_PROVIDER=ollama | lmstudio | openrouter | yandex | custom
VLM_BASE_URL=...   VLM_API_KEY=...   VLM_MODEL=gemma4:12b   VLM_MODEL_FAST=gemma4:e4b
```

## Нефункциональные требования

- SLA < 3 сек (цель), жёсткий лимит скрипта 10 сек. Замер M1: полевой прогон p50 235 мс, p95 358 мс,
  первый запрос после прогрева 955 мс.
- Прогрев обязателен: холодный старт ~6 с выше SLA, поэтому модель греется при старте сервиса.
- Пороги решения калибруются на proxy-наборе (`scripts/eval/calibrate.py`) и переносятся через env.
- Устойчивость: при недоступности VLM — отдаём результат Tier-1; при недоступности геоверификации —
  работаем по эмбеддингам (`GEO_WEIGHT=0`).
- Воспроизводимость: pin версий, `.env.example`, docker-compose, README.
