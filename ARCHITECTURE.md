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

1. `POST /v1/scan` (или `/v1/eval/predict`) принимает `image`.
2. **Нормализация:** EXIF-ориентация, ресайз, детекция этикетки/бутылки (D-FINE/RT-DETRv4),
   выпрямление перспективы, TTA-кропы. При неудаче детектора — кропы OpenCV-эвристикой.
3. **Tier-1 retrieval:** эмбеддинги SigLIP 2 / DINOv2 → kNN в pgvector/FAISS → Top-K.
4. **Геоверификация:** SuperPoint+LightGlue + RANSAC по Top-K → уточнённые скоры и `margin`.
5. **Ветвление:**
   - уверенно (`score ≥ τ` и `margin ≥ δ`) → решение;
   - иначе → **VLM-fallback** (`gemma4:12b`): строгий JSON полей этикетки →
     `matcher` сопоставляет с метаданными каталога и переранжирует Top-K.
6. Ответ: `slug`, `confidence`, `f1_top1`, `f1_top5`, `top5`, `path`, `latency_ms`, `card`.
7. Лог запроса → `active-learning`.

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

- SLA < 3 сек (цель), жёсткий лимит скрипта 10 сек.
- Устойчивость: при недоступности VLM — отдаём результат Tier-1.
- Воспроизводимость: pin версий, `.env.example`, docker-compose, README.
