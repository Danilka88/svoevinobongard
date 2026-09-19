# deploy/yandex — развёртывание в Yandex Cloud

Манифесты **готовы к правке**: плейсхолдеры `<SA_ID>`, `<REGISTRY_ID>`, `<TAG>`,
`<LOCKBOX_SECRET_ID>`, `<NETWORK_ID>`, `<DATABASE_URL>` подставляются при деплое.
Реальный деплой в рамках конкурса не требуется (команды демонстрируют сервис локально),
эти файлы показывают готовность архитектуры к внедрению.

## Состав

```
containers/
  gateway.yaml     # публичный ingress, оркестрация пайплайна
  normalize.yaml   # внутренний, без публичного доступа
  retrieval.yaml   # ML-образ, volume с индексом и весами (HF_HOME)
  vlm.yaml         # провайдеры LLM через Lockbox-секреты
  matcher.yaml     # фьюжн полей с каталогом
  ocr.yaml         # RapidOCR
  web.yaml         # Nuxt/Nitro, публичный ingress
workflow/
  retrain.yaml     # flywheel: логи → ревью → аугментация → дообучение → реиндекс → отчёт
```

## Компоненты

| Компонент | Сервис Yandex Cloud | Замечания |
|---|---|---|
| gateway / normalize / retrieval / vlm / matcher / ocr / web | **Serverless Containers** | отдельный образ на сервис, свой `container.yaml` |
| Оркестрация пайплайна и триггеры переобучения | **Workflow** | `workflow/retrain.yaml`; таймер-триггер nightly |
| Изображения, датасеты, веса, отчёты | **Object Storage** | бакеты `svoe-vino-data`, `svoe-vino-models`, `svoe-vino-logs`, `svoe-vino-reports`, `svoe-vino-config` |
| Каталог + векторный поиск | **Managed PostgreSQL** с расширением **pgvector** | `VECTOR_BACKEND=pgvector`; индекс эталонов строится в БД |
| Секреты | **Lockbox** | `DATABASE_URL`, `OPENROUTER_API_KEY`, `YANDEX_API_KEY` — только через секреты, не в манифестах |
| Образы | **Container Registry** | `cr.yandex/<REGISTRY_ID>/svoe-<service>:<TAG>` |
| Публичный доступ | **API Gateway** (опционально) | единая точка для UI и скрипта организатора |

## Деплой (шаги)

```bash
# 1. Собрать и запушить образы
make images
docker tag svoe-base:latest cr.yandex/<REGISTRY_ID>/svoe-base:latest
# ... по сервисам: docker build -f services/<svc>/Dockerfile -t cr.yandex/<REGISTRY_ID>/svoe-<svc>:<TAG> .
yc container registry configure-docker
docker push cr.yandex/<REGISTRY_ID>/svoe-<svc>:<TAG>

# 2. Секреты
yc lockbox secret create --name svoe-vino --payload '[{"key":"DATABASE_URL","text_value":"..."}]'

# 3. Ревизии контейнеров
yc serverless container revision deploy --spec deploy/yandex/containers/gateway.yaml

# 4. Workflow переобучения
yc serverless workflow create --name svoe-retrain --spec deploy/yandex/workflow/retrain.yaml
yc serverless trigger create timer --name svoe-nightly --cron-expression "0 3 * * *" \
  --invoke-workflow-id <WORKFLOW_ID>
```

## Особенности

- **Локальный VLM/LLM в облако не переносится**: `vlm.yaml` использует облачную часть цепочки
  (`openrouter`, `yandex`), а `LLM_LOCAL_ONLY=0`. Для локального запуска Ollama живёт на хосте
  и доступен контейнерам как `host.docker.internal:11434` (см. `docker-compose.yml`).
- **Веса моделей** монтируются из Object Storage в `/models` (`HF_HOME=/models/hf`), а не запекаются
  в образ: образ остаётся небольшим, модель обновляется без пересборки.
- **CPU-инференс**: `ENCODER_DEVICE=cpu` для Serverless; на своём железе — `auto` (MPS/CUDA).
- **Прогрев**: контейнер `retrieval` стартует с `WARMUP_ON_START=1`, поэтому холодный старт
  не попадает в первый запрос организатора.
