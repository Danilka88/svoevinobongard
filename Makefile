# Единые команды проекта «Своё Вино».
# Работают и БЕЗ Docker (numpy dev-бэкенд): setup/etl/index/eval/test/lint/web-*.
SHELL := /bin/bash
PY    := .venv/bin/python
PIP   := .venv/bin/pip
ENC   ?= siglip2
MODEL ?= google/siglip2-so400m-patch16-naflex
SLUG  ?= kokur-suhoe-2025
PORT  ?= 8080

.PHONY: help setup etl index images up down logs warm eval test lint lint-change web-dev web-build web-preview clean \
        flywheel-ingest flywheel-stats flywheel-queue flywheel-review flywheel-dataset flywheel-train flywheel-registry flywheel-eval demo-eval

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

## --- локальная разработка (без Docker) ---

setup: ## Создать venv и поставить зависимости (python + node)
	python3 -m venv --system-site-packages .venv
	$(PIP) install --quiet --upgrade pip
	$(PIP) install --quiet torch torchvision transformers opencv-python-headless uvicorn \
	  python-multipart httpx pydantic pyyaml rapidfuzz rapidocr-onnxruntime
	cd apps/web && bun install
	@echo "setup: готово. Дальше: make etl && make index"

etl: ## Собрать каталог из материалов конкурса (data/catalog)
	PYTHONPATH=. $(PY) scripts/catalog/etl_catalog.py

index: ## Собрать векторный индекс + признаки геоверификации (+ голова, если есть)
	PYTHONPATH=. $(PY) scripts/catalog/build_index.py \
	  --encoder $(ENC) --model $(MODEL) \
	  $(if $(wildcard models/head.pt),--head models/head.pt,) \
	  --views $(or $(VIEWS),6) --features --out data/index

warm: ## Прогреть retrieval до первого запроса
	curl -fsS -X POST http://127.0.0.1:8082/warmup || true

eval: ## Оценка качества: самодостаточный синтетический baseline (Tier-1 + гео)
	PYTHONPATH=. $(PY) scripts/eval/baseline_report.py --index data/index \
	  --samples $(or $(SAMPLES),200) --seed 42 --normalize --geo --out reports/baseline.json

demo-eval: ## Официальные контрольные фото организатора (опционально, если папка есть)
	@if [ -d data/samples/eval_queries ]; then \
	  python3 scripts/eval/evaluate.py --images data/samples/eval_queries \
	    --endpoint http://127.0.0.1:$(PORT)/v1/eval/predict \
	    --output reports/predictions.eval_queries.jsonl; \
	else echo "data/samples/eval_queries отсутствует — цель demo-eval пропущена"; fi

test: ## Прогнать тесты
	PYTHONPATH=. WARMUP_ON_START=0 $(PY) -m pytest tests/ -q

lint: ## Проверить GRACE-артефакты и разметку
	grace lint --path .

lint-change: ## Гейт конкретного change: make lint-change CHANGE=C-005
	grace lint --path . --change $(CHANGE) --assertions target --run-commands

## --- Docker ---

images: ## Собрать базовые образы (нужно один раз перед up)
	docker build -f deploy/docker/base.Dockerfile -t svoe-base:latest .
	docker build -f deploy/docker/ml-base.Dockerfile -t svoe-ml-base:latest .

up: ## Поднять весь стек (postgres + сервисы + web)
	docker compose up --build -d
	@echo "gateway: http://127.0.0.1:$(PORT)  web: http://127.0.0.1:3000"

down: ## Остановить стек
	docker compose down

logs: ## Хвост логов стека
	docker compose logs -f --tail=100

## --- UI ---

web-dev: ## Dev-сервер Nuxt (http://127.0.0.1:3000)
	cd apps/web && NUXT_PUBLIC_API_BASE=http://127.0.0.1:$(PORT) bun run dev

web-build: ## Прод-сборка Nuxt (.output)
	cd apps/web && bun run build

web-preview: ## Запустить собранный UI
	cd apps/web && NUXT_PUBLIC_API_BASE=http://127.0.0.1:$(PORT) node .output/server/index.mjs

## --- flywheel (C-006): human-in-the-loop ---

flywheel-ingest: ## Загрузить логи запросов gateway в хранилище
	PYTHONPATH=. $(PY) scripts/flywheel/cli.py ingest

flywheel-stats: ## Сводка хранилища: pending/confirmed/corrected/uncertain
	PYTHONPATH=. $(PY) scripts/flywheel/cli.py stats

flywheel-queue: ## Очередь ревью: неуверенные кейсы
	PYTHONPATH=. $(PY) scripts/flywheel/cli.py queue --limit $(or $(LIMIT),10)

flywheel-review: ## Подтвердить метку: make flywheel-review ID=1 [DECISION=corrected CORRECTED=<slug>]
	PYTHONPATH=. $(PY) scripts/flywheel/cli.py review --id $(ID) \
	  --decision $(or $(DECISION),confirmed) --corrected-slug "$(CORRECTED)" --reviewer "$${USER}"

flywheel-dataset: ## Экспорт ТОЛЬКО подтверждённых меток в датасет
	PYTHONPATH=. $(PY) scripts/flywheel/cli.py export

flywheel-train: ## Переобучить голову на подтверждённых данных и записать версию в реестр
	PYTHONPATH=. $(PY) scripts/flywheel/cli.py train --epochs $(or $(EPOCHS),12)

flywheel-registry: ## Показать реестр версий модели
	PYTHONPATH=. $(PY) scripts/flywheel/cli.py registry

flywheel-eval: ## Оценить модель после переобучения (baseline_report)
	PYTHONPATH=. $(PY) scripts/flywheel/cli.py eval --samples $(or $(SAMPLES),200)

clean: ## Убрать кеши и артефакты сборки
	rm -rf reports/*.jsonl reports/*.report.json apps/web/.output apps/web/.nuxt
	find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

gate: ## Финальный quality/SLA-гейт: lint + pytest + eval (acc@1/p95) + живой контракт
	PYTHONPATH=. $(PY) scripts/eval/final_report.py --check --min-acc1 0.75 --max-p95 3000 --out reports/final.json
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) eval
	PYTHONPATH=. $(PY) scripts/eval/final_report.py --check --min-acc1 0.75 --max-p95 3000 --out reports/final.json
	PYTHONPATH=. $(PY) scripts/eval/check_contract.py

final-report: ## Итоговые метрики в reports/final.json
	PYTHONPATH=. $(PY) scripts/eval/final_report.py --out reports/final.json
