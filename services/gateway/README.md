# services/gateway

Публичный API и оркестрация пайплайна. **Единственная точка входа** для UI и скрипта оценки.

Эндпоинты: `/v1/eval/predict`, `/v1/scan`, `/v1/similar/{slug}`, `/v1/sommelier`, `/v1/health`.

Запуск локально:
```bash
pip install -r services/gateway/requirements.txt
uvicorn app.main:app --reload --port 8080   # из services/gateway
```
См. `docker-compose.yml`.

Статус: скелет (M0). Пайплайн подключается в M1+.
