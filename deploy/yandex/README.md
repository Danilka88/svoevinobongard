# deploy/yandex

Манифесты для развёртывания в Yandex Cloud.

Планируется:
- `*/container.yaml` — Serverless Container (Cloud Functions) на каждый сервис.
- `workflow/*.yaml` — оркестрация (пайплайн, триггеры переобучения).
- Object Storage — изображения/датасеты.
- Managed PostgreSQL + pgvector — каталог.
- Lockbox — секреты.

Статус: скелет (M0), реализация — M6.
