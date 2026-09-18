# services/active-learning

Data flywheel — самоулучшение Tier-1.

1. Лог каждого запроса (хэш изображения, эмбеддинг, top-K, скоры, путь, итог, latency).
2. Очередь на ревью (human-in-the-loop).
3. Синтетическая аугментация вырезов → «полевые» фото (Albumentations).
4. Дообучение Tier-1 (contrastive / LoRA).
5. Оценка F1@1/@5, реестр версий (MLflow/DVC), деплой.

Триггер: N новых образцов или расписание (Yandex Cloud Workflow / cron).

Статус: скелет (M0), реализация — M5.
