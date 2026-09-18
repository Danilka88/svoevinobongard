# services/vlm

Единый **OpenAI-совместимый коннектор** к VLM/LLM.

Провайдеры (смена только через env):
`ollama | lmstudio | openrouter | yandex | custom`

```
VLM_BASE_URL=...   VLM_API_KEY=...   VLM_MODEL=gemma4:12b   VLM_MODEL_FAST=gemma4:e4b
```

Требования к ответу: `temperature=0`, строгая JSON-схема, короткий вывод —
**только поля для API** (winery, wine_name, vintage, wine_type, region, grape, confidence).
Constrained decoding (Outlines) + fallback-парсер/ремонт JSON.

Локально: Ollama `gemma4:12b` (основная), `gemma4:e4b` (быстрая).

Статус: скелет (M0), реализация — M3.
