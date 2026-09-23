# apps/web/e2e — PandaScript UI-тесты (Lightpanda)

Автономные поведенческие тесты UI: загрузка страниц, навигация, формы, кнопки, доступность,
визуальная согласованность. Исполняются **без LLM** через `lightpanda run`.

## Что нужно поднять

| Что | Адрес | Команда |
|---|---|---|
| UI (Nuxt) | http://127.0.0.1:3000 | `cd apps/web && bun run build && NUXT_PUBLIC_API_BASE=http://127.0.0.1:8080 node .output/server/index.mjs` |
| gateway | http://127.0.0.1:8080 | `PYTHONPATH=. .venv/bin/uvicorn app.main:app --app-dir services/gateway --port 8080` |
| normalize / retrieval / matcher / ocr | 8081 / 8082 / 8084 / 8085 | см. README проекта |
| фикстуры (реальные фото этикеток) | http://127.0.0.1:8099 | `python3 -m http.server 8099 --bind 127.0.0.1 --directory data/samples/real100` |
| CDP Lightpanda (сбор консоли) | ws://127.0.0.1:9222 | `lightpanda serve --port 9222` |

## Запуск

```bash
bash apps/web/e2e/run_all.sh                      # все кейсы + снятие консоли
lightpanda run apps/web/e2e/smoke.js              # один кейс
node apps/web/e2e/tools/aggregate.mjs reports/e2e_ui_report.json
```

Кейсы `error-state.js` требует **остановленного** gateway:

```bash
kill $(lsof -ti tcp:8080) && lightpanda run apps/web/e2e/error-state.js
```

`catalog-search.js` нагружает gateway на ~15 с (известный баг BUG-001) — запускается последним,
после него gateway стоит перезапустить (в `run_all.sh` для этого есть `GW_RESTART_CMD`).

## Кейсы

| Файл | Поток | Что покрывает |
|---|---|---|
| `smoke.js` | навигация/оболочка | 5 маршрутов, меню/таб-бар, активное состояние, SPA-переход, базовый a11y |
| `scanner.js` | сканер + `/how` | загрузчик, выбор файла, карточка вина, «честный отказ», живой трейс |
| `sommelier.js` | сомелье | мастер 3 шагов, слайдеры, рекомендации, альтернативы, «Мой вкус» |
| `catalog.js` | каталог | витрина, фасетные фильтры, «показать ещё», bottom-sheet |
| `how-metrics.js` | метрики + `/how` | дашборд `/v1/demo/metrics`, бейдж гейта, статика `/how` |
| `theme.js` | тема | light/dark, localStorage, SPA-переход и перезагрузка |
| `error-state.js` | ошибки | gateway недоступен: сообщения, отсутствие белого экрана, unhandledrejection |
| `catalog-search.js` | регрессия поиска | задержка `/v1/catalog?q=` против SLA 3000 мс |

## Формат вывода

Каждый скрипт возвращает объект:

```json
{ "script": "smoke", "status": "PASS|FAIL", "checks": [ { "name": "...", "status": "PASS|FAIL|INFO", "detail": "..." } ] }
```

`tools/aggregate.mjs` собирает `results/*.json` (+ `results/console.json`, `results/analysis.json`)
в `reports/e2e_ui_report.json`.

## Известные ограничения инструмента

- **Viewport фиксирован 1920×1080** — мобильный layout через `lightpanda run` не проверить.
- **MCP `consoleLogs` не работает** в сборке `1.0.0-nightly.9638` — консоль снимается через CDP
  (`tools/cdp_console.mjs`: `Page.addScriptToEvaluateOnNewDocument` + `Runtime.consoleAPICalled` + `Log.entryAdded`).
- **Нет нативного file-picker** — файл собирается внутри страницы
  (`fetch → Blob → File → DataTransfer → input.files → change`), что покрывает UI-путь выбора.
- **Подресурсы не грузятся по умолчанию** (`--load-resources`), поэтому проверки видимости — по DOM.
- `page.evaluate` — единственный способ читать DOM из PandaScript; `page.waitForScript({script,timeout})`
  и `page.waitForSelector({selector,timeout})` принимают объект-аргумент.
