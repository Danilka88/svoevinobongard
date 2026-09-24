# Svoe Vino Scanner — GRACE v4 Project

Сервис распознавания российских вин по фото этикетки (конкурс РСХБ / ЛЦТ).
Фото → Tier-1 retrieval → (геоверификация) → VLM-fallback → `{"slug": "<wine-slug>"}` + confidence + top5.
Цель: 90–100% top-1, SLA < 3 c (жёсткий лимит скрипта организатора — 10 c).

## Keywords

wine label recognition, image retrieval, SigLIP 2, DINOv2, pgvector, FAISS,
geo-verification SuperPoint LightGlue, VLM fallback gemma4, OpenAI-compatible connector,
FastAPI gateway, Nuxt 3 mobile-first, F1@1 F1@5, active learning flywheel, ETL catalog.

## .grace Structure

Обязательные context-артефакты (имена фиксированы грамматикой GRACE 4):

- `.grace/context/requirements.xml` — `GraceRequirements`
- `.grace/context/technology.xml` — `GraceTechnology`
- `.grace/context/principles.xml` — `GracePrinciples`
- `.grace/context/deployment.xml` — `GraceDeployment` (обязателен `<Applicability>`)
- `.grace/context/ux-guidelines.xml` — `GraceUXGuidelines` (обязателен `<Applicability>`)
- `.grace/context/constraints.xml` — дополнительный реестр ограничений (вне канонического набора CLI)

Проекции:

- `.grace/graph/` — `index.xml` (`GraceGraphIndex`: `GraphDocuments` → `GD-*` с `Path` и `Owns`),
  routed `modules.xml` (`GD-001`) и `edges.xml` (`GD-002`)
- `.grace/verification/` — `index.xml` (`GraceVerificationIndex`: `VerificationDocuments` → `VD-*`),
  routed `VD-*.xml` (`GraceVerificationDocument`) с `V-M-*` entry
- `.grace/changes/active/` — `C-XXX/spec.xml` (`GraceChangeSpec`) + `C-XXX/plan.xml` (`GraceChangePlan`)
- `.grace/changes/archive/` — завершённые бандлы (`applied` / `rejected` / `cancelled` / `superseded`)

Идентификатор verification entry — `V-<module-id>`: для модуля `M-CATALOG` это `V-M-CATALOG`.
CLI требует покрытия каждого модуля графа.

## GRACE 4 Rules (обязательные)

### 1. Словарь

| Термин | Значение |
|---|---|
| **module** | Граница кода (`M-*`) с одним контрактом: `gateway`, `retrieval`, `vlm`, … |
| **contract** | `MODULE_CONTRACT` — PURPOSE / SCOPE / DEPENDS / LINKS. Пишется ДО кода. |
| **assertion** | Машино-проверяемая команда/условие. Стадии: `baseline` → `target` → `final`. |
| **scope** | Набор модулей/путей, которые change вправе менять. Вне scope — запрещено. |
| **change** | `C-XXX` = spec + plan. **Нет правок без change.** |

### 2. Иерархия источников контекста

1. `.grace/` — **источник истины** (requirements, graph, verification, changes).
2. `ARCHITECTURE.md` / `README.md` — человекочитаемое описание, вторично.
3. `my/` — только ЧТЕНИЕ как исторический источник требований (вне git, не редактировать).

Перед любой задачей: прочитать `.grace/context/` + `.grace/graph/index.xml`.

### 3. Обязательная семантическая разметка

Каждый исходный файл начинается с двух секций:

```python
# START_MODULE_CONTRACT
# PURPOSE: <одно предложение>
# SCOPE: <что входит>
# DEPENDS: <модули или "none">
# LINKS: M-<ID>, M-<ID>
# ROLE: RUNTIME | SCRIPT | TEST | BARREL | CONFIG | TYPES
# MAP_MODE: EXPORTS | LOCALS | SUMMARY | NONE
# END_MODULE_CONTRACT
# START_MODULE_MAP
# symbolName - краткое описание
# END_MODULE_MAP
```

Логические блоки оборачиваются парными маркерами с именем в UPPER_SNAKE:

```python
# START_BLOCK_<NAME>
...
# END_BLOCK_<NAME>
```

Правила:

- `ROLE` и `MAP_MODE` согласованы: RUNTIME→EXPORTS, SCRIPT/TEST→LOCALS, BARREL→SUMMARY,
  CONFIG→NONE, TYPES→EXPORTS.
- При `MAP_MODE: LOCALS|EXPORTS` перечислить **все** публичные top-level символы
  (для Python — функции, классы, присваивания); иначе `grace lint` даёт `module-map-mismatch`.
- Имена блоков уникальны в файле; маркеры парные; вложенность корректна.
- XML-декларация `<?xml ...?>` в `.grace/*.xml` **запрещена** — CLI принимает её за root-тег.
- Root-атрибуты: только `graceVersion="4.0"` (плюс `status` у change-артефактов).
- Семантические анкоры (`GD-`, `M-`, `V-M-`, `VD-`, `DF-`, `C-`, `T-`) — только XML-теги без атрибутов;
  в значениях атрибутов анкоры недопустимы.

### 3b. Проверка артефактов CLI

```bash
grace lint --path .                        # грамматика + разметка (0 errors — обязательное условие)
grace lint --path . --assertions current   # + baseline активных approved-change
grace status --path .                      # модули, verification, changes, drift
grace module show M-CATALOG --with verification --path .
grace file show scripts/catalog/etl_catalog.py --path .
```

Гейт для change: `grace lint --path . --change C-ID --assertions target --run-commands`.

### 4. Нет правок без change

- Любое изменение кода/артефактов — в рамках `C-XXX`.
- Перед стартом: `spec.xml` (status approved) → `plan.xml` (phases + assertions).
- Запрещено выходить за `scope` своего change; пересечение scope между активными changes недопустимо.
- Завершённый change уезжает в `.grace/changes/archive/C-XXX/`.

### 5. Assertion gates

- **baseline** — что должно быть зелёным ДО (lint, текущие тесты).
- **target** — что должно стать зелёным ПОСЛЕ (новые тесты, проверки).
- **final** — общий гейт проекта перед archive.
- Assertions — только машино-проверяемые команды, не расплывчатые формулировки.

### 6. Evidence-first (верификация)

- Нельзя заявлять «готово / работает / тесты прошли» без **реального вывода** инструмента (`bash`/`read`/test output) в этой же сессии.
- Перед завершением: (а) проверка реально запущена? (б) её вывод виден? (в) смежные/регрессионные области проверены?
- Нет вывода → нет заявления об успехе.

## Constraints (не нарушать)

- Только **permissive** лицензии (Apache-2.0 / MIT). AGPL (YOLOv8/v11 Ultralytics) запрещён.
- **Код из `old/` не переиспользуется** (clean-room). `old/`, `my/`, `материалы для конкурса/` не изменяются.
- **Данные из `old/` не используются нигде** — ни для обучения, ни для оценки, ни для демонстрации.
  Используются только официальные материалы конкурса; производные артефакты из old-данных удалены.
- `services/*` не импортируют друг друга — только HTTP/JSON через `gateway`.
- `apps/web` общается только с `gateway`. UI удаляем без влияния на ML.

## GRACE Commands

- `dd_grace-status` — health проекта
- `dd_grace-ask` — вопросы по проекту
- `dd_grace-spec` — создать change spec
- `dd_grace-plan` — архитектурное планирование (spec → plan)
- `dd_grace-execute` — исполнение плана (sequential / parallel-safe)
- `dd_grace-verification` — verification entries
- `dd_grace-fix` — отладка через граф
- `dd_grace-refresh` — синхронизация артефактов с кодом

## Roadmap → Changes

| Change | Этап | Цель |
|---|---|---|
Составлено по фактическому содержимому `.grace/changes/archive/` (40 архивных change, активных нет). Полный список: C-001…C-040.
Все изменения применялись одним циклом: spec → plan (approved) → execute → `grace lint` → target/final-гейты → archive.

| Change | Этап | Цель |
|---|---|---|
| C-001 | M0 | скелет репозитория, ETL каталога (2103 slug), self-evaluator |
| C-002 | M1 | Tier-1 retrieval: SigLIP2-so400m + обученная голова, numpy-индекс |
| C-003 | M2 | геоверификация ORB+RANSAC, калибровка порогов, SLA-контроль |
| C-004 | M3 | VLM-fallback и matcher fusion (VLM выключен: 34 с против SLA 3 с) |
| C-005 | M4 | Nuxt 3 UI: карточка, сомелье, аналоги, энотуризм |
| C-006 | M5 | flywheel: SQLite-логи, очередь ревью, HITL, экспорт датасета, реестр версий |
| C-007 | M6 | Docker/compose, Makefile, Yandex-манифесты, демо-документация |
| C-008 | M3 | LLM Router: fallback-цепочка провайдеров, circuit breaker, cost breaker |
| C-009 | M2 | движок точности: 0.175 → 0.495 на старом crop-пайплайне |
| C-010 | M3 | OCR-сигнал (RapidOCR): интеграция в матчер, нулевой прирост зафиксирован |
| C-011 | M0 | гигиена данных: полевой набор удалён, только 2 официальных фото, `docs/DATA.md` |
| C-012 | M1 | full-frame мультимодальный вход без кропа этикетки + P&S-тайлы |
| C-013 | M3 | label-text KB и гибридный текстовый retrieval (rapidfuzz), флаги |
| C-014 | M3 | условный текстовый фьюжн (off/conditional/always), кириллический OCR, оценка 26b |
| C-015 | M3 | римские годы в OCR, дефолт `conditional`, повторные замеры |
| C-016 | M5 | офлайн-обогащение 26b: 228 near-dup slug, 137 годов, 102 сладости |
| C-017 | M6 | синк официального API → `catalog_enriched_v3.jsonl` (2035 из 2103 slug) |
| C-018 | M4 | паритет карточки с порталом, сомелье на официальных `dishes`, аналоги через API |
| C-019 | M4 | энотуризм и статьи: 9 туров, 11 регионов, 180 статей + блоки UI |
| C-020 | M3 | OCR-движки: слияние латиница+кириллица, PaddleOCR-VL как офлайн-инструмент |
| C-021 | M2 | финальная калибровка порогов и quality/SLA-гейт `make gate` |
| C-022 | M1 | мультивью-галерея (`--views`), прототипы виноделен, оценка LightGlue |
| C-023 | M1 | OOD-валидация: holdout-семейство, вердикт об отсутствии обобщения |
| C-024 | M6 | синхронизация документации и проверка команд сдачи |
| C-025 | M6 | первая реальная полевая оценка: 21 магазинное фото, слабая GT по идентичности |
| C-026 | M1 | фикс дедупликации top-K по slug + распределение попаданий по глубине top-100 |
| C-027 | M3 | транслит лат↔кир по всему текстовому слою, 7 баг-фиксов второго аудита, потолок текста 28.6% |
| C-028 | M3 | text-first поиск по всему каталогу и RRF-слияние (recall top-100 → 100%, @1 хуже — флаг off) |
| C-029 | M1 | цветовой tie-breaker по палитрам CIELAB (эффекта нет — флаг off, посылка подтверждена) |
| C-030 | M2 | восемь баг-фиксов второго аудита с протоколом baseline → фикс → замер (без регрессий) |
| C-031 | M1 | цилиндрический unwrap по 6 точкам (нестабилен и ухудшает OCR — оставлен офлайн-инструментом) |
| C-032 | M3 | cross-encoder реранк bge-reranker-v2-m3 (вклада нет — флаг off, заметка о Qwen3-VL-Reranker) |
| C-033 | M6 | гигиена артефактов и документации: фактические пути тестов, статусы модулей, таблица изменений |
| C-034 | M4 | UI-бэкенд: медиа-обогащение (градиент/сорт/блюдо/регион), explain-трейс, `/v1/demo/metrics`, `/v1/catalog` |
| C-035 | M4 | дизайн-система (light/dark, токены портала, примитивы) и app-shell (сайдбар/нижняя навигация) |
| C-036 | M4 | экран «Сканер»: карточка на ассетах портала (градиент, бутылка, сорта, блюда, регион) |
| C-037 | M4 | экран «Сомелье»: мастер 3 шагов, sensory-«почему», альтернативы, персональный вкус (localStorage) |
| C-038 | M4 | экраны «Как это работает» (живой трейс/методы/флаги) и «Метрики» (дашборд, статус гейта) |
| C-039 | M4 | экран «Каталог»: поиск, фасетные фильтры, витрина, карточка по `/v1/wine/{slug}` |
| C-040 | M6 | гранулярный trace-этапы, полировка/доступность UI и полный e2e (5 маршрутов, API, gate) |

**Сводка по этапам:** M0 — каркас и данные (C-001, C-011); M1 — retrieval и галерея (C-002, C-012, C-022, C-023);
M2 — калибровка и точность (C-003, C-009, C-021, C-030); M3 — текстовый и VLM-сигнал (C-004, C-008, C-010,
C-013, C-014, C-015, C-020, C-027, C-028, C-032); M4 — UI и продуктовые фичи (C-005, C-018, C-019, C-034–C-039);
M5 — flywheel и обогащение (C-006, C-016); M6 — поставка (C-007, C-017, C-024, C-025, C-033, C-040); M1 — retrieval
и галерея (C-002, C-012, C-022, C-023, C-026, C-029, C-031).

## Слепой участок проверки целостности и регламент разметки

Проверка целостности (`grace lint`) видит только файлы, в которых **уже есть** семантическая
разметка: файл без `START_MODULE_CONTRACT` для неё невидим. Это делало проверку самореферентной —
дрейф из 13 тестов без разметки (C-043…C-058) накопился незамеченным.

**Регламент.** Каждый исходный `.py` в `packages/`, `services/`, `scripts/` и `tests/` начинается с
`START_MODULE_CONTRACT` … `END_MODULE_CONTRACT` и `START_MODULE_MAP` … `END_MODULE_MAP`:
`PURPOSE`, `SCOPE` (все публичные символы), `DEPENDS`, `LINKS` на `M-*`, `ROLE` (`RUNTIME` для кода,
`TEST` для тестов), `MAP_MODE` (`EXPORTS` для модулей с `__all__`, `LOCALS` для скриптов и тестов).
В `LOCALS`-карте перечисляются все публичные имена и **не** перечисляются приватные.

**Стража.** `tests/test_governance_markup.py` перебирает исходники и падает на любом файле без
разметки, поэтому слепой участок виден в `pytest` независимо от внешнего CLI. Из проверки исключены:
`data/models/**` (vendored модели Apache-2.0 в `.gitignore`), `.venv`, `node_modules`, `__pycache__`,
пустые `__init__.py` и файлы без кода.

**Нумерация изменений.** Номера change выдаются по порядку; пропуск фиксируется явно, а не
заполняется бандлом без работы. Пример: номер C-045 (поведенческий e2e-harness) поглощён изменением
C-046 — отметка в `<Note id="numbering-C045">` его спецификации.
