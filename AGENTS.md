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
| C-001 | M0 | скелет репо, ETL каталога, F1-эвалюатор (archive) |
| C-002 | M1 | Tier-1 retrieval + нормализация + Postgres/pgvector |
| C-003 | M2 | геоверификация, margin, confidence, SLA |
| C-004 | M3 | VLM-fallback + matcher fusion |
| C-005 | M4 | Nuxt 3 UI + сомелье + аналоги + энотуризм |
| C-006 | M5 | flywheel (логи → ревью → аугментация → дообучение) |
| C-007 | M6 | Docker/Yandex deploy + видео-демо |
| C-008 | M3 | LLM Router (multi-provider fallback, circuit breaker) |
