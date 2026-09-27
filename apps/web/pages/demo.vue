<script setup lang="ts">
// C-092: демонстрационный стенд «Примеры работы». Показывает на 21 проверенном полевом фото,
// как система работала: чтение этикетки → нормализация под каталог → кандидаты → итог.
// Данные статичны (public/demo/samples.json, сгенерированы один раз), поэтому стенд открывается
// мгновенно и не требует запуска backend.
useHead({ title: 'Примеры работы — Своё Вино' });

interface Field { label: string; value: string }
interface Sample {
  file: string; photo: string; scenario: string; status: string; human_label: string;
  expected_slug: string | null; reading: string; reading_fields: Field[]; normalized: Field[];
  normalized_source: string; candidates: string[]; text_candidates: string[];
  top1: string | null; not_found: boolean; path: string; latency_ms: number; reader_ms: number;
}
interface Payload { total: number; counts: Record<string, number>; metrics: Record<string, number>;
  samples: Sample[] }

const loading = ref(true);
const error = ref('');
const payload = ref<Payload | null>(null);
const filter = ref<string>('all');
const openFile = ref<string | null>(null);

const SCENARIOS: Record<string, { label: string; hint: string }> = {
  matched: { label: 'Однозначно найдено', hint: 'вино есть в каталоге и определено верно' },
  ambiguous: { label: 'Серия (несколько вариантов)', hint: 'этикетки одной серии почти не отличаются' },
  not_found: { label: 'Нет в каталоге', hint: 'иностранное вино или новинка, которой ещё нет' },
};
const PATH_LABELS: Record<string, string> = {
  llm_llm: 'LLM-ядро: чтение → нормализация → выбор',
  retrieval: 'классическое зрение',
  text_first: 'текст этикетки + зрение',
  observer: 'LLM-наблюдатель',
  vlm_fallback: 'зрение LLM',
};

const counts = computed(() => payload.value?.counts || {});
const filtered = computed(() => {
  const samples = payload.value?.samples || [];
  return filter.value === 'all' ? samples : samples.filter(s => s.scenario === filter.value);
});
const scenarioLabel = (key: string) => SCENARIOS[key]?.label || key;
const scenarioHint = (key: string) => SCENARIOS[key]?.hint || '';
const pathLabel = (key: string) => PATH_LABELS[key] || key;

function toggle(file: string) {
  openFile.value = openFile.value === file ? null : file;
}

onMounted(async () => {
  try {
    payload.value = await $fetch<Payload>('/demo/samples.json');
  } catch {
    error.value = 'Не удалось загрузить демонстрационные данные.';
  } finally {
    loading.value = false;
  }
});
</script>

<template>
  <div class="page">
    <div class="hero" style="margin-bottom: 16px">
      <h1>Примеры работы</h1>
      <p>
        Живая демонстрация на 21 проверенном полевом фото: как система читает этикетку, сводит её к каталогу
        и выбирает вино. Это статичный стенд — он не запускает модель и открывается мгновенно.
      </p>
    </div>

    <section v-if="loading" class="card" aria-busy="true">
      <div class="skeleton skeleton-line" style="width: 40%" />
      <div class="skeleton" style="height: 140px" />
    </section>

    <ServiceUnavailable v-if="error" :detail="error" action="Обновить" @retry="() => location.reload()" />

    <template v-if="payload && !loading">
      <section class="card">
        <div class="row" style="justify-content: space-between; align-items: baseline; gap: 12px">
          <h2 style="margin: 0">Как система работала на 21 фото</h2>
          <p class="muted small" style="margin: 0">показано {{ filtered.length }} из {{ payload.total }}</p>
        </div>
        <div class="segments" style="margin-top: 12px; flex-wrap: wrap">
          <button class="segment" :class="{ active: filter === 'all' }" type="button" @click="filter = 'all'">
            Все ({{ payload.total }})
          </button>
          <button v-for="(meta, key) in SCENARIOS" :key="key" class="segment"
                  :class="{ active: filter === key }" type="button" @click="filter = key">
            {{ meta.label }} ({{ counts[key] || 0 }})
          </button>
        </div>
        <p class="muted small" style="margin: 10px 0 0">{{ scenarioHint(filter) || 'Полный набор проверенных полевых фото.' }}</p>
      </section>

      <section v-for="sample in filtered" :key="sample.file" class="card demo-card">
        <div class="demo-head">
          <AppImage v-if="sample.photo" :src="`/demo/${sample.photo}`" :alt="sample.file" class="demo-photo" />
          <div class="demo-meta">
            <p class="state-badge" :class="`state-${sample.scenario === 'not_found' ? 'not_found' : (sample.scenario === 'ambiguous' ? 'uncertain' : 'ok')}`">
              {{ scenarioLabel(sample.scenario) }}
            </p>
            <h3 style="margin: 6px 0 2px">{{ sample.human_label || sample.file }}</h3>
            <p class="muted small" style="margin: 0 0 8px">
              Эталон: <code>{{ sample.expected_slug || 'вне каталога' }}</code>
            </p>
            <div class="demo-steps">
              <span class="demo-step"><b>1</b> Чтение 26b: {{ sample.reader_ms ? Math.round(sample.reader_ms) + ' мс' : '—' }}</span>
              <span class="demo-step"><b>2</b> Нормализация: {{ sample.normalized_source === 'llm' ? 'LLM' : (sample.normalized_source || '—') }}</span>
              <span class="demo-step"><b>3</b> Кандидатов: {{ sample.candidates.length + sample.text_candidates.length }}</span>
              <span class="demo-step"><b>4</b> {{ sample.not_found ? 'Честный отказ' : 'Выбрано вино' }}</span>
            </div>
            <p class="path-hint small" style="margin-top: 10px">
              Путь: <strong>{{ pathLabel(sample.path) }}</strong> · всего {{ Math.round(sample.latency_ms) }} мс
            </p>
          </div>
        </div>

        <div class="demo-grid">
          <div class="demo-block">
            <h4 class="recognized-title">Что прочитала модель</h4>
            <ul class="recognized-list">
              <li v-for="field in sample.reading_fields" :key="field.label">
                <span class="recognized-label">{{ field.label }}</span>
                <span class="recognized-value">{{ field.value }}</span>
              </li>
            </ul>
            <pre v-if="!sample.reading_fields.length" class="demo-pre">{{ sample.reading || '— (пусто)' }}</pre>
          </div>
          <div class="demo-block">
            <h4 class="recognized-title">Сверка с каталогом</h4>
            <ul v-if="sample.normalized.length" class="recognized-list">
              <li v-for="field in sample.normalized" :key="field.label">
                <span class="recognized-label">{{ field.label }}</span>
                <span class="recognized-value">{{ field.value }}</span>
              </li>
            </ul>
            <p v-else class="muted small">Бренд не распознан как каталожный — вероятно, вина нет на портале.</p>
          </div>
        </div>

        <details class="demo-details" @toggle="toggle(sample.file)">
          <summary>Показать кандидатов и результат</summary>
          <div class="demo-grid" style="margin-top: 10px">
            <div class="demo-block">
              <h4 class="recognized-title">Кандидаты (визуальные)</h4>
              <ul class="demo-cand">
                <li v-for="slug in sample.candidates" :key="slug"><code>{{ slug }}</code></li>
                <li v-if="!sample.candidates.length" class="muted small">нет</li>
              </ul>
            </div>
            <div class="demo-block">
              <h4 class="recognized-title">Кандидаты (текстовые)</h4>
              <ul class="demo-cand">
                <li v-for="slug in sample.text_candidates" :key="slug"><code>{{ slug }}</code></li>
                <li v-if="!sample.text_candidates.length" class="muted small">нет</li>
              </ul>
            </div>
          </div>
          <p class="demo-result" :class="{ ok: !sample.not_found }">
            Итог: <strong>{{ sample.not_found ? 'вино не найдено (честный отказ)' : (sample.top1 || '—') }}</strong>
          </p>
          <div class="row" style="margin-top: 10px">
            <NuxtLink v-if="!sample.not_found && sample.top1" class="secondary"
                      :to="`/catalog?slug=${encodeURIComponent(sample.top1)}`">
              Открыть карточку в каталоге
            </NuxtLink>
            <NuxtLink v-else-if="sample.human_label" class="ghost"
                      :to="`/catalog?q=${encodeURIComponent(sample.human_label)}`">
              Найти похожее в каталоге
            </NuxtLink>
          </div>
        </details>
      </section>
    </template>
  </div>
</template>

<style scoped>
.demo-card { display: flex; flex-direction: column; gap: var(--space-3); }
.demo-head { display: flex; gap: var(--space-4); }
.demo-photo { width: 120px; height: 160px; object-fit: contain; background: var(--surface-2); border-radius: var(--radius-sm); flex: none; }
.demo-meta { min-width: 0; flex: 1; }
.demo-steps { display: flex; flex-wrap: wrap; gap: 8px; }
.demo-step { font-size: .8rem; color: var(--text-muted); background: var(--surface-2); border: 1px solid var(--border); border-radius: 999px; padding: 3px 10px; }
.demo-step b { color: var(--accent); margin-right: 4px; }
.demo-grid { display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-4); }
.demo-block { min-width: 0; }
.demo-pre { white-space: pre-wrap; font-size: .82rem; background: var(--surface-2); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: var(--space-3); margin: 0; }
.demo-cand { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 4px; }
.demo-cand code { font-size: .78rem; word-break: break-all; }
.demo-details summary { cursor: pointer; color: var(--accent); font-size: .9rem; }
.demo-result { margin: var(--space-3) 0 0; }
.demo-result.ok strong { color: var(--success); }
/* Локальные стили статуса и полей — те же, что в NotFoundBlock, но в scoped-контексте стенда. */
.state-badge { display: inline-block; margin: 0; padding: 2px 10px; border-radius: 999px;
  font-size: .74rem; font-weight: 700; letter-spacing: .02em; text-transform: uppercase;
  background: var(--surface-2); color: var(--text-muted); }
.state-ok { background: var(--accent-soft); color: var(--accent); }
.state-not_found { background: var(--accent-soft); color: var(--accent); }
.state-uncertain { background: var(--surface-2); color: var(--text-muted); }
.recognized-title { margin: 0 0 var(--space-2); font-size: .95rem; }
.recognized-list { list-style: none; padding: var(--space-3); margin: 0; display: grid; gap: 6px;
  border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--surface-2); }
.recognized-list li { display: flex; gap: var(--space-3); justify-content: space-between; }
.recognized-label { color: var(--text-muted); font-size: .86rem; }
.recognized-value { font-weight: 600; text-align: right; }
@media (max-width: 620px) { .demo-head { flex-direction: column; } .demo-photo { width: 100%; height: 200px; } .demo-grid { grid-template-columns: 1fr; } }
</style>
