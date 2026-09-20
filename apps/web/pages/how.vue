<script setup lang="ts">
import type { ScanResponse } from '~/composables/useGateway';

// Экран прозрачности (C-038): живой трейс пути решения + статичная схема + честные
// ограничения + витрина проверенных методов. Числа методов взяты из наших отчётов.
useHead({ title: 'Как это работает — Своё Вино' });

const { scanExplain } = useGateway();
const busy = ref(false);
const error = ref('');
const result = ref<ScanResponse | null>(null);

const trace = computed(() => result.value?.trace || {});
const hasTrace = computed(() => Boolean((trace.value.stages || []).length));
const stages = computed(() => (trace.value.stages || []).filter(stage => stage.latency_ms > 0));
const flags = computed(() => Object.entries(trace.value.flags || {}));
const methods = computed(() => Object.entries(trace.value.methods || {}));
const topK = computed(() => trace.value.top_k || []);
const maxScore = computed(() => Math.max(...topK.value.map(item => item.score || 0), 0.0001));

const flagNames: Record<string, string> = {
  TEXT_FUSION_MODE: 'Текстовый фьюжн',
  TEXT_FIRST_ENABLED: 'Текст как источник кандидатов',
  TEXT_CE_ENABLED: 'Cross-encoder реранк',
  COLOR_TIEBREAK_ENABLED: 'Цветовой tie-breaker',
  CYLINDER_UNWRAP_ENABLED: 'Разворот цилиндра',
  VLM_FALLBACK_ENABLED: 'VLM в онлайне',
  OCR_ENGINE: 'OCR-движок',
  NORMALIZE_CROP: 'Кроп этикетки',
  VIEWS: 'Видов в галерее',
};
const flagLabel = (key: string) => flagNames[key] || key;
const flagValue = (value: unknown) => (typeof value === 'boolean' ? (value ? 'включено' : 'выключено') : String(value));

const pipeline = [
  { title: 'Кадр целиком', body: 'Кропа этикетки нет: letterbox и плитки покрывают весь кадр, поэтому овальные этикетки, блики и ракурсы не ломают вход.' },
  { title: 'Tier-1: визуальный поиск', body: 'Сигнатура SigLIP2 + обученная голова, галерея из нескольких полевых видов на позицию. Дёшево и детерминированно.' },
  { title: 'Геоверификация ORB + RANSAC', body: 'Проверяет геометрию совпадения по всем видам кандидата: разводит похожие серии одного бренда.' },
  { title: 'Текст с этикетки', body: 'OCR включается только когда визуальные скоры близки: транслит лат↔кир и алиасы сортов приводят обе стороны к одному виду.' },
  { title: 'Решение и честный отказ', body: 'Пороги уверенности и запаса; если сигнала нет — система сообщает not_found, а не угадывает.' },
];

const methodsReview = [
  { name: 'Текстовый фьюжн (RRF)', verdict: 'принято', detail: 'условный режим: 32 срабатывания из 150, улучшений 2, ухудшений 0', source: 'reports/text_fusion.json' },
  { name: 'Текст как источник кандидатов', verdict: 'отклонено как дефолт', detail: 'top-100 достижимость 92.9% → 100%, но identity@1 падала 0.714 → 0.571', source: 'reports/text_first.json' },
  { name: 'Cross-encoder bge-reranker-v2-m3', verdict: 'отклонено', detail: 'нулевой вклад: ce_text и ce_visual не изменили ранги, ce_both ухудшил @1', source: 'reports/text_ce.json' },
  { name: 'Цветовой tie-breaker', verdict: 'отклонено', detail: 'при безопасном весе 0 изменённых рангов, при сильном — только ухудшения', source: 'reports/color_tiebreak.json' },
  { name: 'Разворот цилиндра (unwrap)', verdict: 'оставлено офлайн', detail: 'автопоиск маркеров сработал на 6 из 21 фото, чтение ухудшилось (потолок 6 → 0)', source: 'reports/cylinder.json' },
  { name: 'Мультивью-галерея (6 видов)', verdict: 'принято', detail: 'acc@1 0.675 → 0.785 на тех же 200 сэмплах', source: 'reports/final.json' },
];

const limits = [
  'Полевых фото в обучении нет: набор удалён ради чистоты данных, поэтому качество измерено на эталонах и синтетике.',
  'Синтетический OOD пессимистичнее реальности, а прокси оптимистичнее: ни один синтетический замер не предсказал реальное качество.',
  'Локальный VLM работает десятки секунд на кадр, поэтому в онлайне он выключен и используется только офлайн.',
  'Часть промахов на реальных фото — следствие слабой разметки (имена файлов транслитерированы), то есть числа занижены.',
];

async function onFile(file: File) {
  busy.value = true;
  error.value = '';
  result.value = null;
  try {
    result.value = await scanExplain(file);
  } catch {
    error.value = 'Не удалось получить трейс: проверьте, что gateway и сервисы запущены.';
  } finally {
    busy.value = false;
  }
}
</script>

<template>
  <div class="page">
    <div class="hero" style="margin-bottom: 16px">
      <h1>Как это работает</h1>
      <p>Прозрачный путь кадра: что происходит, сколько занимает и почему методы стоят именно в таком порядке.</p>
    </div>

    <ScanUploader :busy="busy" @selected="onFile" />

    <div v-if="error" class="toast-stack" role="status" aria-live="polite"><div class="toast">{{ error }}</div></div>

    <section v-if="busy" class="card" aria-busy="true">
      <div class="skeleton skeleton-line" style="width: 45%" />
      <div class="skeleton skeleton-line" />
    </section>

    <!-- Живой трейс -->
    <section v-if="result" class="card" aria-live="polite">
      <h2>Живой трейс этого кадра</h2>
      <p class="muted small">
        Решение: <strong>{{ result.slug || 'отказ' }}</strong> · path {{ result.path }} ·
        уверенность {{ result.confidence }} · запас {{ result.margin }} · всего {{ result.latency_ms }} мс
      </p>

      <template v-if="hasTrace">
        <h3 class="block-title" style="margin-top: 16px">Этапы</h3>
        <ol class="timeline">
          <li v-for="stage in stages" :key="stage.name">
            <span class="dot" :class="{ slow: stage.latency_ms > 300 }" aria-hidden="true" />
            <span class="stage-name">{{ stage.name }}</span>
            <span class="stage-ms">{{ stage.latency_ms }} мс</span>
          </li>
        </ol>

        <h3 class="block-title" style="margin-top: 16px">Методы</h3>
        <ul class="chips">
          <li v-for="[key, value] in methods" :key="key" class="chip">{{ key }}: {{ value }}</li>
        </ul>

        <h3 class="block-title" style="margin-top: 16px">Флаги</h3>
        <ul class="flag-list">
          <li v-for="[key, value] in flags" :key="key">
            <span>{{ flagLabel(key) }}</span><code>{{ flagValue(value) }}</code>
          </li>
        </ul>

        <h3 class="block-title" style="margin-top: 16px">Кандидаты</h3>
        <ul class="k-list">
          <li v-for="(item, index) in topK" :key="item.slug">
            <span class="k-rank">{{ index + 1 }}</span>
            <span class="k-slug">{{ item.slug }}</span>
            <span class="k-scores">
              {{ item.score.toFixed(3) }}
              <template v-if="item.geo_score !== undefined"> · geo {{ item.geo_score }}</template>
              <template v-if="item.inliers"> · {{ item.inliers }} инл.</template>
            </span>
            <span class="k-bar"><i :style="{ width: `${Math.max(4, Math.round((item.score / maxScore) * 100))}%` }" /></span>
          </li>
        </ul>

        <p v-if="trace.notes?.length" class="muted small" style="margin-top: 12px">
          Заметки: {{ trace.notes.join('; ') }}
        </p>
      </template>

      <p v-else class="placeholder" style="margin-top: 12px">
        Трейс недоступен: витрина выключена (EXPLAIN_ENABLED=0) или gateway старой версии.
        Ниже — статичная схема пайплайна.
      </p>
    </section>

    <!-- Статичная схема -->
    <section class="card">
      <h2>Схема пайплайна</h2>
      <p class="muted small">Порядок не случаен: сначала дешёвое и детерминированное, потом дорогое — только если нужно.</p>
      <ol class="pipe">
        <li v-for="(step, index) in pipeline" :key="step.title">
          <span class="pipe-num">{{ index + 1 }}</span>
          <div>
            <p class="pipe-title">{{ step.title }}</p>
            <p class="muted small" style="margin: 2px 0 0">{{ step.body }}</p>
          </div>
        </li>
      </ol>
    </section>

    <!-- Витрина методов -->
    <section class="card">
      <h2>Что проверяли и чем закончилось</h2>
      <p class="muted small">Каждый метод измерен на одних и тех же данных; числа — из наших отчётов.</p>
      <ul class="method-grid">
        <li v-for="method in methodsReview" :key="method.name" class="method">
          <p class="method-name">{{ method.name }}</p>
          <span class="badge" :class="method.verdict === 'принято' ? 'badge-success' : (method.verdict === 'оставлено офлайн' ? 'badge-warning' : 'badge-error')">
            {{ method.verdict }}
          </span>
          <p class="muted small" style="margin: 8px 0 0">{{ method.detail }}</p>
          <p class="muted small" style="margin: 4px 0 0"><code>{{ method.source }}</code></p>
        </li>
      </ul>
    </section>

    <!-- Ограничения -->
    <section class="card">
      <h2>Честные ограничения</h2>
      <ul class="limits">
        <li v-for="limit in limits" :key="limit">{{ limit }}</li>
      </ul>
    </section>
  </div>
</template>

<style scoped>
.timeline { list-style: none; padding: 0; margin: var(--space-2) 0 0; display: grid; gap: 6px; }
.timeline li { display: flex; align-items: center; gap: var(--space-3); font-size: .9rem; }
.dot { width: 10px; height: 10px; border-radius: 50%; background: var(--success); flex: none; }
.dot.slow { background: var(--warning); }
.stage-name { flex: 1; }
.stage-ms { font-variant-numeric: tabular-nums; color: var(--text-muted); }
.flag-list { list-style: none; padding: 0; margin: 0; display: grid; gap: 6px; }
.flag-list li { display: flex; justify-content: space-between; gap: var(--space-3); font-size: .86rem; border-bottom: 1px dashed var(--border); padding-bottom: 4px; }
.flag-list code { color: var(--accent); }
.k-list { list-style: none; padding: 0; margin: 0; display: grid; gap: 8px; }
.k-list li { display: grid; grid-template-columns: 22px 1fr auto; gap: var(--space-2); align-items: center; font-size: .84rem; }
.k-rank { color: var(--text-muted); }
.k-slug { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.k-scores { font-variant-numeric: tabular-nums; color: var(--text-muted); }
.k-bar { grid-column: 2 / 4; height: 6px; border-radius: 999px; background: var(--border); overflow: hidden; }
.k-bar i { display: block; height: 100%; background: linear-gradient(90deg, var(--accent), var(--gold)); }
.pipe { list-style: none; padding: 0; margin: var(--space-3) 0 0; display: grid; gap: var(--space-3); }
.pipe li { display: flex; gap: var(--space-3); }
.pipe-num { width: 26px; height: 26px; flex: none; border-radius: 50%; display: grid; place-items: center;
  background: var(--accent-soft); color: var(--accent); font-weight: 700; font-size: .8rem; }
.pipe-title { margin: 0; font-weight: 600; }
.method-grid { list-style: none; padding: 0; margin: var(--space-3) 0 0; display: grid; gap: var(--space-3); }
@media (min-width: 760px) { .method-grid { grid-template-columns: 1fr 1fr; } }
.method { border: 1px solid var(--border); border-radius: var(--radius-sm); padding: var(--space-3); background: var(--surface-2); }
.method-name { margin: 0 0 6px; font-weight: 600; }
.limits { margin: 0; padding-left: 20px; display: grid; gap: 6px; font-size: .9rem; color: var(--text-muted); }
</style>
