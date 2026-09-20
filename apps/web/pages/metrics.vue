<script setup lang="ts">
// Экран метрик (C-038): дашборд качества из /v1/demo/metrics. Без внешних чарт-библиотек —
// полосы и кольца на CSS/SVG. Числа сопровождаются пояснением о домене измерений.
useHead({ title: 'Метрики — Своё Вино' });

const { demoMetrics } = useGateway();
const loading = ref(true);
const error = ref('');
const data = ref<Record<string, any> | null>(null);

const dashboard = computed(() => data.value?.dashboard || {});
const sources = computed(() => data.value?.sources || {});

function pct(value: unknown) {
  const number = Number(value);
  return Number.isFinite(number) ? `${Math.round(number * 1000) / 10}%` : '—';
}
function ms(value: unknown) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? `${Math.round(number)} мс` : '—';
}
function bar(value: unknown) {
  const number = Number(value);
  return `${Math.max(2, Math.min(100, Math.round((Number.isFinite(number) ? number : 0) * 100)))}%`;
}

const gateOk = computed(() => {
  const acc = Number(dashboard.value.proxy_acc_at_1);
  const p95 = Number(dashboard.value.latency_p95);
  const minAcc = Number(dashboard.value.gate_min_acc1 ?? 0.75);
  const sla = Number(dashboard.value.sla_ms ?? 3000);
  return Number.isFinite(acc) && Number.isFinite(p95) ? acc >= minAcc && p95 <= sla : null;
});
const slaShare = computed(() => {
  const p95 = Number(dashboard.value.latency_p95);
  const sla = Number(dashboard.value.sla_ms ?? 3000);
  return Number.isFinite(p95) && sla > 0 ? Math.min(100, Math.round((p95 / sla) * 100)) : 0;
});

const notes = [
  'Прокси-числа получены на эталонах каталога с синтетической аугментацией — это in-distribution, а не полевое качество.',
  'Реальные числа получены на 21 магазинном фото: имена файлов транслитерированы, поэтому часть промахов — верные попадания, и оценка занижена (нижняя граница).',
  'SLA взят из требований конкурса: p95 ответа не должен превышать 3000 мс.',
];

async function load() {
  loading.value = true;
  error.value = '';
  try {
    data.value = await demoMetrics();
  } catch {
    error.value = 'Gateway недоступен: метрики не загружены.';
    data.value = null;
  } finally {
    loading.value = false;
  }
}

onMounted(load);
</script>

<template>
  <div class="page">
    <div class="hero" style="margin-bottom: 16px">
      <h1>Метрики качества</h1>
      <p>Честные числа проекта: синтетический прокси, реальные полевые фото и границы применимости.</p>
    </div>

    <div v-if="error" class="toast-stack" role="status" aria-live="polite">
      <div class="toast">{{ error }}</div>
    </div>

    <section v-if="loading" class="card" aria-busy="true">
      <div class="skeleton skeleton-line" style="width: 40%" />
      <div class="skeleton" style="height: 90px" />
    </section>

    <template v-if="data && !loading">
      <section class="card">
        <div class="row">
          <h2 style="margin: 0">Состояние гейта</h2>
          <span v-if="gateOk === true" class="badge badge-success">пройден</span>
          <span v-else-if="gateOk === false" class="badge badge-error">не пройден</span>
          <span v-else class="badge">нет данных</span>
          <button class="ghost" type="button" @click="load">Обновить</button>
        </div>
        <p class="muted small" style="margin-top: 8px">
          Условия: accuracy@1 ≥ {{ dashboard.gate_min_acc1 ?? 0.75 }} и p95 ≤ {{ dashboard.sla_ms ?? 3000 }} мс.
        </p>
      </section>

      <section class="card">
        <h2>Прокси (эталоны + синтетика)</h2>
        <div class="metric-grid">
          <div class="metric">
            <p class="metric-label">accuracy@1</p>
            <p class="metric-value">{{ pct(dashboard.proxy_acc_at_1) }}</p>
            <span class="metric-bar"><i :style="{ width: bar(dashboard.proxy_acc_at_1) }" /></span>
          </div>
          <div class="metric">
            <p class="metric-label">accuracy@5</p>
            <p class="metric-value">{{ pct(dashboard.proxy_acc_at_5) }}</p>
            <span class="metric-bar"><i :style="{ width: bar(dashboard.proxy_acc_at_5) }" /></span>
          </div>
          <div class="metric">
            <p class="metric-label">near-dup @1</p>
            <p class="metric-value">{{ pct(dashboard.near_dup_acc_at_1) }}</p>
            <span class="metric-bar"><i :style="{ width: bar(dashboard.near_dup_acc_at_1) }" /></span>
          </div>
          <div class="metric">
            <p class="metric-label">latency p50 / p95</p>
            <p class="metric-value">{{ ms(dashboard.latency_p50) }} / {{ ms(dashboard.latency_p95) }}</p>
            <span class="metric-bar sla"><i :style="{ width: `${slaShare}%` }" /></span>
            <p class="muted small" style="margin: 4px 0 0">это {{ slaShare }}% от SLA</p>
          </div>
        </div>
      </section>

      <section class="card">
        <h2>Реальные полевые фото</h2>
        <div class="metric-grid">
          <div class="metric">
            <p class="metric-label">identity@1 (российские)</p>
            <p class="metric-value">{{ pct(dashboard.real_identity_at_1) }}</p>
            <span class="metric-bar real"><i :style="{ width: bar(dashboard.real_identity_at_1) }" /></span>
          </div>
          <div class="metric">
            <p class="metric-label">identity@5 (российские)</p>
            <p class="metric-value">{{ pct(dashboard.real_identity_at_5) }}</p>
            <span class="metric-bar real"><i :style="{ width: bar(dashboard.real_identity_at_5) }" /></span>
          </div>
          <div class="metric">
            <p class="metric-label">фото в замере</p>
            <p class="metric-value">{{ dashboard.real_samples ?? '—' }}</p>
          </div>
          <div class="metric">
            <p class="metric-label">p95 на реальных</p>
            <p class="metric-value">{{ ms(dashboard.real_latency_p95) }}</p>
          </div>
        </div>
      </section>

      <section class="card">
        <h2>Как читать эти числа</h2>
        <ul class="limits">
          <li v-for="note in notes" :key="note">{{ note }}</li>
        </ul>
        <p class="muted small" style="margin-top: 12px">
          Источники: <template v-for="(status, name) in sources" :key="name"><code>{{ name }}</code>: {{ status }}<template v-if="name !== Object.keys(sources).slice(-1)[0]"> · </template></template>
        </p>
      </section>
    </template>
  </div>
</template>

<style scoped>
.metric-grid { display: grid; gap: var(--space-3); grid-template-columns: 1fr; margin-top: var(--space-3); }
@media (min-width: 720px) { .metric-grid { grid-template-columns: 1fr 1fr; } }
.metric { border: 1px solid var(--border); border-radius: var(--radius-sm); padding: var(--space-3); background: var(--surface-2); }
.metric-label { margin: 0; font-size: .78rem; letter-spacing: .04em; text-transform: uppercase; color: var(--text-muted); }
.metric-value { margin: 4px 0 8px; font-size: 1.5rem; font-weight: 700; font-variant-numeric: tabular-nums; }
.metric-bar { display: block; height: 8px; border-radius: 999px; background: var(--border); overflow: hidden; }
.metric-bar i { display: block; height: 100%; background: linear-gradient(90deg, var(--accent), var(--gold)); }
.metric-bar.real i { background: linear-gradient(90deg, var(--gold), var(--success)); }
.metric-bar.sla i { background: var(--success); }
.limits { margin: 0; padding-left: 20px; display: grid; gap: 6px; font-size: .9rem; color: var(--text-muted); }
</style>
