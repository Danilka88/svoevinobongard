<script setup lang="ts">
// Обучение и качество (C-055): механика обратной связи для судей конкурса.
//
// Страница показывает полный цикл: спорные этикетки в очереди, состояние консенсуса между агентами
// и голосами, качество структурированного вывода моделей, рост датасета и условную панель модератора.
// Все данные аддитивные: страница только читает и отправляет решения, поиск и метрики не затрагивает.
useHead({ title: 'Обучение — Своё Вино' });

const gateway = useGateway();
const telemetry = useTelemetry();

const loading = ref(true);
const unavailable = ref(false);
const error = ref('');
const summary = ref<Record<string, any> | null>(null);
const queue = ref<Record<string, any>[]>([]);
const validation = ref<Record<string, number>>({});
const dataset = ref<Record<string, any> | null>(null);
const decided = ref<Record<string, string>>({});
const decisionBusy = ref('');

const REASON_LABELS: Record<string, string> = {
  wrong_wine: 'не то вино', wrong_year: 'неверный год', wrong_grape: 'не тот сорт',
  bad_photo: 'плохое фото', other: 'другое', rate_limited: 'превышен лимит частоты',
};

const rules = computed(() => summary.value?.rules || {});
const scenarios = computed(() => ({
  up: Number(summary.value?.up || 0),
  down: Number(summary.value?.down || 0),
  trusted: Number(summary.value?.trusted_targets || 0),
  queued: Number(summary.value?.queued_targets || 0),
}));
const reasons = computed(() => Object.entries(summary.value?.reasons || {}) as [string, number][]);
const validationRows = computed(() => ['exact', 'repaired', 'retried', 'failed', 'truncated']
  .map(key => ({ key, label: VALIDATION_LABELS[key] || key, value: Number(validation.value[key] || 0) })));
const VALIDATION_LABELS: Record<string, string> = {
  exact: 'Ответ сразу верный', repaired: 'Восстановлен из «грязного»',
  retried: 'Понадобился повтор', failed: 'Не разобран', truncated: 'Обрыв ответа',
};
const datasetRows = computed(() => (dataset.value?.splits || []) as { split: string; files: number; records: number }[]);

async function retry() {
  void load();
}

function reasonText(codes: string[]) {
  return (codes || []).map(code => REASON_LABELS[code] || code).join(', ') || 'без причины';
}

async function decide(item: Record<string, any>, decision: 'accept' | 'reject') {
  const key = item.slug || item.image_sha256 || 'case';
  decisionBusy.value = key;
  try {
    const result = await gateway.learningDecision({
      decision, slug: item.slug || '', image_sha256: item.image_sha256 || '',
      reason: decision === 'accept' ? 'подтверждено модератором' : 'отклонено модератором',
    });
    decided.value = { ...decided.value, [key]: decision };
    if (result?.summary) summary.value = result.summary;
    telemetry.track('action', { action: `moderator_${decision}`, value: key.slice(0, 40) });
  } catch {
    error.value = 'Не удалось сохранить решение. Попробуйте ещё раз.';
  } finally {
    decisionBusy.value = '';
  }
}

async function load() {
  loading.value = true;
  error.value = '';
  unavailable.value = false;
  try {
    const [summaryData, queueData, telemetryData, datasetData] = await Promise.all([
      gateway.learningSummary(), gateway.learningQueue(), gateway.telemetrySummary(), gateway.learningDataset(),
    ]);
    if (summaryData?.unavailable || queueData?.unavailable) {
      unavailable.value = true;
      error.value = 'Сервис недоступен: данные обучения не загружены.';
      return;
    }
    summary.value = summaryData;
    queue.value = queueData?.items || [];
    validation.value = telemetryData?.llm_validation || {};
    dataset.value = datasetData;
  } catch {
    unavailable.value = true;
    error.value = 'Сервис недоступен: данные обучения не загружены.';
  } finally {
    loading.value = false;
  }
}

onMounted(load);
</script>

<template>
  <div>
    <div class="hero" style="margin-bottom: 16px">
      <h1>Обучение и качество</h1>
      <p>Как обратная связь превращается в данные: очередь спорных этикеток, консенсус агентов, качество ответов моделей и рост датасета</p>
    </div>

    <div class="page">
      <ServiceUnavailable v-if="unavailable" :detail="error" @retry="retry" />
      <div v-if="error && !unavailable" class="toast-stack" role="status" aria-live="polite">
        <div class="toast">{{ error }}</div>
      </div>

      <section v-if="loading" class="card" aria-busy="true">
        <div class="skeleton skeleton-line" style="width: 40%" />
        <div class="skeleton skeleton-line" />
        <div class="skeleton" style="height: 120px" />
      </section>

      <template v-else-if="!unavailable">
        <section class="card" aria-labelledby="l-consensus">
          <h2 id="l-consensus">Состояние консенсуса</h2>
          <p class="muted">
            Одиночный голос не влияет на данные: вклад становится доверенным только если совпали
            <strong>≥{{ rules.min_sessions || 3 }} независимых сессий</strong> и голос согласуется с
            консенсусом агентов (ридеры и каталог). Остальное уходит модератору.
          </p>
          <div class="metrics-row">
            <div class="metric"><span class="metric-value">{{ scenarios.up }}</span><span class="metric-label">👍 верно</span></div>
            <div class="metric"><span class="metric-value">{{ scenarios.down }}</span><span class="metric-label">👎 не то</span></div>
            <div class="metric"><span class="metric-value">{{ scenarios.trusted }}</span><span class="metric-label">доверенных вкладов</span></div>
            <div class="metric"><span class="metric-value">{{ scenarios.queued }}</span><span class="metric-label">в очереди модератора</span></div>
          </div>
          <p class="muted small">
            Правила: сессий ≥ {{ rules.min_sessions || 3 }}, согласие ≥ {{ rules.agreement || 0.67 }},
            лимит {{ rules.rate_limit_per_minute || 6 }} голосов/мин на сессию.
          </p>
          <div v-if="reasons.length" class="chips">
            <span v-for="[code, count] in reasons" :key="code" class="chip-accent">
              {{ reasonText([code]) }}: {{ count }}
            </span>
          </div>
          <p v-else class="muted small">Пока нет голосов с причинами.</p>
        </section>

        <section class="card" aria-labelledby="l-queue">
          <h2 id="l-queue">Очередь спорных этикеток</h2>
          <p class="muted">Кейсы, которые не получили доверия: их видит модератор, автоматически они ни на что не влияют.</p>
          <ul v-if="queue.length" class="queue">
            <li v-for="item in queue" :key="item.slug + item.image_sha256" class="queue-item">
              <AppImage v-if="item.image_url" class="queue-photo" :src="item.image_url" :alt="item.slug" />
              <div class="queue-body">
                <p class="queue-title">{{ item.slug || 'вино не определено' }}</p>
                <p class="muted small">
                  Голосов: {{ item.up }} 👍 / {{ item.down }} 👎 · независимых сессий: {{ item.sessions }} ·
                  голос агентов: {{ item.agent_vote || 'нет' }}
                </p>
                <p class="muted small">
                  Причины: {{ reasonText((item.provenance || []).flatMap(p => p.reason || [])) }}
                </p>
                <p v-if="decided[item.slug || item.image_sha256]" class="small" role="status">
                  Решение модератора: {{ decided[item.slug || item.image_sha256] === 'accept' ? 'принято' : 'отклонено' }}
                </p>
                <div v-else class="queue-actions">
                  <button class="secondary" type="button" :disabled="decisionBusy === (item.slug || item.image_sha256)"
                          @click="decide(item, 'accept')">Принять</button>
                  <button class="ghost" type="button" :disabled="decisionBusy === (item.slug || item.image_sha256)"
                          @click="decide(item, 'reject')">Отклонить</button>
                </div>
              </div>
            </li>
          </ul>
          <p v-else class="muted small">Очередь пуста: все голоса согласованы с агентами.</p>
          <p class="moderator-flag">Раздел для модератора — демонстрационный, без реальной аутентификации</p>
        </section>

        <section class="card" aria-labelledby="l-llm">
          <h2 id="l-llm">Качество ответов моделей</h2>
          <p class="muted">Сколько ответов моделей пришло сразу в нужной форме, сколько пришлось восстанавливать и сколько не удалось разобрать.</p>
          <ul class="bars">
            <li v-for="row in validationRows" :key="row.key">
              <span class="bars-label">{{ row.label }}</span>
              <span class="bars-value">{{ row.value }}</span>
            </li>
          </ul>
        </section>

        <section class="card" aria-labelledby="l-dataset">
          <h2 id="l-dataset">Рост датасета</h2>
          <p class="muted">Принятое консенсусом идёт в датасет, спорное остаётся в holdout и никогда не попадает в обучение.</p>
          <div v-if="datasetRows.length" class="metrics-row">
            <div v-for="row in datasetRows" :key="row.split" class="metric">
              <span class="metric-value">{{ row.records }}</span>
              <span class="metric-label">{{ row.split }} · файлов {{ row.files }}</span>
            </div>
          </div>
          <p v-else class="muted small">Датасет пока пуст: разметка ещё не запускалась.</p>
          <p class="muted small">Решений модератора: {{ dataset?.decisions?.total || 0 }} (принято {{ dataset?.decisions?.accepted || 0 }}, отклонено {{ dataset?.decisions?.rejected || 0 }})</p>
        </section>
      </template>
    </div>
  </div>
</template>

<style scoped>
.metrics-row { display: flex; flex-wrap: wrap; gap: var(--space-3); margin: var(--space-3) 0; }
.metric { flex: 1 1 140px; border: 1px solid var(--border); border-radius: var(--radius-sm); padding: var(--space-3); background: var(--surface-2); }
.metric-value { display: block; font-size: 1.5rem; font-weight: 700; }
.metric-label { color: var(--text-muted); font-size: .84rem; }
.queue { list-style: none; padding: 0; margin: var(--space-3) 0 0; display: grid; gap: var(--space-3); }
.queue-item { display: flex; gap: var(--space-3); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: var(--space-3); background: var(--surface-2); }
.queue-photo { width: 72px; height: 72px; flex: none; object-fit: cover; border-radius: var(--radius-sm); }
.queue-body { min-width: 0; }
.queue-title { margin: 0 0 4px; font-weight: 600; }
.queue-actions { display: flex; gap: var(--space-2); margin-top: var(--space-2); }
.bars { list-style: none; padding: 0; margin: var(--space-3) 0 0; display: grid; gap: 6px; }
.bars li { display: flex; justify-content: space-between; border-bottom: 1px solid var(--border); padding-bottom: 4px; }
.bars-value { font-weight: 700; }
.moderator-flag { margin: var(--space-3) 0 0; padding: 6px 10px; border-radius: 999px; display: inline-block; background: var(--gold-soft); color: var(--gold-on-soft); font-size: .78rem; font-weight: 600; }
</style>
