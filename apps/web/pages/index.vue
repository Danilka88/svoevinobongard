<script setup lang="ts">
import type { ScanResponse, SimilarItem, TopKItem } from '~/composables/useGateway';

const { scan, similar } = useGateway();
const telemetry = useTelemetry();

const busy = ref(false);
const error = ref('');
// C-048/REG-001: сбой связи показываем как недоступность сервиса, а не как отказ распознавания.
const unavailable = ref(false);
let lastFile: File | null = null;
const current = ref<ScanResponse | null>(null);
const similarItems = ref<SimilarItem[]>([]);
const similarLoading = ref(false);

// C-067: при неуверенности показываем расширенную выдачу (до десяти кандидатов), а если её нет —
// прежнюю пятёрку: layout и существующие данные не ломаются.
const candidates = computed<TopKItem[]>(() => {
  const result = current.value;
  if (!result) return [];
  return (result.top10?.length ? result.top10 : result.top5) || [];
});

async function onFile(file: File) {
  busy.value = true;
  error.value = '';
  unavailable.value = false;
  lastFile = file;
  // Размер снимка в килобайтах: имя файла не отправляем, это пользовательские данные.
  telemetry.track('scan', { action: 'scan_start', value: Math.round(file.size / 1024) });
  current.value = null;
  similarItems.value = [];
  try {
    const result = await scan(file);
    current.value = result;
    telemetry.track('scan', {
      action: result.not_found ? 'scan_not_found' : 'scan_done',
      ok: !result.not_found,
      value: result.reason || result.path,
      detail: result.degraded ? 'degraded' : undefined,
    });
    if (result.slug && !result.not_found) {
      await loadSimilar(result.slug);
    }
  } catch (err) {
    unavailable.value = true;
    error.value = 'Сервис недоступен. Проверьте, что gateway запущен, и попробуйте снова.';
  } finally {
    busy.value = false;
  }
}

function retry() {
  if (lastFile) void onFile(lastFile);
}

async function loadSimilar(slug: string) {
  similarLoading.value = true;
  try {
    similarItems.value = await similar(slug);
  } catch (err) {
    similarItems.value = [];
  } finally {
    similarLoading.value = false;
  }
}

async function pick(slug: string) {
  telemetry.track('action', { action: 'pick_candidate', value: slug.slice(0, 40) });
  try {
    const card = await $fetch<ScanResponse['card']>(`${useGateway().base}/v1/scan`, { method: 'POST' });
    void card;
  } catch (err) {
    /* переход по кандидату без фото недоступен: показываем аналоги этого вина */
  }
  await loadSimilar(slug);
}

// C-091: путь решения понятными словами — видно, каким способом получен ответ
// (классическое зрение, OCR, режим наблюдателя или LLM-ядро), без изменения самого ответа.
const PATH_LABELS: Record<string, string> = {
  retrieval: 'классическое зрение',
  text_first: 'текст этикетки + зрение',
  ocr_fallback: 'OCR этикетки',
  vlm_fallback: 'зрение LLM',
  observer: 'LLM-наблюдатель',
  llm_match: 'LLM-ядро: чтение → нормализация → выбор',
  near_dup: 'различение серии',
  ambiguous: 'серия: несколько кандидатов',
};
const pathLabel = computed(() => PATH_LABELS[current.value?.path || ''] || current.value?.path || '');

</script>

<template>
  <div>
    <div class="hero" style="margin-bottom: 16px">
      <h1>Своё Вино</h1>
      <p>Наведите камеру на этикетку — покажем карточку, подскажем подачу и похожие вина</p>
    </div>

    <div class="page">
      <ScanUploader :busy="busy" @selected="onFile" />

      <ServiceUnavailable v-if="unavailable" :detail="error" action="Повторить снимок" @retry="retry" />

      <div v-if="error" class="toast-stack" role="status" aria-live="polite">
        <div class="toast">{{ error }}</div>
      </div>
      <section v-if="busy" class="card" aria-busy="true">
        <div class="skeleton skeleton-line" style="width: 50%" />
        <div class="skeleton skeleton-line" />
        <div class="skeleton" style="height: 120px" />
      </section>

      <template v-if="current">
        <p v-if="pathLabel" class="path-hint muted small" role="status">
          Как получен ответ: <strong>{{ pathLabel }}</strong>
          <template v-if="current.latency_ms"> · {{ Math.round(current.latency_ms) }} мс</template>
        </p>
        <WineCard v-if="current.card && !current.not_found" :card="current.card" :slug="current.slug || ''" />
        <NotFoundBlock
          v-else
          :candidates="candidates"
          :alternatives="current.alternatives || []"
          :recognized-fields="current.recognized_fields || {}"
          :normalization="current.normalization || {}"
          :reason="current.reason || ''"
          :degraded="current.degraded"
          :not-found="current.not_found"
          :ambiguous="current.ambiguous || false"
          @pick="pick"
        />

        <SommelierPanel v-if="current.slug && !current.not_found" :slug="current.slug" />
        <SimilarWines :items="similarItems" :loading="similarLoading" @pick="pick" />
        <EnotourismBlock v-if="current.card && !current.not_found" :card="current.card" :slug="current.slug || ''" />
      </template>
    </div>
  </div>
</template>
