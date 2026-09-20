<script setup lang="ts">
import type { ScanResponse, SimilarItem } from '~/composables/useGateway';

const { scan, similar } = useGateway();

const busy = ref(false);
const error = ref('');
const current = ref<ScanResponse | null>(null);
const similarItems = ref<SimilarItem[]>([]);
const similarLoading = ref(false);

async function onFile(file: File) {
  busy.value = true;
  error.value = '';
  current.value = null;
  similarItems.value = [];
  try {
    const result = await scan(file);
    current.value = result;
    if (result.slug && !result.not_found) {
      await loadSimilar(result.slug);
    }
  } catch (err) {
    error.value = 'Сервис недоступен. Проверьте, что gateway запущен, и попробуйте снова.';
  } finally {
    busy.value = false;
  }
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
  try {
    const card = await $fetch<ScanResponse['card']>(`${useGateway().base}/v1/scan`, { method: 'POST' });
    void card;
  } catch (err) {
    /* переход по кандидату без фото недоступен: показываем аналоги этого вина */
  }
  await loadSimilar(slug);
}
</script>

<template>
  <div>
    <div class="hero" style="margin-bottom: 16px">
      <h1>Своё Вино</h1>
      <p>Наведите камеру на этикетку — покажем карточку, подскажем подачу и похожие вина</p>
    </div>

    <div class="page">
      <ScanUploader :busy="busy" @selected="onFile" />

      <div v-if="error" class="toast-stack" role="status" aria-live="polite">
        <div class="toast">{{ error }}</div>
      </div>
      <section v-if="busy" class="card" aria-busy="true">
        <div class="skeleton skeleton-line" style="width: 50%" />
        <div class="skeleton skeleton-line" />
        <div class="skeleton" style="height: 120px" />
      </section>

      <template v-if="current">
        <WineCard v-if="current.card && !current.not_found" :card="current.card" :slug="current.slug || ''" />
        <NotFoundBlock v-else :candidates="current.top5" @pick="pick" />

        <SommelierPanel v-if="current.slug && !current.not_found" :slug="current.slug" />
        <SimilarWines :items="similarItems" :loading="similarLoading" @pick="pick" />
        <EnotourismBlock v-if="current.card && !current.not_found" :card="current.card" :slug="current.slug || ''" />
      </template>
    </div>
  </div>
</template>
