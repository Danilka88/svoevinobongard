<script setup lang="ts">
import type { Card, EnotourismResponse } from '~/composables/useGateway';

// Энотуризм: факт о регионе, винные туры с адресами и «Истории в бокале».
// Все данные — из официального API портала через gateway; при пустом ответе блок скрывается,
// чтобы не показывать пользователю пустые заголовки.
const props = defineProps<{ card: Card; slug: string }>();

const gateway = useGateway();
const data = ref<EnotourismResponse | null>(null);
const loading = ref(false);

async function load(slug: string) {
  if (!slug) { data.value = null; return; }
  loading.value = true;
  try {
    data.value = await gateway.enotourism(slug);
  } catch {
    data.value = null; // фича удержания не должна ломать скан
  } finally {
    loading.value = false;
  }
}

watch(() => props.slug, (slug) => { void load(slug); }, { immediate: true });

const tours = computed(() => data.value?.tours || []);
const articles = computed(() => data.value?.articles || []);
const fact = computed(() => data.value?.region?.fact || '');
const hasContent = computed(() => Boolean(fact.value) || tours.value.length > 0 || articles.value.length > 0);
</script>

<template>
  <section v-if="hasContent" class="card">
    <h2>Энотуризм</h2>

    <p class="muted">
      Это вино делают
      <strong v-if="props.card.winery"> в «{{ props.card.winery }}»</strong>
      <strong v-if="props.card.region">, регион {{ props.card.region }}</strong>.
    </p>

    <div v-if="fact" class="block">
      <h3 class="block-title">Регион</h3>
      <p class="muted">{{ fact }}</p>
      <p v-if="data?.region?.events_description" class="muted small">
        События региона: {{ data.region.events_description }}
        <template v-if="data?.region?.events_count"> ({{ data.region.events_count }})</template>
      </p>
    </div>

    <div v-if="tours.length" class="block">
      <h3 class="block-title">Винные туры</h3>
      <ul class="tours">
        <li v-for="tour in tours" :key="tour.slug" class="tour">
          <a class="tour-title" :href="tour.url" target="_blank" rel="noopener">{{ tour.title || tour.slug }}</a>
          <p v-if="tour.address" class="muted small">📍 {{ tour.address }}</p>
          <p class="muted small">
            <template v-if="tour.price">от {{ tour.price }} ₽</template>
            <template v-if="tour.match === 'winery'"> · тур этой винодельни</template>
            <template v-else-if="tour.match === 'region'"> · тур региона</template>
          </p>
        </li>
      </ul>
    </div>

    <ArticlesBlock :articles="articles" />
  </section>
</template>

<style scoped>
.block { margin-top: 16px; }
.block-title {
  margin: 0 0 6px;
  font-size: 0.82rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  opacity: 0.7;
}
.tours { list-style: none; padding: 0; margin: 10px 0 0; display: grid; gap: 12px; }
.tour { border-left: 2px solid rgba(255, 255, 255, 0.18); padding-left: 12px; }
.tour-title { font-weight: 600; text-decoration: none; color: inherit; }
.tour-title:hover { text-decoration: underline; }
.small { font-size: 0.78rem; margin: 4px 0 0; }
</style>
