<script setup lang="ts">
import type { Card, EnotourismResponse } from '~/composables/useGateway';

// Энотуризм (C-035/C-036): картинка и факт региона, туры с адресом, ценой и изображением,
// «Истории в бокале» с картинками. Все данные — из официального API портала через gateway;
// при пустом ответе блок скрывается, чтобы не показывать пустые заголовки.
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
const region = computed(() => data.value?.region);
const hasContent = computed(() => Boolean(fact.value) || tours.value.length > 0 || articles.value.length > 0);
</script>

<template>
  <section v-if="loading || hasContent" class="card" aria-labelledby="enotourism-title">
    <h2 id="enotourism-title">Энотуризм</h2>

    <div v-if="loading" class="stack">
      <div class="skeleton skeleton-line" style="width: 60%" />
      <div class="skeleton skeleton-line" />
    </div>

    <template v-else>
      <div v-if="region?.image_url || fact" class="region-hero">
        <img v-if="region?.image_url" :src="region.image_url" :alt="region?.name || ''" class="region-hero-photo" loading="lazy">
        <div>
          <p class="region-hero-name">{{ region?.name }}</p>
          <p v-if="fact" class="muted small" style="margin: 4px 0 0">{{ fact }}</p>
          <ul class="chips" style="margin-top: 8px">
            <li v-if="region?.wines_count" class="chip">{{ region.wines_count }} вин в каталоге</li>
            <li v-if="region?.manufacturers_count" class="chip">{{ region.manufacturers_count }} виноделен</li>
            <li v-if="region?.events_count" class="chip">{{ region.events_count }} событий</li>
          </ul>
        </div>
      </div>

      <div v-if="tours.length" class="block">
        <h3 class="block-title">Винные туры</h3>
        <ul class="tour-grid">
          <li v-for="tour in tours" :key="tour.slug" class="tour">
            <img v-if="tour.image_url" :src="tour.image_url" :alt="tour.title" class="tour-photo" loading="lazy">
            <div>
              <a class="tour-title" :href="tour.url" target="_blank" rel="noopener">{{ tour.title || tour.slug }}</a>
              <p v-if="tour.address" class="muted small" style="margin: 4px 0 0">{{ tour.address }}</p>
              <ul class="chips" style="margin-top: 8px">
                <li v-if="tour.price" class="chip chip-accent">от {{ tour.price }} ₽</li>
                <li v-if="tour.match === 'winery'" class="chip">тур этой винодельни</li>
                <li v-else-if="tour.match === 'region'" class="chip">тур региона</li>
              </ul>
            </div>
          </li>
        </ul>
      </div>

      <ArticlesBlock :articles="articles" />
    </template>
  </section>
</template>

<style scoped>
.region-hero { display: flex; gap: var(--space-4); align-items: center; margin-top: var(--space-3); }
.region-hero-photo { width: 132px; height: 92px; object-fit: cover; border-radius: var(--radius-sm); flex: none; }
.region-hero-name { margin: 0; font-family: var(--font-serif); font-size: 1.1rem; }
.tour-grid { list-style: none; padding: 0; margin: var(--space-3) 0 0; display: grid; gap: var(--space-3); }
.tour { display: flex; gap: var(--space-3); border-left: 2px solid var(--border); padding-left: var(--space-3); }
.tour-photo { width: 72px; height: 72px; object-fit: cover; border-radius: var(--radius-sm); flex: none; }
.tour-title { font-weight: 600; }
@media (max-width: 520px) {
  .region-hero { flex-direction: column; align-items: flex-start; }
  .region-hero-photo { width: 100%; height: 140px; }
}
</style>
