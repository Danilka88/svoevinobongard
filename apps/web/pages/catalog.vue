<script setup lang="ts">
import type { Card, EnotourismResponse } from '~/composables/useGateway';

// Экран каталога (C-039): поиск с задержкой, фильтры из фасетов, витрина с медиа
// и переход в полную карточку в нижней шторке. Тяжёлых моделей нет — только /v1/catalog
// и /v1/wine/{slug} через gateway.
useHead({ title: 'Каталог — Своё Вино' });

const gateway = useGateway();
const PER_PAGE = 24;

const query = ref('');
const region = ref('');
const category = ref('');
const grape = ref('');
const winery = ref('');
const sort = ref('name');

const items = ref<Record<string, any>[]>([]);
const facets = ref<Record<string, { value: string; count: number }[]>>({});
const total = ref(0);
const pages = ref(1);
const page = ref(1);
const loading = ref(false);
const loadingMore = ref(false);
const error = ref('');
// C-048/REG-001: отличаем сбой связи от пустой выдачи.
const unavailable = ref(false);

const sheetOpen = ref(false);
const sheetLoading = ref(false);
const sheetCard = ref<Card | null>(null);
const sheetSlug = ref('');
const sheetEnotourism = ref<EnotourismResponse | null>(null);

const hasFilters = computed(() => Boolean(query.value || region.value || category.value || grape.value || winery.value));
const canLoadMore = computed(() => page.value < pages.value);

async function load(nextPage = 1) {
  const more = nextPage > 1;
  more ? (loadingMore.value = true) : (loading.value = true);
  error.value = '';
  if (!more) unavailable.value = false;
  try {
    const response = await gateway.catalog({
      q: query.value, region: region.value, category: category.value,
      grape: grape.value, winery: winery.value, sort: sort.value,
      page: nextPage, per_page: PER_PAGE,
    });
    if (response.unavailable) {
      unavailable.value = true;
      error.value = 'Сервис недоступен: каталог не загрузился.';
      if (!more) items.value = [];
      return;
    }
    items.value = more ? [...items.value, ...(response.items || [])] : (response.items || []);
    facets.value = response.facets || {};
    total.value = response.total || 0;
    pages.value = response.pages || 1;
    page.value = response.page || nextPage;
  } catch {
    unavailable.value = true;
    error.value = 'Сервис недоступен: каталог не загрузился.';
    if (!more) items.value = [];
  } finally {
    loading.value = false;
    loadingMore.value = false;
  }
}

let timer: ReturnType<typeof setTimeout> | undefined;
watch([query, region, category, grape, winery, sort], () => {
  if (timer) clearTimeout(timer);
  timer = setTimeout(() => void load(1), 250);
});

function retry() {
  void load(1);
}

function reset() {
  query.value = '';
  region.value = '';
  category.value = '';
  grape.value = '';
  winery.value = '';
  sort.value = 'name';
  void load(1);
}

async function openCard(slug: string) {
  sheetSlug.value = slug;
  sheetOpen.value = true;
  sheetLoading.value = true;
  sheetCard.value = null;
  sheetEnotourism.value = null;
  try {
    const response = await gateway.wine(slug, true);
    sheetCard.value = response.card as Card;
    sheetEnotourism.value = (response.enotourism || null) as EnotourismResponse | null;
  } catch {
    sheetCard.value = null;
  } finally {
    sheetLoading.value = false;
  }
}

function closeCard() {
  sheetOpen.value = false;
  sheetCard.value = null;
  sheetEnotourism.value = null;
}

onMounted(() => { void load(1); });
</script>

<template>
  <div class="page">
    <div class="hero" style="margin-bottom: 16px">
      <h1>Каталог вин</h1>
      <p>2103 позиции с официальными данными портала: регион, винодельня, сорт, категория и изображение.</p>
    </div>

    <ServiceUnavailable v-if="unavailable" :detail="error" @retry="retry" />
    <div v-if="error" class="toast-stack" role="status" aria-live="polite"><div class="toast">{{ error }}</div></div>

    <!-- Поиск и фильтры -->
    <section class="card">
      <div class="field">
        <label class="field-label" for="q">Поиск по названию, винодельне или сорту</label>
        <input id="q" v-model="query" class="input" type="search" placeholder="например, рислинг или Шато Тамань" autocomplete="off">
      </div>

      <div class="filter-grid">
        <div class="field">
          <label class="field-label" for="f-region">Регион</label>
          <select id="f-region" v-model="region" class="select">
            <option value="">Любой</option>
            <option v-for="facet in facets.region || []" :key="facet.value" :value="facet.value">{{ facet.value }} ({{ facet.count }})</option>
          </select>
        </div>
        <div class="field">
          <label class="field-label" for="f-cat">Категория</label>
          <select id="f-cat" v-model="category" class="select">
            <option value="">Любая</option>
            <option v-for="facet in facets.category || []" :key="facet.value" :value="facet.value">{{ facet.value }} ({{ facet.count }})</option>
          </select>
        </div>
        <div class="field">
          <label class="field-label" for="f-grape">Сорт</label>
          <select id="f-grape" v-model="grape" class="select">
            <option value="">Любой</option>
            <option v-for="facet in (facets.grape || []).slice(0, 60)" :key="facet.value" :value="facet.value">{{ facet.value }} ({{ facet.count }})</option>
          </select>
        </div>
        <div class="field">
          <label class="field-label" for="f-winery">Винодельня</label>
          <select id="f-winery" v-model="winery" class="select">
            <option value="">Любая</option>
            <option v-for="facet in (facets.winery || []).slice(0, 80)" :key="facet.value" :value="facet.value">{{ facet.value }} ({{ facet.count }})</option>
          </select>
        </div>
      </div>

      <div class="row" style="margin-top: 8px">
        <span class="badge badge-gold" aria-live="polite">Найдено: {{ total }}</span>
        <button v-if="hasFilters" class="ghost" type="button" @click="reset">Сбросить</button>
        <button class="ghost" type="button" @click="load(1)">Обновить</button>
      </div>
    </section>

    <!-- Витрина -->
    <section v-if="loading" class="grid" aria-busy="true">
      <div v-for="n in 6" :key="`sk-${n}`" class="card" style="margin-top: 0">
        <div class="skeleton" style="height: 132px" />
        <div class="skeleton skeleton-line" style="width: 70%; margin-top: 10px" />
        <div class="skeleton skeleton-line" style="width: 40%" />
      </div>
    </section>

    <template v-else>
      <section v-if="items.length" class="grid">
        <article v-for="item in items" :key="item.slug" class="card wine-tile" style="margin-top: 0">
          <button class="tile-hit" type="button" :aria-label="`Открыть карточку ${item.name || item.slug}`" @click="openCard(item.slug)">
            <div class="tile-hero" :style="{ background: item.category_gradient || 'linear-gradient(135deg, var(--accent-strong), var(--accent))' }">
              <img v-if="item.image_url" :src="item.image_url" :alt="item.name" class="tile-photo" loading="lazy">
            </div>
            <div class="tile-body">
              <h2 class="tile-title">{{ item.name || item.slug }}</h2>
              <p class="muted small" style="margin: 2px 0 6px">{{ item.winery }}</p>
              <ul class="chips">
                <li v-if="item.region" class="chip">{{ item.region }}</li>
                <li v-if="item.grape" class="chip">{{ item.grape }}</li>
                <li v-if="item.year" class="chip">{{ item.year }}</li>
              </ul>
            </div>
          </button>
        </article>
      </section>

      <section v-else-if="!unavailable" class="card placeholder">Ничего не найдено. Попробуйте изменить запрос или сбросить фильтры.</section>

      <div v-if="canLoadMore" class="row" style="justify-content: center; margin-top: 16px">
        <button class="secondary" type="button" :disabled="loadingMore" @click="load(page + 1)">
          {{ loadingMore ? 'Загружаю…' : `Показать ещё (${total - items.length})` }}
        </button>
      </div>
    </template>

    <!-- Полная карточка -->
    <div v-if="sheetOpen" class="sheet-backdrop" @click.self="closeCard">
      <div class="sheet" role="dialog" aria-modal="true" :aria-label="`Карточка ${sheetSlug}`">
        <div class="sheet-grip" aria-hidden="true" />
        <div v-if="sheetLoading" class="stack"><div class="skeleton skeleton-line" style="width: 50%" /><div class="skeleton" style="height: 140px" /></div>
        <template v-else-if="sheetCard">
          <div class="sheet-hero" :style="{ background: sheetCard.category_gradient || 'linear-gradient(135deg, var(--accent-strong), var(--accent))' }">
            <img v-if="sheetCard.image_url" :src="sheetCard.image_url" :alt="sheetCard.name" class="sheet-photo">
            <div>
              <h2 class="wine-title" style="color: #fff; margin: 0 0 4px">{{ sheetCard.name }}</h2>
              <p style="color: rgba(255,255,255,.9); margin: 0">{{ sheetCard.winery }}<template v-if="sheetCard.region"> · {{ sheetCard.region }}</template></p>
            </div>
          </div>

          <ul class="chips">
            <li v-if="sheetCard.category" class="chip">{{ sheetCard.category }}</li>
            <li v-if="sheetCard.sweetness" class="chip">{{ sheetCard.sweetness }}</li>
            <li v-if="sheetCard.year" class="chip">{{ sheetCard.year }}</li>
            <li v-if="sheetCard.alcohol_pct" class="chip">{{ sheetCard.alcohol_pct }}% об.</li>
            <li v-if="sheetCard.temperature" class="chip">{{ sheetCard.temperature }} °C</li>
          </ul>

          <div v-if="sheetCard.grape_images?.length" class="block">
            <h3 class="block-title">Сорта</h3>
            <ul class="chips">
              <li v-for="grapeImage in sheetCard.grape_images" :key="grapeImage.name" class="chip">
                <img v-if="grapeImage.url" :src="grapeImage.url" alt="" class="mini"> {{ grapeImage.name }}
              </li>
            </ul>
          </div>

          <div v-if="sheetCard.dish_images?.length" class="block">
            <h3 class="block-title">К чему подходит</h3>
            <ul class="chips">
              <li v-for="dish in sheetCard.dish_images" :key="dish.name" class="chip chip-dish">
                <img v-if="dish.url" :src="dish.url" alt="" class="mini"> {{ dish.name }}
              </li>
            </ul>
          </div>

          <div v-if="sheetCard.region_image_url" class="block">
            <h3 class="block-title">Регион</h3>
            <div class="row">
              <img :src="sheetCard.region_image_url" :alt="sheetCard.region" class="region-mini">
              <div>
                <p style="margin: 0; font-weight: 600">{{ sheetCard.region }}</p>
                <p v-if="sheetEnotourism?.region?.fact" class="muted small" style="margin: 4px 0 0">{{ sheetEnotourism.region.fact }}</p>
              </div>
            </div>
          </div>

          <p v-if="sheetCard.description" class="muted" style="margin-top: 12px">{{ sheetCard.description }}</p>

          <div class="row" style="margin-top: 16px">
            <NuxtLink class="primary" :to="`/sommelier?slug=${encodeURIComponent(sheetSlug)}`" @click="closeCard">Сомелье от этого вина</NuxtLink>
            <a v-if="sheetCard.winery_url" class="secondary" :href="`https://vino-svoe.ru${sheetCard.winery_url}`" target="_blank" rel="noopener">Винодельня на портале</a>
            <button class="ghost" type="button" @click="closeCard">Закрыть</button>
          </div>
        </template>
        <p v-else class="placeholder">Не удалось загрузить карточку.</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.filter-grid { display: grid; gap: var(--space-3); grid-template-columns: 1fr; }
@media (min-width: 720px) { .filter-grid { grid-template-columns: 1fr 1fr; } }
.grid { display: grid; gap: var(--space-4); grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); margin-top: var(--space-4); }
.wine-tile { padding: 0; overflow: hidden; }
.tile-hit { display: block; width: 100%; text-align: left; background: transparent; border: 0; cursor: pointer; color: inherit; font: inherit; padding: 0; }
.tile-hero { display: grid; place-items: center; min-height: 148px; padding: var(--space-3); }
.tile-photo { max-height: 128px; object-fit: contain; filter: drop-shadow(0 10px 18px rgba(0,0,0,.3)); }
.tile-body { padding: var(--space-3) var(--space-3) var(--space-4); }
.tile-title { font-family: var(--font-serif); font-size: 1rem; margin: 0; }
.sheet-hero { display: flex; gap: var(--space-3); align-items: center; border-radius: var(--radius-sm); padding: var(--space-3); margin-bottom: var(--space-3); }
.sheet-photo { width: 84px; height: 118px; object-fit: contain; flex: none; }
.mini { width: 18px; height: 18px; border-radius: 50%; object-fit: cover; }
.region-mini { width: 96px; height: 64px; object-fit: cover; border-radius: var(--radius-sm); }
</style>
