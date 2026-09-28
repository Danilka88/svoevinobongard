<script setup lang="ts">
import type { Card } from '~/composables/useGateway';
import { heroInk } from '~/utils/contrast';

// Карточка вина (C-036): hero с градиентом категории и бутылкой, характеристики, сорта
// с миниатюрами, гастрономия с картинками, регион с изображением. Метрики уверенности
// в UI не показываем — по ТЗ они видны только в API.
const props = defineProps<{ card: Card; slug: string }>();

const FALLBACK_GRADIENT = 'linear-gradient(135deg, var(--accent-strong), var(--accent))';

const heroStyle = computed(() => ({
  background: props.card.category_gradient || FALLBACK_GRADIENT,
}));
// C-096: цвет текста поверх градиента выбирается по контрасту — на светлых
// категорийных фонах портала белый текст невидим.
const ink = computed(() => heroInk(props.card.category_gradient || FALLBACK_GRADIENT));
const heroTextStyle = computed(() => ({ color: ink.value.title, '--hero-sub': ink.value.sub }));
const bottle = computed(() => props.card.image_url || '');
const chips = computed(() => {
  const items: string[] = [];
  if (props.card.category) items.push(props.card.category);
  if (props.card.sweetness) items.push(props.card.sweetness);
  if (props.card.year) items.push(`${props.card.year}`);
  if (props.card.alcohol_pct) items.push(`${props.card.alcohol_pct}% об.`);
  if (props.card.temperature) items.push(`${props.card.temperature} °C`);
  return items;
});
const grapeChips = computed(() => {
  const withImages = (props.card.grape_images || []).filter(item => item?.name);
  if (withImages.length) return withImages.map(item => ({ name: item.name, url: item.url }));
  return (props.card.grapes?.length ? props.card.grapes : [props.card.grape])
    .filter(Boolean).map(name => ({ name: String(name), url: '' }));
});
const dishes = computed(() => {
  const withImages = (props.card.dish_images || []).filter(item => item?.name);
  if (withImages.length) return withImages.map(item => ({ name: item.name, url: item.url }));
  return (props.card.dishes || []).filter(Boolean).map(name => ({ name: String(name), url: '' }));
});
const wineryHref = computed(() => (props.card.winery_url ? `https://vino-svoe.ru${props.card.winery_url}` : ''));
</script>

<template>
  <section class="card" style="padding: 0; overflow: hidden" aria-labelledby="wine-title">
    <div class="wine-hero" :style="heroStyle">
      <AppImage v-if="bottle" class="wine-hero-bottle" :src="bottle" :alt="props.card.name || props.slug" loading="lazy" />
      <div class="wine-hero-text" :style="heroTextStyle">
        <h2 id="wine-title" class="wine-title">{{ props.card.name || props.slug }}</h2>
        <p class="wine-sub">
          <a v-if="wineryHref" :href="wineryHref" target="_blank" rel="noopener">{{ props.card.winery }}</a>
          <template v-else>{{ props.card.winery }}</template>
          <template v-if="props.card.region"> · {{ props.card.region }}</template>
        </p>
        <p v-if="props.card.category" class="wine-hero-cat">
          {{ props.card.category }}
          <span v-if="props.card.year"> · {{ props.card.year }}</span>
        </p>
      </div>
    </div>

    <div style="padding: var(--space-5)">
      <ul class="chips">
        <li v-for="chip in chips" :key="chip" class="chip">{{ chip }}</li>
      </ul>

      <div v-if="grapeChips.length" class="block">
        <h3 class="block-title">Сорта</h3>
        <ul class="chips">
          <li v-for="grape in grapeChips" :key="grape.name" class="chip chip-grape">
            <AppImage v-if="grape.url" :src="grape.url" alt="" class="chip-thumb" loading="lazy" />
            {{ grape.name }}
          </li>
        </ul>
      </div>

      <div v-if="dishes.length" class="block">
        <h3 class="block-title">К чему подходит</h3>
        <ul class="dish-grid">
          <li v-for="dish in dishes" :key="dish.name" class="dish">
            <AppImage v-if="dish.url" :src="dish.url" :alt="dish.name" class="dish-photo" loading="lazy" />
            <span class="dish-name">{{ dish.name }}</span>
          </li>
        </ul>
        <p class="muted small" style="margin-top: 8px">Гастросочетания подобраны порталом «Своё Вино» — без выдумок модели.</p>
      </div>

      <div v-if="props.card.region_image_url || props.card.region_geojson_url" class="block">
        <h3 class="block-title">Регион</h3>
        <div class="region-card">
          <AppImage v-if="props.card.region_image_url" :src="props.card.region_image_url" :alt="props.card.region" class="region-photo" loading="lazy" />
          <div>
            <p style="margin: 0; font-weight: 600">{{ props.card.region || '—' }}</p>
            <p v-if="props.card.region_geojson_url" class="muted small" style="margin: 4px 0 0">
              Есть геометрия региона — пригодится для карты виноделен.
            </p>
          </div>
        </div>
      </div>

      <p v-if="props.card.color" class="muted small" style="margin-top: 14px">Цвет: {{ props.card.color }}</p>
      <p v-if="props.card.description" class="muted" style="margin-top: 12px">{{ props.card.description }}</p>
      <p v-if="!props.card.api_match" class="muted small" style="margin-top: 12px">
        Для этой позиции нет официальных данных портала — показаны поля каталога.
      </p>
    </div>
    <FeedbackButtons :slug="props.slug" />
  </section>
</template>

<style scoped>
.wine-hero {
  position: relative;
  display: flex; align-items: center; gap: var(--space-4);
  padding: var(--space-5);
  color: #fff;
  min-height: 176px;
}
.wine-hero-bottle {
  width: 96px; height: 140px; object-fit: contain; flex: none;
  filter: drop-shadow(0 12px 24px rgba(0, 0, 0, .35));
}
/* C-096: краска текста задаётся контрастом (heroInk), а не жёстким белым. */
.wine-hero-text { min-width: 0; }
.wine-hero .wine-title { color: inherit; margin: 0 0 4px; }
.wine-hero .wine-sub { color: var(--hero-sub, rgba(255, 255, 255, .9)); }
.wine-hero .wine-sub a { color: inherit; text-decoration: underline; }
.wine-hero-cat { margin: 8px 0 0; font-size: .8rem; letter-spacing: .04em; text-transform: uppercase; opacity: .85; }
.chip-thumb { width: 18px; height: 18px; border-radius: 50%; object-fit: cover; }
.chip-grape { padding-left: 6px; }
.dish-grid {
  list-style: none; padding: 0; margin: 0;
  display: grid; gap: var(--space-3); grid-template-columns: repeat(auto-fill, minmax(132px, 1fr));
}
.dish {
  display: flex; flex-direction: column; gap: 6px;
  border: 1px solid color-mix(in srgb, var(--gold) 30%, var(--border));
  background: var(--gold-soft);
  border-radius: var(--radius-sm); padding: var(--space-2); text-align: center;
}
.dish-photo { width: 100%; height: 92px; object-fit: cover; border-radius: var(--radius-sm); }
.dish-name { font-size: .82rem; font-weight: 600; }
.region-card { display: flex; gap: var(--space-3); align-items: center; }
.region-photo { width: 96px; height: 68px; object-fit: cover; border-radius: var(--radius-sm); flex: none; }
@media (max-width: 520px) {
  .wine-hero { flex-direction: column; align-items: flex-start; text-align: left; }
  .wine-hero-bottle { width: 84px; height: 118px; }
}
</style>
