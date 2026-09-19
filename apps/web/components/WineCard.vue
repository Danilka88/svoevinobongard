<script setup lang="ts">
import type { Card } from '~/composables/useGateway';

// Метрики уверенности (F1/confidence) в UI не показываем — по ТЗ они видны только в API.
const props = defineProps<{ card: Card; slug: string }>();

// Основные чипы: категория, сладость, год, крепость, температура подачи.
const chips = computed(() => {
  const items: string[] = [];
  if (props.card.category) items.push(props.card.category);
  if (props.card.sweetness) items.push(props.card.sweetness);
  if (props.card.year) items.push(`${props.card.year}`);
  if (props.card.alcohol_pct) items.push(`${props.card.alcohol_pct}% об.`);
  if (props.card.temperature) items.push(`${props.card.temperature} °C`);
  return items;
});

// Сорта: список из официального API, иначе одиночное поле каталога.
const grapes = computed(() => {
  const list = (props.card.grapes || []).filter(Boolean);
  return list.length ? list : [props.card.grape].filter(Boolean) as string[];
});

// Ссылка на винодельню: официальный slug из API (портал отдаёт /wineries/{slug}).
const wineryHref = computed(() => {
  if (props.card.winery_url) return `https://vino-svoe.ru${props.card.winery_url}`;
  return '';
});
</script>

<template>
  <section class="card">
    <div class="wine-head">
      <img
        v-if="props.card.image_url"
        class="wine-photo"
        :src="props.card.image_url"
        :alt="props.card.name || props.slug"
        loading="lazy"
      />
      <div>
        <h2 class="wine-title">{{ props.card.name || props.slug }}</h2>
        <p class="wine-sub">
          <a v-if="wineryHref" :href="wineryHref" target="_blank" rel="noopener">{{ props.card.winery }}</a>
          <template v-else>{{ props.card.winery }}</template>
          <template v-if="props.card.region"> · {{ props.card.region }}</template>
        </p>
      </div>
    </div>

    <ul class="chips">
      <li v-for="chip in chips" :key="chip" class="chip">{{ chip }}</li>
    </ul>

    <p v-if="props.card.color" class="muted" style="margin-top: 10px">Цвет: {{ props.card.color }}</p>

    <div v-if="grapes.length" class="block">
      <h3 class="block-title">Сорта</h3>
      <ul class="chips">
        <li v-for="grape in grapes" :key="grape" class="chip chip-soft">{{ grape }}</li>
      </ul>
    </div>

    <div v-if="props.card.dishes?.length" class="block">
      <h3 class="block-title">Гастросочетания</h3>
      <ul class="chips">
        <li v-for="dish in props.card.dishes" :key="dish" class="chip chip-dish">{{ dish }}</li>
      </ul>
      <p class="muted small">Подобрано порталом «Своё Вино» — без выдумок модели.</p>
    </div>

    <p v-if="props.card.description" class="muted" style="margin-top: 14px">{{ props.card.description }}</p>
  </section>
</template>

<style scoped>
.wine-photo {
  width: 84px;
  height: 84px;
  object-fit: contain;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.04);
}
.block { margin-top: 16px; }
.block-title {
  margin: 0 0 8px;
  font-size: 0.82rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  opacity: 0.7;
}
.chip-soft { opacity: 0.85; }
.chip-dish { font-weight: 600; }
.small { font-size: 0.78rem; margin-top: 8px; }
</style>
