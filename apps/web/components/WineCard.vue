<script setup lang="ts">
import type { Card } from '~/composables/useGateway';

// Метрики уверенности (F1/confidence) в UI не показываем — по ТЗ они видны только в API.
const props = defineProps<{ card: Card; slug: string }>();

const chips = computed(() => {
  const items: string[] = [];
  if (props.card.category) items.push(props.card.category);
  if (props.card.sweetness) items.push(props.card.sweetness);
  if (props.card.year) items.push(`${props.card.year}`);
  if (props.card.grape) items.push(props.card.grape);
  return items;
});
</script>

<template>
  <section class="card">
    <div class="wine-head">
      <div>
        <h2 class="wine-title">{{ props.card.name || props.slug }}</h2>
        <p class="wine-sub">{{ props.card.winery }}<template v-if="props.card.region"> · {{ props.card.region }}</template></p>
      </div>
    </div>

    <ul class="chips">
      <li v-for="chip in chips" :key="chip" class="chip">{{ chip }}</li>
    </ul>

    <p v-if="props.card.description" class="muted" style="margin-top: 14px">{{ props.card.description }}</p>
  </section>
</template>
