<script setup lang="ts">
import type { Card } from '~/composables/useGateway';

const props = defineProps<{ card: Card }>();
</script>

<template>
  <section class="card">
    <h2>Энотуризм</h2>
    <p class="muted">
      Это вино делают
      <strong v-if="props.card.winery"> в «{{ props.card.winery }}»</strong>
      <strong v-if="props.card.region">, регион {{ props.card.region }}</strong>.
      Многие российские винодельни принимают гостей: дегустации, экскурсии по погребу и виды на виноградники.
    </p>
    <ul class="pairs">
      <li v-if="props.card.region" class="chip">Регион: {{ props.card.region }}</li>
      <li v-if="props.card.winery" class="chip">Винодельня: {{ props.card.winery }}</li>
    </ul>
    <p class="muted" style="margin-top: 10px">
      Совет: постройте маршрут по региону — обычно рядом несколько виноделен этого же региона.
    </p>
  </section>
</template>
