<script setup lang="ts">
import type { SimilarItem } from '~/composables/useGateway';

const props = defineProps<{ items: SimilarItem[]; loading?: boolean }>();
const emit = defineEmits<{ (event: 'pick', slug: string): void }>();
</script>

<template>
  <section class="card">
    <h2>Похожие вина из других виноделен</h2>
    <p class="muted">Подобраны по сорту, цвету и региону — чтобы было что попробовать рядом.</p>
    <div v-if="props.loading" class="skeleton" />
    <ul v-else-if="props.items.length" class="list">
      <li v-for="item in props.items" :key="item.slug" class="item">
        <div>
          <div class="name">{{ item.name || item.slug }}</div>
          <div class="why">{{ item.winery }} · {{ item.reason }}</div>
        </div>
        <button class="btn ghost" style="width: auto; min-height: 40px; padding: 0 14px"
                @click="emit('pick', item.slug)">Открыть</button>
      </li>
    </ul>
    <p v-else class="muted">Аналоги не найдены.</p>
  </section>
</template>
