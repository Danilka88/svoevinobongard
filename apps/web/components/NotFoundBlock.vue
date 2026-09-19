<script setup lang="ts">
import type { TopKItem } from '~/composables/useGateway';

const props = defineProps<{ candidates: TopKItem[] }>();
const emit = defineEmits<{ (event: 'pick', slug: string): void }>();
</script>

<template>
  <section class="card">
    <h2>Вино не найдено точно</h2>
    <p class="muted">
      Показываем ближайшие вина каталога — возможно, это одно из них. Уточните фото этикетки целиком,
      если ни одно не подходит.
    </p>
    <ul v-if="props.candidates.length" class="list">
      <li v-for="item in props.candidates" :key="item.slug" class="item">
        <div>
          <div class="name">{{ item.name || item.slug }}</div>
          <div class="why">{{ item.winery || '—' }}</div>
        </div>
        <button class="btn ghost" style="width: auto; min-height: 40px; padding: 0 14px"
                @click="emit('pick', item.slug)">Открыть</button>
      </li>
    </ul>
    <p v-else class="muted">Каталог не дал кандидатов — попробуйте другое фото.</p>
  </section>
</template>
