<script setup lang="ts">
import type { TopKItem } from '~/composables/useGateway';

// Честный отказ (C-036): объясняем, почему не уверены, и показываем кандидатов
// с их относительными скорами — решение остаётся за человеком.
const props = defineProps<{ candidates: TopKItem[] }>();
const emit = defineEmits<{ (event: 'pick', slug: string): void }>();

const top = computed(() => (props.candidates || []).slice(0, 5));
const maxScore = computed(() => Math.max(...top.value.map(item => Number(item.score) || 0), 0.0001));

function width(score: number) {
  return `${Math.max(6, Math.round((Number(score) / maxScore.value) * 100))}%`;
}
</script>

<template>
  <section class="card" aria-labelledby="nf-title">
    <h2 id="nf-title">Не уверен в ответе</h2>
    <p class="muted">
      Этикетка прочитана, но уверенности для точного ответа не хватило: возможно, вина нет в каталоге
      (например, оно не российское), либо снимок под неудобным углом. Ниже — ближайшие варианты: если
      среди них есть ваше вино, выберите его, и карточка откроется.
    </p>

    <ul class="chips" v-if="top.length">
      <li class="chip">Проверьте бренд и год</li>
      <li class="chip">Снимите этикетку целиком, без обрезки</li>
    </ul>

    <ul class="candidates" v-if="top.length">
      <li v-for="(item, index) in top" :key="item.slug" class="candidate">
        <div class="candidate-head">
          <span class="candidate-rank">{{ index + 1 }}</span>
          <div style="min-width: 0">
            <p class="candidate-name">{{ item.name || item.slug }}</p>
            <p class="candidate-winery muted small">{{ item.winery || 'винодельня не определена' }}</p>
          </div>
        </div>
        <div class="score-bar" :aria-label="`относительный скор ${Math.round(Number(item.score) * 100) / 100}`">
          <span :style="{ width: width(item.score) }" />
        </div>
        <button class="secondary" type="button" @click="emit('pick', item.slug)">Открыть</button>
      </li>
    </ul>

    <p v-else class="muted small">Кандидатов нет — попробуйте другое фото.</p>
  </section>
</template>

<style scoped>
.candidates { list-style: none; padding: 0; margin: var(--space-4) 0 0; display: grid; gap: var(--space-3); }
.candidate {
  display: grid; gap: var(--space-2);
  border: 1px solid var(--border); border-radius: var(--radius-sm);
  padding: var(--space-3); background: var(--surface-2);
}
.candidate-head { display: flex; align-items: center; gap: var(--space-3); }
.candidate-rank {
  width: 26px; height: 26px; flex: none; border-radius: 50%;
  display: grid; place-items: center;
  background: var(--accent-soft); color: var(--accent); font-size: .78rem; font-weight: 700;
}
.candidate-name { margin: 0; font-weight: 600; }
.candidate-winery { margin: 2px 0 0; }
.score-bar { height: 6px; border-radius: 999px; background: var(--border); overflow: hidden; }
.score-bar span { display: block; height: 100%; background: linear-gradient(90deg, var(--accent), var(--gold)); }
</style>
