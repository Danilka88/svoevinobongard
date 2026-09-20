<script setup lang="ts">
import type { EnotourismArticle } from '~/composables/useGateway';

// «Истории в бокале» (C-036): статьи портала с изображениями, автором и датой.
// Данные приходят из gateway (/v1/enotourism/{slug}) — UI не ходит в API портала напрямую.
const props = defineProps<{ articles: EnotourismArticle[] }>();

const visible = computed(() => (props.articles || []).filter(article => article.title));

function when(iso: string) {
  if (!iso) return '';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleDateString('ru-RU', { day: 'numeric', month: 'long', year: 'numeric' });
}
</script>

<template>
  <div v-if="visible.length" class="block">
    <h3 class="block-title">Истории в бокале</h3>
    <p class="muted small">Материалы портала «Своё Вино» о регионе и стиле этого вина.</p>

    <ul class="story-grid">
      <li v-for="article in visible" :key="article.slug" class="story">
        <img v-if="article.image_url" :src="article.image_url" alt="" class="story-photo" loading="lazy">
        <div>
          <a class="story-title" :href="article.url" target="_blank" rel="noopener">{{ article.title }}</a>
          <p class="story-meta">
            <template v-if="article.category">{{ article.category }}</template>
            <template v-if="article.author"> · {{ article.author }}</template>
            <template v-if="when(article.published_at)"> · {{ when(article.published_at) }}</template>
          </p>
        </div>
      </li>
    </ul>
  </div>
</template>

<style scoped>
.story-grid { list-style: none; padding: 0; margin: var(--space-3) 0 0; display: grid; gap: var(--space-3); }
.story { display: flex; gap: var(--space-3); border-left: 2px solid var(--border); padding-left: var(--space-3); }
.story-photo { width: 76px; height: 56px; object-fit: cover; border-radius: var(--radius-sm); flex: none; }
.story-title { font-weight: 600; line-height: 1.35; display: block; }
.story-meta { margin: 4px 0 0; font-size: .78rem; color: var(--text-muted); }
</style>
