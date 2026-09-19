<script setup lang="ts">
import type { EnotourismArticle } from '~/composables/useGateway';

// «Истории в бокале»: статьи портала, подобранные по региону, категории и сорту вина.
// Данные приходят из gateway (/v1/enotourism/{slug}) — UI не ходит в API портала напрямую.
const props = defineProps<{ articles: EnotourismArticle[] }>();

const visible = computed(() => (props.articles || []).filter(a => a.title));

function when(iso: string) {
  if (!iso) return '';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleDateString('ru-RU', { day: 'numeric', month: 'long', year: 'numeric' });
}
</script>

<template>
  <section v-if="visible.length" class="card">
    <h2>Истории в бокале</h2>
    <p class="muted">Материалы портала «Своё Вино» о регионе и стиле этого вина.</p>

    <ul class="stories">
      <li v-for="article in visible" :key="article.slug" class="story">
        <a class="story-title" :href="article.url" target="_blank" rel="noopener">{{ article.title }}</a>
        <p class="story-meta">
          <template v-if="article.category">{{ article.category }}</template>
          <template v-if="article.author"> · {{ article.author }}</template>
          <template v-if="when(article.published_at)"> · {{ when(article.published_at) }}</template>
        </p>
      </li>
    </ul>
  </section>
</template>

<style scoped>
.stories { list-style: none; padding: 0; margin: 12px 0 0; display: grid; gap: 12px; }
.story { border-left: 2px solid rgba(255, 255, 255, 0.18); padding-left: 12px; }
.story-title { font-weight: 600; text-decoration: none; color: inherit; line-height: 1.35; display: block; }
.story-title:hover { text-decoration: underline; }
.story-meta { margin: 4px 0 0; font-size: 0.78rem; opacity: 0.65; }
</style>
