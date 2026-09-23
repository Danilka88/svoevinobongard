<script setup lang="ts">
import type { SimilarItem } from '~/composables/useGateway';

// Аналоги из других виноделен (C-036): изображения из медиа портала, причина совпадения
// и переход к карточке. Источник — официальный список похожих вин gateway.
const props = defineProps<{ items: SimilarItem[]; loading?: boolean }>();
const emit = defineEmits<{ (event: 'pick', slug: string): void }>();
</script>

<template>
  <section class="card" aria-labelledby="similar-title">
    <h2 id="similar-title">Похожие вина других виноделен</h2>
    <p class="muted small">Список приходит из официальных похожих вин портала.</p>

    <ul v-if="props.loading" class="similar-grid">
      <li v-for="n in 2" :key="`sk-${n}`" class="similar">
        <div class="skeleton similar-photo" />
        <div class="skeleton skeleton-line" style="width: 70%" />
        <div class="skeleton skeleton-line" style="width: 40%" />
      </li>
    </ul>

    <ul v-else-if="props.items?.length" class="similar-grid">
      <li v-for="item in props.items" :key="item.slug" class="similar">
        <AppImage v-if="item.image_url" :src="item.image_url" :alt="item.name" class="similar-photo" loading="lazy" />
        <div v-else class="similar-photo similar-photo-empty" aria-hidden="true" />
        <p class="similar-name">{{ item.name || item.slug }}</p>
        <p class="muted small" style="margin: 2px 0 0">{{ item.winery }}<template v-if="item.region"> · {{ item.region }}</template></p>
        <p v-if="item.reason" class="muted small" style="margin: 6px 0 0">{{ item.reason }}</p>
        <div class="row" style="margin-top: 10px">
          <button class="secondary" type="button" @click="emit('pick', item.slug)">Открыть</button>
          <a v-if="item.source === 'official'" class="ghost" :href="`https://vino-svoe.ru/wine/${item.slug}`" target="_blank" rel="noopener">На портале</a>
        </div>
      </li>
    </ul>

    <p v-else class="muted small">Пока нет данных об аналогах.</p>
  </section>
</template>

<style scoped>
.similar-grid {
  list-style: none; padding: 0; margin: var(--space-3) 0 0;
  display: grid; gap: var(--space-4); grid-template-columns: 1fr;
}
@media (min-width: 720px) { .similar-grid { grid-template-columns: 1fr 1fr; } }
.similar { border: 1px solid var(--border); border-radius: var(--radius-sm); padding: var(--space-3); background: var(--surface-2); }
.similar-photo { width: 100%; height: 128px; object-fit: contain; background: var(--surface); border-radius: var(--radius-sm); }
.similar-photo-empty { display: block; }
.similar-name { margin: var(--space-2) 0 0; font-weight: 600; }
</style>
