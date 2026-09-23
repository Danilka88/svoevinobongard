<script setup lang="ts">
// Явное состояние «Сервис недоступен» (C-048, REG-001).
//
// Отделяет сбой связи от пустого результата: раньше страницы показывали «нет данных» или
// «ничего не найдено», и сбой маскировался под пустоту. Здесь пользователь видит, что дело
// в сервисе, и может повторить попытку.
defineProps<{ detail?: string; action?: string }>();
const emit = defineEmits<{ (event: 'retry'): void }>();
</script>

<template>
  <section class="card unavailable" role="alert" aria-labelledby="unavailable-title">
    <p class="unavailable-badge">Сервис недоступен</p>
    <h2 id="unavailable-title">Не удалось получить данные</h2>
    <p class="muted">
      Gateway не отвечает: возможно, он не запущен или потерял связь. Это не пустой результат —
      данные просто не загрузились.
    </p>
    <p v-if="detail" class="muted small">{{ detail }}</p>
    <p class="unavailable-actions">
      <button class="primary" type="button" @click="emit('retry')">{{ action || 'Повторить' }}</button>
    </p>
  </section>
</template>

<style scoped>
.unavailable { border-color: #b91c1c55; }
.unavailable-badge {
  display: inline-block; margin: 0 0 var(--space-2);
  padding: 2px 10px; border-radius: 999px;
  background: #7f1d1d22; color: #b91c1c;
  font-size: .74rem; font-weight: 700; letter-spacing: .02em; text-transform: uppercase;
}
.unavailable-actions { margin: var(--space-3) 0 0; }
</style>
