<script setup lang="ts">
const props = defineProps<{ busy?: boolean }>();
const emit = defineEmits<{ (event: 'selected', file: File): void }>();

const preview = ref<string>('');
const input = ref<HTMLInputElement | null>(null);

function onPick(event: Event) {
  const target = event.target as HTMLInputElement;
  const file = target.files?.[0];
  if (!file) return;
  preview.value = URL.createObjectURL(file);
  emit('selected', file);
}

function openCamera() {
  input.value?.click();
}

onBeforeUnmount(() => {
  if (preview.value) URL.revokeObjectURL(preview.value);
});
</script>

<template>
  <section class="card">
    <h2>Фото этикетки</h2>
    <div class="uploader">
      <input ref="input" type="file" accept="image/*" capture="environment" @change="onPick">
      <button class="btn" :disabled="props.busy" @click="openCamera">
        {{ props.busy ? 'Распознаём…' : 'Снять или выбрать фото' }}
      </button>
      <p class="hint">Наведите камеру на этикетку: сервис найдёт карточку вина в каталоге «Своё Вино».</p>
    </div>
    <img v-if="preview" class="preview" :src="preview" alt="Превью этикетки">
    <div v-if="props.busy" class="skeleton" />
  </section>
</template>
