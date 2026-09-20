<script setup lang="ts">
// Загрузчик фото (C-036): съёмка или выбор файла, drag and drop на десктопе,
// превью выбранного кадра, состояния busy/error, доступность с клавиатуры.
const props = defineProps<{ busy?: boolean }>();
const emit = defineEmits<{ (event: 'selected', file: File): void }>();

const input = ref<HTMLInputElement | null>(null);
const preview = ref('');
const fileName = ref('');
const dragging = ref(false);

function accept(file?: File | null) {
  if (!file) return;
  if (!file.type.startsWith('image/')) return;
  preview.value = URL.createObjectURL(file);
  fileName.value = file.name;
  emit('selected', file);
}

function onDrop(event: DragEvent) {
  dragging.value = false;
  accept(event.dataTransfer?.files?.[0]);
}

function clear() {
  preview.value = '';
  fileName.value = '';
  if (input.value) input.value.value = '';
}

onBeforeUnmount(() => { if (preview.value) URL.revokeObjectURL(preview.value); });
</script>

<template>
  <section class="card" aria-labelledby="uploader-title">
    <h2 id="uploader-title" class="block-title">Фото этикетки</h2>

    <div
      class="dropzone"
      :class="{ dragging, busy: props.busy }"
      role="button"
      tabindex="0"
      :aria-busy="props.busy ? 'true' : 'false'"
      aria-describedby="uploader-hint"
      @click="input?.click()"
      @keydown.enter.prevent="input?.click()"
      @keydown.space.prevent="input?.click()"
      @dragover.prevent="dragging = true"
      @dragleave.prevent="dragging = false"
      @drop.prevent="onDrop"
    >
      <img v-if="preview" :src="preview" alt="Выбранное фото" class="dropzone-preview">
      <div v-else class="dropzone-placeholder">
        <svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true">
          <path d="M4 8V6a2 2 0 0 1 2-2h2M16 4h2a2 2 0 0 1 2 2v2M20 16v2a2 2 0 0 1-2 2h-2M8 20H6a2 2 0 0 1-2-2v-2M4 12h16" />
        </svg>
        <p class="muted" style="margin: 8px 0 0">Наведите камеру на этикетку целиком</p>
      </div>
    </div>

    <p id="uploader-hint" class="muted small" style="margin-top: 10px">
      Подойдёт обычный снимок с телефона: этикетку обрезать не нужно, кадр целиком помогает найти вино даже под углом.
      На компьютере можно перетащить файл в область выше.
    </p>

    <div class="row" style="margin-top: 12px">
      <button class="primary" type="button" :disabled="props.busy" @click="input?.click()">
        {{ props.busy ? 'Ищу вино…' : 'Снять или выбрать фото' }}
      </button>
      <button v-if="preview" class="ghost" type="button" :disabled="props.busy" @click.stop="clear">Сбросить</button>
      <span v-if="fileName" class="muted small">{{ fileName }}</span>
    </div>

    <input
      ref="input"
      class="sr-only"
      type="file"
      accept="image/*"
      capture="environment"
      aria-label="Выбрать или снять фото этикетки"
      @change="accept(($event.target as HTMLInputElement).files?.[0])"
    >
  </section>
</template>

<style scoped>
.dropzone {
  position: relative;
  display: grid;
  place-items: center;
  min-height: 210px;
  border: 2px dashed var(--border-strong);
  border-radius: var(--radius);
  background: var(--surface-2);
  color: var(--text-muted);
  cursor: pointer;
  overflow: hidden;
  transition: border-color var(--transition), background var(--transition), transform var(--transition);
}
.dropzone:hover { border-color: var(--accent); }
.dropzone.dragging { border-color: var(--accent); background: var(--accent-soft); transform: scale(1.01); }
.dropzone.busy { cursor: progress; opacity: .85; }
.dropzone-preview { width: 100%; max-height: 320px; object-fit: contain; background: var(--surface); }
.dropzone-placeholder { text-align: center; padding: var(--space-5); }
.sr-only {
  position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
  overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; border: 0;
}
</style>
