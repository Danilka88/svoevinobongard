<script setup lang="ts">
// Обёртка изображения (C-050).
//
// Зачем: картинки портала по «сырым» адресам /uploads/... отдают 404, и <img> оставался пустым.
// Компонент грузит по переданному адресу (обычно уже прокси-адрес портала), не отправляет Referer,
// откладывает загрузку и при ошибке подставляет локальный запасной ассет или плейсхолдер — пустого
// места на экране не остаётся.
//
// Корневой элемент — сам <img>, поэтому существующие классы и размеры продолжают работать:
// атрибуты (class, style, width, height) прокидываются на изображение.
const props = withDefaults(defineProps<{
  src?: string | null;
  alt?: string;
  fallback?: string;
}>(), {
  src: '',
  alt: '',
  fallback: '/media/placeholder.svg',
});

const failed = ref(false);
const loaded = ref(false);

const current = computed(() => {
  if (!props.src || failed.value) return props.fallback;
  return props.src;
});
const state = computed(() => (failed.value ? 'fallback' : (loaded.value ? 'loaded' : 'loading')));

function onLoad() {
  loaded.value = true;
}

function onError() {
  // Ошибка загрузки не должна оставлять пустое место: переключаемся на запасной ассет.
  if (!failed.value) failed.value = true;
}
</script>

<template>
  <img
    :src="current"
    :alt="alt"
    :data-state="state"
    loading="lazy"
    decoding="async"
    referrerpolicy="no-referrer"
    @load="onLoad"
    @error="onError"
  >
</template>
