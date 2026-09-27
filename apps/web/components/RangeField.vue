<script setup lang="ts">
// C-091: единый ползунок с градациями словами и кросс-браузерными стилями.
// Проблема прежних слайдеров: стили только под WebKit → в Firefox трек/бегунок не видны,
// а значения 0..3 не объяснены. Здесь подписи min/max, текущее значение словами и aria-valuetext.
const props = defineProps<{
  modelValue: string | number;
  label: string;
  min?: number;
  max?: number;
  step?: number;
  // Градации словами по возрастанию значения (индекс = значение при step=1) либо через функцию.
  grades?: string[];
  minLabel?: string;
  maxLabel?: string;
  hint?: string;
}>();

const emit = defineEmits<{ (event: 'update:modelValue', value: string): void }>();

const min = computed(() => props.min ?? 0);
const max = computed(() => props.max ?? 3);
const step = computed(() => props.step ?? 1);

const numeric = computed(() => {
  const value = Number(props.modelValue);
  return Number.isFinite(value) ? value : min.value;
});

// Значение словами: берём градацию, если задана; иначе округляем число.
const gradeText = computed(() => {
  const value = numeric.value;
  if (props.grades?.length) {
    const index = Math.max(0, Math.min(props.grades.length - 1, Math.round(value - min.value)));
    return props.grades[index];
  }
  return String(value);
});

const percent = computed(() => {
  const span = max.value - min.value;
  return span > 0 ? Math.round(((numeric.value - min.value) / span) * 100) : 0;
});

function onInput(event: Event) {
  emit('update:modelValue', (event.target as HTMLInputElement).value);
}
</script>

<template>
  <div class="slider">
    <label :for="`rf-${label}`" class="field-label">
      {{ label }}: <span class="slider-value">{{ gradeText }}</span>
    </label>
    <input
      :id="`rf-${label}`"
      type="range"
      :min="min"
      :max="max"
      :step="step"
      :value="numeric"
      :aria-valuetext="gradeText"
      :style="{ '--rf-fill': `${percent}%` }"
      @input="onInput"
    >
    <div class="slider-scale">
      <span>{{ minLabel ?? min }}</span>
      <span v-if="hint" class="muted small">{{ hint }}</span>
      <span>{{ maxLabel ?? max }}</span>
    </div>
  </div>
</template>
