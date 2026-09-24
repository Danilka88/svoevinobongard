<script setup lang="ts">
import type { SommelierItem, SommelierResponse } from '~/composables/useGateway';

// Экран сомелье (C-037): пошаговый подбор. Шаг 1 — что за столом и интенсивности,
// шаг 2 — повод и настроение, шаг 3 — предпочтения. Результаты приходят из gateway
// (/v1/sommelier), персонализация — из localStorage через useTaste.
useHead({ title: 'Сомелье — Своё Вино' });

const { sommelier } = useGateway();
const taste = useTaste();
const route = useRoute();

const baseSlug = computed(() => String(route.query.slug || ''));
const step = ref(0);
const busy = ref(false);
const error = ref('');
// C-048/REG-001: «сервис недоступен» — отдельное состояние, а не «подходящего не нашлось».
const unavailable = ref(false);
const result = ref<SommelierResponse | null>(null);

const dishes = [
  { id: 'мясо', label: 'Мясо и стейки' },
  { id: 'птица', label: 'Птица' },
  { id: 'рыба', label: 'Рыба' },
  { id: 'морепродукты', label: 'Морепродукты' },
  { id: 'сыры', label: 'Сыры' },
  { id: 'азиатское', label: 'Азиатское' },
  { id: 'bbq', label: 'BBQ и гриль' },
  { id: 'десерт', label: 'Десерт' },
];
const occasions = [
  { id: 'ужин', label: 'Будничный ужин' },
  { id: 'свидание', label: 'Свидание' },
  { id: 'вечеринка', label: 'Вечеринка' },
  { id: 'подарок', label: 'Подарок' },
];
const styles = ['', 'сухое', 'полусухое', 'полусладкое', 'сладкое', 'игристое'];
const budgets = [
  { id: '', label: 'Не важно' },
  { id: 'низкий', label: 'До 800 ₽' },
  { id: 'средний', label: '800–2000 ₽' },
  { id: 'высокий', label: 'Выше 2000 ₽' },
];

const answers = reactive<Record<string, string>>({
  dish: 'мясо', spice: '1', fat: '1', acidity: '1', sweetness: '0',
  occasion: 'ужин', budget: '', style: '', region: '', adventurousness: '0.5',
});
const regionInput = ref('');

const steps = ['Что за столом', 'Повод и настроение', 'Предпочтения'];
const canSubmit = computed(() => Boolean(answers.dish));

async function submit() {
  busy.value = true;
  error.value = '';
  unavailable.value = false;
  try {
    const payload: Record<string, string> = { ...answers };
    if (regionInput.value.trim()) payload.region = regionInput.value.trim();
    result.value = await sommelier(baseSlug.value, payload);
    // C-071: недоступность — только когда gateway реально не ответил. Валидный
    // детерминированный подбор (source official|deterministic) показывается как результат.
    if (result.value?.source === 'unavailable') {
      unavailable.value = true;
      error.value = 'Сервис недоступен: gateway не ответил, подбор не получен.';
    }
  } catch (e) {
    unavailable.value = true;
    error.value = 'Сервис недоступен: gateway не ответил, подбор не получен.';
    result.value = null;
  } finally {
    busy.value = false;
  }
}

function toTasteItem(item: SommelierItem) {
  return { slug: item.slug, name: item.name, winery: item.winery,
           grape: item.grape, region: item.region, category: item.category };
}

// при входе со сканера подбор запускается сразу от базового вина
onMounted(() => { taste.load(); if (baseSlug.value) void submit(); });
</script>

<template>
  <div class="page">
    <div class="hero" style="margin-bottom: 16px">
      <h1>Сомелье</h1>
      <p>
        <template v-if="baseSlug">Подбираем к вашему вину ({{ baseSlug }}) — уточните стол и повод.</template>
        <template v-else>Подбор с нуля: расскажите про стол, повод и предпочтения.</template>
      </p>
    </div>

    <ServiceUnavailable v-if="unavailable" :detail="error" action="Попробовать снова" @retry="submit" />

    <div class="toast-stack" role="status" aria-live="polite" v-if="error">
      <div class="toast">{{ error }}</div>
    </div>

    <section class="card">
      <ol class="steps" aria-label="Шаги подбора">
        <li v-for="(label, index) in steps" :key="label" :class="{ active: step === index, done: step > index }">
          <span class="step-num">{{ index + 1 }}</span>{{ label }}
        </li>
      </ol>

      <!-- Шаг 1 -->
      <div v-if="step === 0" class="stack">
        <h2>Что за столом</h2>
        <ul class="chips">
          <li v-for="dish in dishes" :key="dish.id">
            <button class="chip" type="button" :class="{ 'chip-accent': answers.dish === dish.id }" :aria-pressed="answers.dish === dish.id" @click="answers.dish = dish.id">
              {{ dish.label }}
            </button>
          </li>
        </ul>
        <div class="slider" v-for="slider in ([
          { key: 'spice', label: 'Острота' }, { key: 'fat', label: 'Жирность' },
          { key: 'acidity', label: 'Кислотность' }, { key: 'sweetness', label: 'Сладость соуса' }])" :key="slider.key">
          <label :for="`sl-${slider.key}`" class="field-label">{{ slider.label }}: <span class="slider-value">{{ answers[slider.key] }}</span></label>
          <input :id="`sl-${slider.key}`" v-model="answers[slider.key]" type="range" min="0" max="3" step="1">
        </div>
      </div>

      <!-- Шаг 2 -->
      <div v-else-if="step === 1" class="stack">
        <h2>Повод и настроение</h2>
        <ul class="chips">
          <li v-for="occasion in occasions" :key="occasion.id">
            <button class="chip" type="button" :class="{ 'chip-accent': answers.occasion === occasion.id }" :aria-pressed="answers.occasion === occasion.id" @click="answers.occasion = occasion.id">
              {{ occasion.label }}
            </button>
          </li>
        </ul>
        <div class="slider">
          <label for="adv" class="field-label">Смелость: <span class="slider-value">{{ answers.adventurousness }}</span></label>
          <input id="adv" v-model="answers.adventurousness" type="range" min="0" max="1" step="0.1">
          <p class="muted small" style="margin: 0">Слева — проверенное и знакомое, справа — необычные сорта и регионы.</p>
        </div>
      </div>

      <!-- Шаг 3 -->
      <div v-else class="stack">
        <h2>Предпочтения</h2>
        <div class="field">
          <label class="field-label" for="budget">Бюджет</label>
          <select id="budget" v-model="answers.budget" class="select">
            <option v-for="budget in budgets" :key="budget.id" :value="budget.id">{{ budget.label }}</option>
          </select>
        </div>
        <div class="field">
          <label class="field-label" for="style">Стиль</label>
          <select id="style" v-model="answers.style" class="select">
            <option v-for="style in styles" :key="style" :value="style">{{ style || 'Любой' }}</option>
          </select>
        </div>
        <div class="field">
          <label class="field-label" for="region">Регион</label>
          <input id="region" v-model="regionInput" class="input" type="text" placeholder="Любой (например, Крым)">
        </div>
        <p v-if="taste.items.value.length" class="muted small">
          Учтём «Мой вкус»: {{ taste.profile.value.grapes.slice(0, 4).join(', ') || 'предпочтения сохранены' }}
        </p>
      </div>

      <div class="row" style="margin-top: 16px">
        <button class="secondary" type="button" :disabled="step === 0 || busy" @click="step--">Назад</button>
        <button v-if="step < 2" class="primary" type="button" @click="step++">Дальше</button>
        <button v-else class="primary" type="button" :disabled="busy || !canSubmit" @click="submit">
          {{ busy ? 'Подбираю…' : 'Подобрать вино' }}
        </button>
      </div>
    </section>

    <!-- Результаты -->
    <section v-if="busy" class="card" aria-busy="true">
      <div class="skeleton skeleton-line" style="width: 40%" />
      <div class="skeleton" style="height: 110px" />
    </section>

    <template v-if="result && !busy">
      <section class="card" aria-live="polite">
        <h2>Рекомендации</h2>
        <p class="muted small">
          Источник: {{ result.source }}<template v-if="taste.likes.value.length"> · учтено вин из «Мой вкус»: {{ taste.likes.value.length }}</template>
        </p>
        <p v-if="result.llm_degraded" class="muted small" style="margin: 0">
          Резервный подбор по каталогу: текстовая модель не успела или вернула пустой ответ — пары и сорта взяты из каталога.
        </p>
        <ul class="rec-grid">
          <li v-for="item in result.recommendations" :key="item.slug" class="rec">
            <AppImage v-if="item.image_url" :src="item.image_url" :alt="item.name" class="rec-photo" loading="lazy" />
            <div class="rec-body">
              <h3 class="rec-title">{{ item.name || item.slug }}</h3>
              <p class="muted small" style="margin: 2px 0 6px">{{ item.winery }}</p>
              <p v-if="item.sensory_note" class="muted small">{{ item.sensory_note }}</p>
              <ul class="chips">
                <li v-if="item.grape" class="chip">{{ item.grape }}</li>
                <li v-if="item.region" class="chip">{{ item.region }}</li>
                <li v-if="item.category" class="chip">{{ item.category }}</li>
                <li v-if="item.year" class="chip">{{ item.year }}</li>
              </ul>
              <ul v-if="item.pairing_cards?.length" class="chips" style="margin-top: 8px">
                <li v-for="dish in item.pairing_cards" :key="dish.name" class="chip chip-dish">
                  <AppImage v-if="dish.url" :src="dish.url" alt="" class="dish-thumb" />{{ dish.name }}
                </li>
              </ul>
              <div class="row" style="margin-top: 10px">
                <button class="secondary" type="button" @click="taste.toggle(toTasteItem(item))"
                        :aria-pressed="taste.has(item.slug)">
                  {{ taste.has(item.slug) ? '♥ В моём вкусе' : '♡ Сохранить вкус' }}
                </button>
              </div>
            </div>
          </li>
        </ul>
      </section>

      <section v-if="result.alternatives?.length" class="card">
        <h2>Альтернативы</h2>
        <p class="muted small">Другие варианты под те же условия: другой регион или цена.</p>
        <ul class="rec-grid">
          <li v-for="alt in result.alternatives" :key="alt.slug" class="rec rec-compact">
            <AppImage v-if="alt.image_url" :src="alt.image_url" :alt="alt.name" class="rec-photo" loading="lazy" />
            <div class="rec-body">
              <p class="rec-title">{{ alt.name || alt.slug }}</p>
              <p class="muted small" style="margin: 2px 0 0">{{ alt.winery }}</p>
            </div>
          </li>
        </ul>
      </section>
    </template>

    <!-- Мой вкус -->
    <section class="card">
      <div class="row">
        <h2 style="margin: 0">Мой вкус</h2>
        <button v-if="taste.items.value.length" class="ghost" type="button" @click="taste.clear()">Очистить</button>
      </div>
      <p class="muted small">Хранится только в вашем браузере и учитывается при подборе.</p>
      <ul v-if="taste.items.value.length" class="chips">
        <li v-for="item in taste.items.value" :key="item.slug" class="chip">
          {{ item.name || item.slug }}
          <button class="chip-remove" type="button" :aria-label="`Убрать ${item.name || item.slug}`" @click="taste.remove(item.slug)">×</button>
        </li>
      </ul>
      <p v-else class="muted small">Пока пусто: отметьте ♥ вина, которые понравились.</p>
    </section>
  </div>
</template>

<style scoped>
.steps { list-style: none; display: flex; gap: var(--space-3); padding: 0; margin: 0 0 var(--space-4); flex-wrap: wrap; }
.steps li { display: flex; align-items: center; gap: 8px; color: var(--text-muted); font-size: .86rem; }
.steps li.active { color: var(--accent); font-weight: 600; }
.steps li.done { color: var(--success); }
.step-num { width: 22px; height: 22px; border-radius: 50%; display: grid; place-items: center; background: var(--surface-2); border: 1px solid var(--border); font-size: .75rem; }
.steps li.active .step-num { background: var(--accent-soft); border-color: var(--accent); color: var(--accent); }
.rec-grid { list-style: none; padding: 0; margin: var(--space-3) 0 0; display: grid; gap: var(--space-3); }
.rec { display: flex; gap: var(--space-3); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: var(--space-3); background: var(--surface-2); }
.rec-compact { align-items: center; }
.rec-photo { width: 86px; height: 116px; object-fit: contain; background: var(--surface); border-radius: var(--radius-sm); flex: none; }
.rec-body { min-width: 0; }
.rec-title { margin: 0; font-family: var(--font-serif); font-size: 1.05rem; }
.dish-thumb { width: 18px; height: 18px; border-radius: 50%; object-fit: cover; }
.chip-remove { border: 0; background: transparent; color: inherit; font-size: 1rem; cursor: pointer; min-width: 22px; }
.chips li button.chip { cursor: pointer; border: 1px solid var(--border); background: var(--surface-2); }
@media (max-width: 520px) { .rec { flex-direction: column; } .rec-photo { width: 100%; height: 140px; } }
</style>
