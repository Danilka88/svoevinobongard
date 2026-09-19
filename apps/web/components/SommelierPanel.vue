<script setup lang="ts">
import type { SommelierResponse } from '~/composables/useGateway';

const props = defineProps<{ slug: string }>();
const { sommelier } = useGateway();

const DISHES = ['рыба', 'морепродукты', 'мясо', 'птица', 'сыры', 'десерт'];
const BUDGETS = ['любой', 'низкий', 'средний', 'высокий'];
const STYLES = ['сухое', 'полусухое', 'полусладкое', 'сладкое', 'брют'];

const dish = ref(DISHES[0]);
const budget = ref(BUDGETS[0]);
const style = ref(STYLES[0]);
const busy = ref(false);
const error = ref('');
const result = ref<SommelierResponse | null>(null);

async function ask() {
  busy.value = true;
  error.value = '';
  try {
    result.value = await sommelier(props.slug, { dish: dish.value, budget: budget.value, style: style.value });
  } catch (err) {
    error.value = 'Не удалось получить совет сомелье. Попробуйте ещё раз.';
  } finally {
    busy.value = false;
  }
}
</script>

<template>
  <section class="card">
    <h2>Цифровой сомелье</h2>
    <p class="muted">Ответьте на пару вопросов — подберём, что попробовать, и с чем подать.</p>

    <label class="muted">Блюдо</label>
    <select v-model="dish" class="btn secondary" style="margin-bottom: 10px">
      <option v-for="option in DISHES" :key="option" :value="option">{{ option }}</option>
    </select>

    <label class="muted">Бюджет</label>
    <select v-model="budget" class="btn secondary" style="margin-bottom: 10px">
      <option v-for="option in BUDGETS" :key="option" :value="option">{{ option }}</option>
    </select>

    <label class="muted">Стиль</label>
    <select v-model="style" style="margin-bottom: 12px" class="btn secondary">
      <option v-for="option in STYLES" :key="option" :value="option">{{ option }}</option>
    </select>

    <button class="btn" :disabled="busy" @click="ask">{{ busy ? 'Подбираем…' : 'Подобрать вино' }}</button>
    <p v-if="error" class="status error">{{ error }}</p>

    <template v-if="result">
      <p v-if="result.pairing.length" class="status ok">Подача: <strong>{{ result.pairing.join(', ') }}</strong></p>
      <ul v-if="result.recommendations.length" class="list">
        <li v-for="item in result.recommendations" :key="item.slug" class="item">
          <div>
            <div class="name">{{ item.name || item.slug }}</div>
            <div class="why">{{ item.why }}</div>
          </div>
        </li>
      </ul>
      <p v-else class="muted">Подходящих вариантов не нашлось.</p>
      <p class="muted"><span class="badge">{{ result.source === 'llm' ? 'совет модели' : 'подбор по каталогу' }}</span></p>
    </template>
  </section>
</template>
