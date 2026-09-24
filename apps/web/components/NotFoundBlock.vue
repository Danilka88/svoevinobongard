<script setup lang="ts">
import type { TopKItem } from '~/composables/useGateway';

// Честный отказ (C-036 → C-043): экран «никогда не тупик».
// Показываем три разные ситуации (нет в каталоге / не уверены / сервис недоступен),
// распознанный с этикетки текст, ближайших кандидатов и путь к ручному поиску на портале.
const props = defineProps<{
  candidates: TopKItem[];
  alternatives?: TopKItem[];
  recognizedFields?: Record<string, string | number | null> | null;
  reason?: string;
  degraded?: boolean;
  notFound?: boolean;
  ambiguous?: boolean;
}>();
const emit = defineEmits<{ (event: 'pick', slug: string): void }>();

const PORTAL_SEARCH = 'https://vino-svoe.ru/wines';

// C-067: при неуверенности показываем до десяти кандидатов (расширенная выдача top10).
const top = computed(() => (props.candidates || []).slice(0, 10));
const alternatives = computed(() => (props.alternatives || []).slice(0, 6));
const maxScore = computed(() => Math.max(...top.value.map(item => Number(item.score) || 0), 0.0001));

// Состояние отказа: сначала недоступность сервиса, затем «нет в каталоге», иначе неуверенность.
const state = computed<'degraded' | 'not_found' | 'uncertain'>(() => {
  const reason = String(props.reason || '');
  // «Сервис недоступен» — только при явной причине сбоя сервиса, а не при любом degraded:
  // backend ставит degraded и для «нет совпадения в каталоге», где падать нечему.
  if (!props.notFound && ['normalize_unavailable', 'ocr_unavailable'].includes(reason)) return 'degraded';
  if (props.notFound || ['no_catalog_match', 'no_candidates'].includes(reason)) return 'not_found';
  return 'uncertain';
});

const REASONS: Record<string, string> = {
  no_catalog_match: 'Этикетка прочитана, но среди российских вин портала совпадения нет.',
  no_candidates: 'Этикетка прочитана, но похожих вин в каталоге не нашлось.',
  low_confidence: 'Вина с похожей этикеткой есть, но разрыв между вариантами слишком мал.',
  ocr_unavailable: 'Распознавание текста было недоступно, поэтому часть подсказок потерялась.',
  normalize_unavailable: 'Сервис подготовки изображения недоступен — попробуйте ещё раз.',
  internal_error: 'Внутренняя ошибка: ответ неполный, но сервис продолжает работать.',
};

const headline = computed(() => {
  if (state.value === 'degraded') return 'Сервис сейчас недоступен';
  if (state.value === 'not_found') return 'Такого вина нет на «Своё Вино»';
  if (props.ambiguous) return 'Похоже на несколько вин из одной серии';
  return 'Не уверены — возможно, дело в фото';
});

const lead = computed(() => {
  if (state.value === 'degraded') {
    return 'Мы не смогли обработать снимок полностью. Попробуйте повторить через минуту — остальные данные портала доступны.';
  }
  if (state.value === 'not_found') {
    return 'Скорее всего, вино не представлено на портале «Своё Вино»: например, оно не российское или выпущено малым тиражом. Ниже — что удалось прочитать и похожие вина из каталога.';
  }
  if (props.ambiguous) {
    return 'Этикетки внутри одной серии почти не отличаются на фото. Ниже — все близкие варианты: выберите своё вино или уточните год и сладость.';
  }
  return 'Этикетка прочитана, но уверенности для точного ответа не хватило: снимок под неудобным углом, блики или несколько бутылок. Если среди вариантов ниже есть ваше вино — выберите его.';
});

const reasonText = computed(() => REASONS[String(props.reason || '')] || '');

// Распознанные поля показываем только заполненными: пустые строки не создают «шум» в интерфейсе.
const readFields = computed(() => {
  const labels: Record<string, string> = {
    winery: 'Винодельня', wine_name: 'Название', vintage: 'Год',
    grape: 'Сорт', sweetness: 'Сладость', region: 'Регион',
  };
  return Object.entries(props.recognizedFields || {})
    .filter(([, value]) => value !== null && value !== undefined && String(value).trim() !== '')
    .map(([key, value]) => ({ label: labels[key] || key, value: String(value) }));
});

const query = computed(() => {
  const fields = readFields.value;
  const name = fields.find(item => item.label === 'Название')?.value || '';
  const winery = fields.find(item => item.label === 'Винодельня')?.value || '';
  return name || winery;
});

const portalUrl = computed(() =>
  query.value ? `${PORTAL_SEARCH}?search=${encodeURIComponent(query.value)}` : PORTAL_SEARCH);

const TIPS = [
  'Снимайте этикетку целиком, не обрезая края',
  'Держите бутылку против света, чтобы не было бликов',
  'Заполните кадр этикеткой — фон не нужен',
  'Одна бутылка в кадре: несколько этикеток сбивают поиск',
  'Протрите объектив и подождите фокус перед съёмкой',
];

function width(score: number) {
  return `${Math.max(6, Math.round((Number(score) / maxScore.value) * 100))}%`;
}
</script>

<template>
  <section class="card" aria-labelledby="nf-title">
    <p class="state-badge" :class="`state-${state}`">
      {{ state === 'degraded' ? 'Сервис недоступен' : (state === 'not_found' ? 'Нет в каталоге' : 'Низкая уверенность') }}
    </p>
    <h2 id="nf-title">{{ headline }}</h2>
    <p class="muted">{{ lead }}</p>
    <p v-if="reasonText" class="muted small">{{ reasonText }}</p>

    <section v-if="readFields.length" class="recognized" aria-labelledby="nf-read">
      <h3 id="nf-read" class="recognized-title">Мы прочитали с этикетки</h3>
      <ul class="recognized-list">
        <li v-for="field in readFields" :key="field.label">
          <span class="recognized-label">{{ field.label }}</span>
          <span class="recognized-value">{{ field.value }}</span>
        </li>
      </ul>
    </section>

    <section aria-labelledby="nf-next">
      <h3 id="nf-next" class="recognized-title">Что попробовать дальше</h3>
      <ul class="candidates" v-if="top.length">
        <li v-for="(item, index) in top" :key="item.slug" class="candidate">
          <div class="candidate-head">
            <span class="candidate-rank">{{ index + 1 }}</span>
            <div style="min-width: 0">
              <p class="candidate-name">{{ item.name || item.slug }}</p>
              <p class="candidate-winery muted small">{{ item.winery || 'винодельня не определена' }}</p>
            </div>
          </div>
          <div class="score-bar" :aria-label="`относительный скор ${Math.round(Number(item.score) * 100) / 100}`">
            <span :style="{ width: width(item.score) }" />
          </div>
          <button class="secondary" type="button" @click="emit('pick', item.slug)">Открыть карточку</button>
        </li>
      </ul>
      <p v-else class="muted small">Кандидатов нет — это нормально, если вина нет в каталоге.</p>
    </section>

    <section v-if="state === 'not_found' && alternatives.length" class="alt" aria-labelledby="nf-alt">
      <h3 id="nf-alt" class="recognized-title">Похожие вина других виноделен</h3>
      <p class="muted small">Этого вина нет на портале — вот близкие аналоги из каталога.</p>
      <ul class="candidates alt-grid">
        <li v-for="item in alternatives" :key="item.slug" class="candidate">
          <div class="candidate-head">
            <div style="min-width: 0">
              <p class="candidate-name">{{ item.name || item.slug }}</p>
              <p class="candidate-winery muted small">{{ item.winery || 'винодельня не определена' }}</p>
            </div>
          </div>
          <button class="secondary" type="button" @click="emit('pick', item.slug)">Открыть карточку</button>
        </li>
      </ul>
    </section>

    <p class="actions">
      <a class="primary" :href="portalUrl" target="_blank" rel="noopener noreferrer">Найти вручную на «Своё Вино»</a>
    </p>

    <section class="tips" aria-labelledby="nf-tips">
      <h3 id="nf-tips" class="recognized-title">Как переснять, чтобы получилось</h3>
      <ul class="chips">
        <li v-for="tip in TIPS" :key="tip" class="chip">{{ tip }}</li>
      </ul>
    </section>
  </section>
</template>

<style scoped>
.state-badge {
  display: inline-block; margin: 0 0 var(--space-2);
  padding: 2px 10px; border-radius: 999px;
  font-size: .74rem; font-weight: 700; letter-spacing: .02em; text-transform: uppercase;
}
.state-not_found { background: var(--accent-soft); color: var(--accent); }
.state-uncertain { background: var(--surface-2); color: var(--muted); }
.state-degraded { background: #7f1d1d22; color: #b91c1c; }
.recognized { margin-top: var(--space-4); }
.recognized-title { margin: 0 0 var(--space-2); font-size: .95rem; }
.recognized-list {
  list-style: none; padding: var(--space-3); margin: 0;
  display: grid; gap: 6px; border: 1px solid var(--border);
  border-radius: var(--radius-sm); background: var(--surface-2);
}
.recognized-list li { display: flex; gap: var(--space-3); justify-content: space-between; }
.recognized-label { color: var(--muted); font-size: .86rem; }
.recognized-value { font-weight: 600; text-align: right; }
.candidates { list-style: none; padding: 0; margin: var(--space-4) 0 0; display: grid; gap: var(--space-3); }
.candidate {
  display: grid; gap: var(--space-2);
  border: 1px solid var(--border); border-radius: var(--radius-sm);
  padding: var(--space-3); background: var(--surface-2);
}
.candidate-head { display: flex; align-items: center; gap: var(--space-3); }
.candidate-rank {
  width: 26px; height: 26px; flex: none; border-radius: 50%;
  display: grid; place-items: center;
  background: var(--accent-soft); color: var(--accent); font-size: .78rem; font-weight: 700;
}
.candidate-name { margin: 0; font-weight: 600; }
.candidate-winery { margin: 2px 0 0; }
.score-bar { height: 6px; border-radius: 999px; background: var(--border); overflow: hidden; }
.score-bar span { display: block; height: 100%; background: linear-gradient(90deg, var(--accent), var(--gold)); }
/* C-067: аналоги из других виноделен — две колонки на широком экране, одна на мобиле. */
.alt { margin-top: var(--space-4); }
.alt-grid { grid-template-columns: 1fr; }
@media (min-width: 720px) { .alt-grid { grid-template-columns: 1fr 1fr; } }
.actions { margin: var(--space-4) 0 0; }
.actions .primary { display: inline-block; text-decoration: none; }
.tips { margin-top: var(--space-4); }
</style>
