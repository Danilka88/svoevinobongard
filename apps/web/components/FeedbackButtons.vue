<script setup lang="ts">
// Обратная связь по результату скана (C-054).
//
// Голос пользователя никогда не меняет данные сам по себе: он попадает в очередь модератора и
// становится «доверенным вкладом» только при согласии нескольких независимых сессий и консенсуса
// агентов. Поэтому интерфейс честно говорит, что именно произошло с голосом.
const props = defineProps<{ slug?: string; imageSha256?: string }>();
const { feedback } = useGateway();
const telemetry = useTelemetry();

const CHIPS = [
  { code: 'wrong_wine', label: 'Не то вино' },
  { code: 'wrong_year', label: 'Неверный год' },
  { code: 'wrong_grape', label: 'Не тот сорт' },
  { code: 'bad_photo', label: 'Плохое фото' },
  { code: 'other', label: 'Другое' },
];

const picked = ref<string | null>(null);
const reasons = ref<string[]>([]);
const busy = ref(false);
const done = ref('');
const error = ref('');

function toggleReason(code: string) {
  reasons.value = reasons.value.includes(code)
    ? reasons.value.filter(item => item !== code)
    : [...reasons.value, code];
}

async function send(vote: 'up' | 'down') {
  // Идемпотентность: в одной сессии голос по этому вину отправляется один раз.
  if (busy.value || done.value) return;
  busy.value = true;
  error.value = '';
  try {
    const result = await feedback({
      vote,
      reason: vote === 'down' ? reasons.value : [],
      slug: props.slug || '',
      image_sha256: props.imageSha256 || '',
    });
    picked.value = vote;
    done.value = result?.trusted
      ? 'Спасибо! Голос учтён как доверенный вклад.'
      : 'Спасибо! Голос отправлен модератору — так мы не даём одному голосу влиять на данные.';
    telemetry.track('feedback', { action: vote === 'up' ? 'feedback_up' : 'feedback_down',
                                  ok: true, value: result?.trusted ? 'trusted' : 'queued' });
  } catch {
    error.value = 'Не удалось отправить голос. Попробуйте ещё раз.';
  } finally {
    busy.value = false;
  }
}
</script>

<template>
  <section class="feedback" aria-labelledby="feedback-title">
    <p id="feedback-title" class="feedback-title muted small">Это ваше вино?</p>
    <div class="feedback-actions" role="group" aria-label="Оценка результата распознавания">
      <button class="feedback-button" type="button" :disabled="busy || !!done"
              :aria-pressed="picked === 'up'" aria-label="Да, распознано верно" @click="send('up')">
        👍 <span>Верно</span>
      </button>
      <button class="feedback-button" type="button" :disabled="busy || !!done"
              :aria-pressed="picked === 'down'" aria-label="Нет, распознано неверно" @click="send('down')">
        👎 <span>Не то</span>
      </button>
    </div>

    <fieldset v-if="picked === 'down' && !done" class="feedback-chips">
      <legend class="muted small">Что именно не так?</legend>
      <label v-for="chip in CHIPS" :key="chip.code" class="chip-accent" :class="{ picked: reasons.includes(chip.code) }">
        <input type="checkbox" :checked="reasons.includes(chip.code)" @change="toggleReason(chip.code)">
        {{ chip.label }}
      </label>
    </fieldset>

    <p v-if="done" class="feedback-done small" role="status" aria-live="polite">{{ done }}</p>
    <p v-if="error" class="feedback-error small" role="alert">{{ error }}</p>
  </section>
</template>

<style scoped>
.feedback { margin-top: var(--space-4); border-top: 1px solid var(--border); padding-top: var(--space-3); }
.feedback-title { margin: 0 0 var(--space-2); }
.feedback-actions { display: flex; gap: var(--space-2); flex-wrap: wrap; }
.feedback-button {
  display: inline-flex; align-items: center; gap: 6px;
  min-height: var(--tap); padding: 0 var(--space-3);
  border: 1px solid var(--border-strong); border-radius: var(--radius-sm);
  background: var(--surface); color: var(--text); font: inherit; cursor: pointer;
}
.feedback-button:hover:not(:disabled) { border-color: var(--accent); }
.feedback-button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.feedback-button[aria-pressed="true"] { background: var(--accent-soft); color: var(--accent-on-soft); border-color: var(--accent); }
.feedback-button:disabled { opacity: .7; cursor: default; }
.feedback-chips { border: 0; padding: 0; margin: var(--space-3) 0 0; display: flex; flex-wrap: wrap; gap: 6px; }
.feedback-chips legend { padding: 0 0 var(--space-2); }
.feedback-chips .chip-accent { cursor: pointer; }
.feedback-chips .chip-accent.picked { background: var(--accent); color: var(--text-inverse); border-color: var(--accent); }
.feedback-chips input { position: absolute; opacity: 0; width: 1px; height: 1px; }
.feedback-done { color: var(--success-on-soft); margin: var(--space-2) 0 0; }
.feedback-error { color: var(--error-on-soft); margin: var(--space-2) 0 0; }
</style>
