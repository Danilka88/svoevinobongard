/**
 * Обезличенная телеметрия интерфейса (C-044).
 *
 * Приватность прежде всего: уходят только структурные события из белого списка — маршрут, метка
 * действия, число, признак успеха, задержка и короткий код. Никаких изображений, свободного текста
 * и персональных данных. Идентификатор сессии — случайный, хранится локально и с человеком не связан.
 *
 * Отправка best-effort: батчами по порогу или таймеру, при уходе со страницы — через sendBeacon.
 * Любой сбой телеметрии молча игнорируется и не влияет на работу интерфейса.
 */
export type TelemetryType = 'page_view' | 'action' | 'scan' | 'sommelier' | 'catalog'
  | 'theme' | 'api_error' | 'js_error';

export interface TelemetryPayload {
  route?: string;
  action?: string;
  value?: string | number;
  ok?: boolean;
  latency_ms?: number;
  detail?: string;
}

const BATCH_SIZE = 20;
const FLUSH_INTERVAL_MS = 15000;
const STORAGE_KEY = 'sv_session_id';

const queue: Array<Record<string, unknown>> = [];
let seq = 0;
let timer: ReturnType<typeof setInterval> | null = null;
let sessionId = '';

/** Читает идентификатор сессии из localStorage, создавая его при первом обращении. */
function ensureSessionId(): string {
  if (sessionId) return sessionId;
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored) {
      sessionId = stored;
      return sessionId;
    }
    const generated = (window.crypto?.randomUUID?.() || `s-${Date.now()}-${Math.random().toString(16).slice(2)}`);
    window.localStorage.setItem(STORAGE_KEY, generated);
    sessionId = generated;
  } catch {
    // Приватный режим или запрет хранилища: работаем с идентификатором в памяти, телеметрия не критична.
    sessionId = `mem-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }
  return sessionId;
}

/** Адрес приёма телеметрии: берётся из публичной конфигурации Nuxt. */
function endpoint(): string {
  const config = useRuntimeConfig();
  const base = String(config.public.apiBase || '').replace(/\/$/, '');
  return `${base}/v1/telemetry`;
}

/** Отправляет накопленный батч; при уходе со страницы используется sendBeacon. */
function flush(beacon = false) {
  if (!queue.length) return;
  const events = queue.splice(0, queue.length);
  const body = JSON.stringify({ session_id: ensureSessionId(), events });
  try {
    if (beacon && navigator.sendBeacon) {
      navigator.sendBeacon(endpoint(), new Blob([body], { type: 'application/json' }));
      return;
    }
    void fetch(endpoint(), {
      method: 'POST', body, keepalive: true,
      headers: { 'Content-Type': 'application/json', 'X-Session-Id': ensureSessionId() },
    }).catch(() => undefined);
  } catch {
    // Сбой телеметрии никогда не влияет на интерфейс.
  }
}

/** Планирует отправку по таймеру: один интервал на всё приложение. */
function schedule() {
  if (timer !== null || typeof window === 'undefined') return;
  timer = setInterval(() => flush(false), FLUSH_INTERVAL_MS);
}

export function useTelemetry() {
  /** Записывает событие: поля строго из белого списка, длинные строки не сохраняются. */
  function track(type: TelemetryType, payload: TelemetryPayload = {}) {
    if (import.meta.server) return;
    seq += 1;
    queue.push({
      type, seq,
      route: payload.route ?? window.location.pathname,
      action: payload.action,
      value: typeof payload.value === 'string' ? payload.value.slice(0, 64) : payload.value,
      ok: payload.ok,
      latency_ms: payload.latency_ms,
      detail: payload.detail?.slice(0, 64),
    });
    schedule();
    if (queue.length >= BATCH_SIZE) flush(false);
  }

  /** Замеряет длительность действия и фиксирует результат. */
  async function time<T>(type: TelemetryType, action: string, run: () => Promise<T>, extra: TelemetryPayload = {}) {
    const started = Date.now();
    try {
      const result = await run();
      track(type, { ...extra, action, ok: true, latency_ms: Date.now() - started });
      return result;
    } catch (error) {
      track(type, { ...extra, action, ok: false, latency_ms: Date.now() - started, detail: 'failed' });
      // Неуспешный вызов gateway фиксируем отдельным типом: по нему в сводке видно, что сломалось.
      track('api_error', { ...extra, action, ok: false, detail: 'request_failed' });
      throw error;
    }
  }

  return { track, time, flush, sessionId: () => ensureSessionId(), endpoint };
}
