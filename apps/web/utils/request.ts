/**
 * Таймауты запросов к gateway (C-049).
 *
 * Против мгновенного отказа (ECONNREFUSED) хватало обработки ошибки, но против «висящего»
 * соединения (сервис принял соединение и молчит) интерфейс оставался в скелетоне навсегда.
 * Утилита даёт сигнал прерывания по времени; ошибки прерывания классифицируются как сбой связи,
 * чтобы вызывающий код возвращал состояние недоступности, а не выбрасывал исключение.
 */

/** Сигнал прерывания по времени: основной путь — AbortSignal.timeout, запасной — контроллер. */
export function requestTimeout(ms: number): AbortSignal {
  const budget = Number.isFinite(ms) && ms > 0 ? ms : 8000;
  if (typeof AbortSignal !== 'undefined' && typeof AbortSignal.timeout === 'function') {
    return AbortSignal.timeout(budget);
  }
  const controller = new AbortController();
  setTimeout(() => controller.abort(new Error('timeout')), budget);
  return controller.signal;
}

/** Отличает прерывание по времени и сбой связи от прочих ошибок. */
export function isTimeout(error: unknown): boolean {
  const name = (error as { name?: string } | null)?.name;
  return name === 'TimeoutError' || name === 'AbortError';
}
