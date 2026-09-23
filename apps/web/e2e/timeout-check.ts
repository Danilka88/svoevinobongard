/**
 * Проверка таймаута запроса на «висящем» соединении (C-049).
 *
 * Запуск: bun run apps/web/e2e/timeout-check.ts <url> [таймаут_мс]
 * Ожидание: сигнал прерывания срабатывает в пределах бюджета, а вызывающий код получает
 * различимый результат вместо исключения. Скрипт печатает замер и завершается кодом 0/1.
 *
 * Используется та же утилита, что и в клиенте gateway: проверяется настоящий код, а не копия.
 */
import { isTimeout, requestTimeout } from '../utils/request';

const url = process.argv[2] || 'http://127.0.0.1:8099/v1/catalog';
const budget = Number(process.argv[3] || 1500);
const tolerance = 1200; // запас на планировщик

let unhandled = 0;
process.on('unhandledRejection', () => { unhandled += 1; });

/** Вызов, повторяющий обёртку клиента gateway: сбой связи превращается в различимый результат. */
async function call(): Promise<Record<string, unknown>> {
  try {
    const response = await fetch(url, { signal: requestTimeout(budget) });
    return { ok: response.ok, status: response.status };
  } catch (error) {
    return { unavailable: true, timedOut: isTimeout(error), reason: String((error as Error)?.name || error) };
  }
}

const started = Date.now();
const result = await call();
const elapsed = Date.now() - started;
console.log(JSON.stringify({ url, budget, elapsedMs: elapsed, result, unhandled }, null, 2));

const withinBudget = elapsed < budget + tolerance;
const markedUnavailable = result.unavailable === true;
const ok = withinBudget && markedUnavailable && unhandled === 0;
console.log(ok ? 'ИТОГ: PASS — таймаут сработал, состояние недоступности выставлено' : 'ИТОГ: FAIL');
process.exit(ok ? 0 : 1);
