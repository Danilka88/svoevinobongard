/**
 * Глобальный сбор телеметрии (C-044): просмотры страниц, переключение темы, ошибки скрипта.
 *
 * Плагин клиентский: на сервере ничего не выполняется, поэтому разметка не меняется.
 * Строка о прозрачности добавляется в существующий подвал, если он есть; при его отсутствии
 * ничего не происходит — интерфейс не ломается.
 */
export default defineNuxtPlugin(() => {
  const { track, flush, sessionId } = useTelemetry();
  const router = useRouter();

  // Просмотры страниц: только путь, без параметров запроса (там могут быть значения пользователя).
  const recordView = (path: string) => track('page_view', { route: path.split('?')[0], action: 'view' });
  recordView(router.currentRoute.value.path);
  router.afterEach((to) => recordView(to.path));

  // Ошибки скрипта: только короткий код, без текстов сообщений и стеков.
  window.addEventListener('error', (event) => {
    track('js_error', { action: 'window_error', detail: (event as ErrorEvent).message ? 'runtime' : 'resource' });
  });
  window.addEventListener('unhandledrejection', (event) => {
    // C-046/BUG-006: отказ сетевого запроса — это состояние интерфейса, а не ошибка скрипта.
    // Такие отказы уже учтены как api_error в клиенте gateway, поэтому шум не создаём.
    const reason = (event as PromiseRejectionEvent).reason as { status?: number; response?: unknown } | undefined;
    if (reason && (typeof reason.status === 'number' || reason.response)) {
      event.preventDefault();
      return;
    }
    track('js_error', { action: 'unhandled_rejection', detail: 'promise' });
  });

  // Переключение темы: следим за атрибутом на корне документа, не вмешиваясь в логику темы.
  const observer = new MutationObserver(() => {
    const theme = document.documentElement.dataset.theme || document.documentElement.className || '';
    track('theme', { action: 'switch', value: String(theme).slice(0, 24) || 'unknown' });
  });
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme', 'class'] });

  // Уход со страницы: досылаем остаток через sendBeacon, чтобы не терять хвост сессии.
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') flush(true);
  });
  window.addEventListener('pagehide', () => flush(true));

  // Прозрачность: короткая строка в подвале о том, что собирается обезличенная телеметрия.
  const NOTE = 'Обезличенная телеметрия для улучшения сервиса: без персональных данных и изображений.';
  const addNote = () => {
    const footer = document.querySelector('footer');
    if (!footer || footer.querySelector('[data-telemetry-note]')) return;
    const note = document.createElement('p');
    note.setAttribute('data-telemetry-note', '');
    note.className = 'muted small';
    note.style.opacity = '0.7';
    note.style.marginTop = '8px';
    note.textContent = `${NOTE} Сессия: ${sessionId().slice(0, 8)}`;
    footer.appendChild(note);
  };
  addNote();
  router.afterEach(() => nextTick(addNote));
});
