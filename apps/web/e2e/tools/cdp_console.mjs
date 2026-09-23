#!/usr/bin/env node
/**
 * cdp_console.mjs — сбор консольных и JS-ошибок UI через CDP Lightpanda.
 *
 * Зачем: `lightpanda run` не отдаёт console.log/error, а MCP-tool consoleLogs
 * в этой сборке Lightpanda (1.0.0-nightly) ничего не возвращает. Поэтому консоль
 * снимаем через CDP: Page.addScriptToEvaluateOnNewDocument ставит перехватчик
 * ДО выполнения скриптов страницы, поэтому ловятся и ошибки гидрации/SSR-клиента.
 *
 * Требует запущенного `lightpanda serve --port 9222`.
 *
 * Использование:
 *   node e2e/tools/cdp_console.mjs http://127.0.0.1:3000 / /metrics /catalog /sommelier /how
 *   CDP_URL=ws://127.0.0.1:9222/ node e2e/tools/cdp_console.mjs ...
 *
 * Вывод: JSON { base, routes: { "<route>": { console: [...], errors: [...] } } } в stdout.
 */

const CDP_URL = process.env.CDP_URL || 'ws://127.0.0.1:9222/';
const base = process.argv[2] || 'http://127.0.0.1:3000';
const routes = process.argv.slice(3);
if (!routes.length) routes.push('/', '/sommelier', '/how', '/catalog', '/metrics');

const COLLECTOR = `(function(){
  window.__e2e = { console: [], errors: [] };
  var str = function(a){ try { return typeof a === 'string' ? a : JSON.stringify(a); } catch (e) { return String(a); } };
  ['log','info','warn','error','debug'].forEach(function(kind){
    var orig = console[kind] ? console[kind].bind(console) : function(){};
    console[kind] = function(){ try { window.__e2e.console.push({ kind: kind, text: Array.prototype.map.call(arguments, str).join(' ').slice(0, 400) }); } catch(e){} return orig.apply(null, arguments); };
  });
  window.addEventListener('error', function(e){
    window.__e2e.errors.push({ type: 'error', message: String(e.message || ''), source: String(e.filename || ''), line: e.lineno || 0 });
  });
  window.addEventListener('unhandledrejection', function(e){
    var r = e && e.reason;
    window.__e2e.errors.push({ type: 'unhandledrejection', message: String((r && r.message) || r || ''), stack: String((r && r.stack) || '').split('\\n').slice(0, 5).join(' | ') });
  });
})();`;

class Cdp {
  constructor(ws) {
    this.ws = ws;
    this.nextId = 0;
    this.pending = new Map();
    this.listeners = [];
    ws.addEventListener('message', (event) => {
      let msg;
      try { msg = JSON.parse(typeof event.data === 'string' ? event.data : String(event.data)); } catch { return; }
      if (msg.id !== undefined && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (msg.error) reject(new Error(JSON.stringify(msg.error)));
        else resolve(msg.result);
        return;
      }
      if (msg.method) for (const fn of this.listeners) fn(msg);
    });
  }
  static async connect(url) {
    const ws = new WebSocket(url);
    await new Promise((resolve, reject) => {
      ws.addEventListener('open', () => resolve());
      ws.addEventListener('error', (e) => reject(new Error('ws error: ' + (e.message || 'unknown'))));
    });
    return new Cdp(ws);
  }
  send(method, params = {}, sessionId) {
    const id = ++this.nextId;
    const payload = { id, method, params };
    if (sessionId) payload.sessionId = sessionId;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify(payload));
      setTimeout(() => {
        if (this.pending.has(id)) { this.pending.delete(id); reject(new Error('timeout: ' + method)); }
      }, 60000);
    });
  }
  on(fn) { this.listeners.push(fn); }
  close() { try { this.ws.close(); } catch {} }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  const cdp = await Cdp.connect(CDP_URL);
  const created = await cdp.send('Target.createTarget', { url: 'about:blank' });
  const targetId = created.targetId;
  const attached = await cdp.send('Target.attachToTarget', { targetId, flatten: true });
  const session = attached.sessionId;

  const loadEvents = [];
  const cdpConsole = [];
  const cdpErrors = [];
  cdp.on((msg) => {
    if (msg.method === 'Page.loadEventFired') loadEvents.push(Date.now());
    if (msg.method === 'Runtime.consoleAPICalled') {
      const args = (msg.params && msg.params.args) || [];
      cdpConsole.push({
        kind: msg.params.type,
        text: args.map((a) => JSON.stringify(a.value !== undefined ? a.value : a.description || a.type)).join(' ').slice(0, 400),
      });
    }
    if (msg.method === 'Runtime.exceptionThrown') {
      const d = (msg.params && msg.params.exceptionDetails) || {};
      const desc = (d.exception && (d.exception.description || d.exception.value)) || d.text || 'unknown';
      cdpErrors.push({ type: 'exceptionThrown', message: String(desc).split('\n')[0].slice(0, 400) });
    }
    if (msg.method === 'Log.entryAdded') {
      const e = msg.params && msg.params.entry;
      if (!e) return;
      const item = { kind: e.level, text: String(e.text || '').slice(0, 400) };
      if (e.level === 'error' || e.level === 'warning') cdpConsole.push(item);
    }
  });

  await cdp.send('Page.enable', {}, session);
  await cdp.send('Runtime.enable', {}, session);
  await cdp.send('Log.enable', {}, session);
  await cdp.send('Page.addScriptToEvaluateOnNewDocument', { source: COLLECTOR }, session);

  const evaluate = async (expression) => {
    const res = await cdp.send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true }, session);
    if (res.exceptionDetails) return { __error: res.exceptionDetails.text || 'exception' };
    return res.result ? res.result.value : undefined;
  };

  const result = { base, lightpanda: 'CDP', routes: {}, summary: {} };

  for (const route of routes) {
    const before = loadEvents.length;
    const consoleMark = cdpConsole.length;
    const errorMark = cdpErrors.length;
    await cdp.send('Page.navigate', { url: base + route }, session);
    for (let i = 0; i < 100 && loadEvents.length === before; i++) await sleep(100);
    // Дать клиентским фетчам и гидрации завершиться (SPA-страницы грузят данные после load).
    await sleep(2500);
    const view = await evaluate(`(function(){
      var out = { url: location.pathname, title: document.title, readyState: document.readyState,
                  h1: Array.from(document.querySelectorAll('h1')).map(function(n){return n.textContent.trim()}),
                  toast: (document.querySelector('.toast') || {}).textContent || null,
                  bodyText: (document.body.innerText || '').replace(/\\s+/g,' ').slice(0, 160),
                  console: (window.__e2e && window.__e2e.console) || [],
                  errors: (window.__e2e && window.__e2e.errors) || [] };
      return out;
    })()`);
    const inPage = view && !view.__error ? view : { error: view && view.__error };
    inPage.cdpConsole = cdpConsole.slice(consoleMark);
    inPage.cdpErrors = cdpErrors.slice(errorMark);
    result.routes[route] = inPage;
  }

  const allConsole = [];
  const allErrors = [];
  for (const [route, view] of Object.entries(result.routes)) {
    for (const line of (view && view.console) || []) allConsole.push({ route, source: 'in-page', ...line });
    for (const line of (view && view.cdpConsole) || []) allConsole.push({ route, source: 'cdp', ...line });
    for (const line of (view && view.errors) || []) allErrors.push({ route, source: 'in-page', ...line });
    for (const line of (view && view.cdpErrors) || []) allErrors.push({ route, source: 'cdp', ...line });
  }
  result.summary = {
    routes: routes.length,
    console: allConsole.length,
    errors: allErrors.length,
    consoleLines: allConsole,
    errorLines: allErrors,
  };

  cdp.close();
  process.stdout.write(JSON.stringify(result, null, 2) + '\n');
}

main().catch((err) => {
  process.stderr.write('FATAL: ' + (err && err.stack ? err.stack : String(err)) + '\n');
  process.exit(1);
});
