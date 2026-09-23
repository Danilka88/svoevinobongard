// e2e/fixes.js — целевая проверка фиксов C-046/C-047 по разметке и загрузке данных.
//
// Запуск: lightpanda run apps/web/e2e/fixes.js
//         lightpanda run --experimental-features cors apps/web/e2e/fixes.js   (реальная CORS-семантика)
//
// Проверяет:
//   BUG-003: <html lang="ru"> на всех 5 маршрутах.
//   BUG-004: ровно один <h1> на каждом маршруте, с ожидаемым текстом.
//   BUG-005: в браузерном режиме (в т.ч. с включённым CORS) дашборд /metrics и витрина
//            /catalog реально получают данные от gateway (метрики и плитки), а не тост об ошибке.
const page = new Page();
const BASE = "http://127.0.0.1:3000";

const checks = [];
const note = (name, status, detail) => checks.push({ name: name, status: status, detail: String(detail === undefined ? "" : detail) });
const ok = (name, cond, detail) => note(name, cond ? "PASS" : "FAIL", detail);
const info = (name, detail) => note(name, "INFO", detail);
const ev = (js) => { try { return page.evaluate(js); } catch (e) { return "__EVAL_ERR__:" + String(e); } };
const j = (js) => { const raw = ev(js); try { return JSON.parse(raw); } catch (e) { return { __raw: String(raw) }; } };
const waitFor = (script, ms) => { try { page.waitForScript({ script: script, timeout: ms || 20000 }); return true; } catch (e) { return false; } };

const ROUTES = [
  { path: "/", h1: "Своё Вино" },
  { path: "/sommelier", h1: "Сомелье" },
  { path: "/how", h1: "Как это работает" },
  { path: "/catalog", h1: "Каталог вин" },
  { path: "/metrics", h1: "Метрики качества" },
];

// BUG-003 / BUG-004: язык документа и единственный h1 на каждом маршруте
for (const route of ROUTES) {
  await page.goto(BASE + route.path);
  waitFor("document.readyState==='complete' && !!document.querySelector('main h1')", 20000);
  const v = j("JSON.stringify({" +
    "lang:document.documentElement.lang||''," +
    "h1count:document.querySelectorAll('h1').length," +
    "h1all:Array.from(document.querySelectorAll('h1')).map(function(h){return h.textContent.trim()})," +
    "mainh1:(document.querySelector('main h1')||{}).textContent||null" +
    "})");
  ok("BUG-003 " + route.path + " html[lang]=ru", v.lang === "ru", "lang=" + JSON.stringify(v.lang));
  ok("BUG-004 " + route.path + " exactly one h1", v.h1count === 1, "h1count=" + v.h1count + " texts=" + JSON.stringify(v.h1all));
  ok("BUG-004 " + route.path + " h1 text", (v.h1all || [])[0] === route.h1, "h1=" + JSON.stringify(v.h1all) + " expected=" + JSON.stringify(route.h1));
}

// BUG-005: данные реально доезжают от gateway до UI (в браузерном контексте; с CORS-флагом —
// это ещё и полноценная проверка preflight/ACAO: без заголовков запрос был бы отклонён браузером)
await page.goto(BASE + "/metrics");
const metricsReady = waitFor("document.querySelectorAll('.metric').length>=6 || !!document.querySelector('.toast')", 30000);
const met = j("JSON.stringify({metrics:document.querySelectorAll('.metric').length,toast:(document.querySelector('.toast')||{}).textContent||null})");
ok("BUG-005 /metrics loads data from gateway", metricsReady && met.metrics >= 6 && met.toast === null, JSON.stringify(met));

await page.goto(BASE + "/catalog");
const catReady = waitFor("document.querySelectorAll('.wine-tile').length>0 || !!document.querySelector('.toast')", 30000);
const cat = j("JSON.stringify({tiles:document.querySelectorAll('.wine-tile').length,total:(document.querySelector('.badge')||{}).textContent||null,toast:(document.querySelector('.toast')||{}).textContent||null})");
ok("BUG-005 /catalog loads data from gateway", catReady && cat.tiles > 0 && cat.toast === null, JSON.stringify(cat));

info("mode", "cors enforcement depends on --experimental-features cors at lightpanda run");

const failed = checks.filter((c) => c.status === "FAIL");
return { script: "fixes", status: failed.length ? "FAIL" : "PASS", checks: checks };
