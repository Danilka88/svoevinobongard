// e2e/catalog-search.js — регрессия BUG-001: текстовый поиск в каталоге.
//
// Ожидаемое поведение: ввод в поле «Поиск по названию, винодельне или сорту» фильтрует
// витрину и обновляет счётчик «Найдено: N» за время в пределах SLA проекта (p95 ≤ 3000 мс).
//
// Фактическое: каждый запрос с q= занимает ~14 с и держит ядро на 100% CPU.
// Причина: services/gateway/app/media.py::catalog_search для каждой из 2103 записей,
// у которой запрос не является подстрокой, вызывает packages/common/text_norm.py::similarity()
// (~6.7 мс на вызов) — это ~2000 вызовов × 6.7 мс на каждый поиск.
// Следствие для UI: витрина показывает скелетоны 14+ с, а клиенты с коротким таймаутом
// (например, Lightpanda) успевают отвалиться и получают «Не удалось загрузить каталог».
//
// Запуск: lightpanda run apps/web/e2e/catalog-search.js
// ВАЖНО: скрипт нагружает gateway на ~30 с; запускать последним и перезапускать gateway после.
const page = new Page();
const BASE = "http://127.0.0.1:3000";
const SLA_MS = 3000;

const checks = [];
const note = (name, status, detail) => checks.push({ name: name, status: status, detail: String(detail === undefined ? "" : detail) });
const ok = (name, cond, detail) => note(name, cond ? "PASS" : "FAIL", detail);
const info = (name, detail) => note(name, "INFO", detail);
const ev = (js) => { try { return page.evaluate(js); } catch (e) { return "__EVAL_ERR__:" + String(e); } };
const j = (js) => { const raw = ev(js); try { return JSON.parse(raw); } catch (e) { return { __raw: String(raw) }; } };
const waitFor = (script, ms) => { try { page.waitForScript({ script: script, timeout: ms || 20000 }); return true; } catch (e) { return false; } };

await page.goto(BASE + "/catalog");
waitFor("document.readyState==='complete' && document.querySelectorAll('.wine-tile').length>0", 25000);
const before = j("JSON.stringify({total:(document.querySelector('.badge')||{}).textContent,tiles:document.querySelectorAll('.wine-tile').length})");
ok("catalog loaded before search", before.tiles === 24, JSON.stringify(before));

// прямой замер gateway-эндпоинта из контекста страницы: тот же путь, что делает UI
ev("(function(){window.__e2eQ='start';window.__e2eQms=0;var t0=Date.now();" +
  "fetch('http://127.0.0.1:8080/v1/catalog?q=%D1%80%D0%B8%D1%81%D0%BB%D0%B8%D0%BD%D0%B3&sort=name&page=1&per_page=24')" +
  ".then(function(r){return r.text().then(function(t){window.__e2eQms=Date.now()-t0;window.__e2eQ='HTTP '+r.status+' len='+t.length})})" +
  ".catch(function(e){window.__e2eQms=Date.now()-t0;window.__e2eQ='ERR:'+e});return 'kicked'})()");
const directSettled = waitFor("window.__e2eQ!=='start'", 120000);
const direct = j("JSON.stringify({result:window.__e2eQ, ms:window.__e2eQms})");
ok("gateway /v1/catalog?q= responds", directSettled && String(direct.result).indexOf("HTTP 200") === 0, String(direct.result));
ok("gateway search latency <= SLA " + SLA_MS + "ms", direct.ms <= SLA_MS, "measured=" + direct.ms + " ms (SLA " + SLA_MS + " ms)");
info("gateway direct probe", direct.result + " in " + direct.ms + " ms");

// тот же путь через UI: измерить время от ввода до результатов
const t0 = Number(ev("Date.now()"));
page.fill({ selector: "#q", value: "рислинг" });
const settled = waitFor("!!document.querySelector('.toast') || (document.querySelectorAll('.wine-tile').length>0 && ((document.querySelector('.badge')||{}).textContent||'')!=='Найдено: 2103')", 120000);
const elapsed = Number(ev("Date.now()")) - t0;
const after = j("JSON.stringify({" +
  "q:document.getElementById('q').value," +
  "total:(document.querySelector('.badge')||{}).textContent," +
  "tiles:document.querySelectorAll('.wine-tile').length," +
  "toast:(document.querySelector('.toast')||{}).textContent||null," +
  "hasReset:!!Array.from(document.querySelectorAll('button')).find(function(x){return x.textContent.trim()==='Сбросить'})" +
  "})");
info("ui search", "settled=" + settled + " elapsed=" + elapsed + " ms " + JSON.stringify(after));

ok("search input accepts text", after.q === "рислинг", "q=" + after.q);
ok("search filters the showcase", after.tiles > 0 && after.total !== "Найдено: 2103", "tiles=" + after.tiles + " total=" + after.total);
ok("search shows no error toast", after.toast === null, String(after.toast));
ok("UI search latency <= SLA " + SLA_MS + "ms", elapsed <= SLA_MS, "measured=" + elapsed + " ms (SLA " + SLA_MS + " ms)");

const failed = checks.filter((c) => c.status === "FAIL");
return { script: "catalog-search", status: failed.length ? "FAIL" : "PASS", checks: checks };
