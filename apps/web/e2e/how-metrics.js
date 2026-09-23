// e2e/how-metrics.js — поток «Метрики» (/metrics): дашборд из /v1/demo/metrics,
// состояние гейта, честные оговорки и источники. Плюс статическая часть /how без загрузки фото.
// Запуск: lightpanda run apps/web/e2e/how-metrics.js   (UI :3000 + gateway :8080)
const page = new Page();
const BASE = "http://127.0.0.1:3000";

const checks = [];
const note = (name, status, detail) => checks.push({ name: name, status: status, detail: String(detail === undefined ? "" : detail) });
const ok = (name, cond, detail) => note(name, cond ? "PASS" : "FAIL", detail);
const info = (name, detail) => note(name, "INFO", detail);
const ev = (js) => { try { return page.evaluate(js); } catch (e) { return "__EVAL_ERR__:" + String(e); } };
const j = (js) => { const raw = ev(js); try { return JSON.parse(raw); } catch (e) { return { __raw: String(raw) }; } };
const waitFor = (script, ms) => { try { page.waitForScript({ script: script, timeout: ms || 20000 }); return true; } catch (e) { return false; } };

// --- /metrics: данные подгружаются клиентским fetch после гидрации ---
await page.goto(BASE + "/metrics");
waitFor("document.readyState==='complete'", 15000);
const loaded = waitFor("!!document.querySelector('.metric') || !!document.querySelector('.toast')", 25000);
const m = j("JSON.stringify({" +
  "metrics:document.querySelectorAll('.metric').length," +
  "labels:Array.from(document.querySelectorAll('.metric-label')).map(function(n){return n.textContent})," +
  "values:Array.from(document.querySelectorAll('.metric-value')).map(function(n){return n.textContent})," +
  "gate:(document.querySelector('section.card .badge')||{}).textContent||null," +
  "sections:Array.from(document.querySelectorAll('section.card h2')).map(function(h){return h.textContent})," +
  "limits:document.querySelectorAll('.limits li').length," +
  "sources:(function(){var p=Array.from(document.querySelectorAll('p.muted.small')).find(function(x){return x.textContent.indexOf('Источники')!==-1});return p?p.textContent.slice(0,120):null})()," +
  "skeletons:document.querySelectorAll('.skeleton').length," +
  "toast:(document.querySelector('.toast')||{}).textContent||null," +
  "refreshBtn:!!Array.from(document.querySelectorAll('button')).find(function(b){return b.textContent.trim()==='Обновить'})" +
  "})");
ok("metrics dashboard loads", loaded && m.metrics >= 4, JSON.stringify(m).slice(0, 200));
ok("metrics no error toast", m.toast === null, String(m.toast));
ok("metrics skeletons replaced", m.skeletons === 0, "skeletons=" + m.skeletons);
ok("gate status badge", m.gate === "пройден" || m.gate === "не пройден" || m.gate === "нет данных", "badge=" + m.gate);
ok("metrics sections present", (m.sections || []).length >= 4, JSON.stringify(m.sections));
ok("metrics honest notes", m.limits >= 3, "notes=" + m.limits);
ok("metrics sources listed", typeof m.sources === "string" && m.sources.indexOf("Источники") !== -1, String(m.sources));
ok("metrics refresh button", m.refreshBtn === true, "Обновить");
info("metrics values", JSON.stringify(m.labels) + " => " + JSON.stringify(m.values));

// повторная загрузка кнопкой «Обновить»
ev("(function(){var b=Array.from(document.querySelectorAll('button')).find(function(x){return x.textContent.trim()==='Обновить'});if(b)b.click();return 1})()");
const refreshed = waitFor("document.querySelectorAll('.metric').length>=4 && !document.querySelector('.toast')", 20000);
ok("metrics refresh keeps data", refreshed, JSON.stringify(j("JSON.stringify({metric:document.querySelectorAll('.metric').length,toast:(document.querySelector('.toast')||{}).textContent||null})")));

// --- /how: статическая часть (схема, витрина методов, ограничения) без фото ---
await page.goto(BASE + "/how");
waitFor("document.readyState==='complete' && !!document.querySelector('.pipe')", 15000);
const how = j("JSON.stringify({" +
  "pipeline:document.querySelectorAll('.pipe li').length," +
  "methods:document.querySelectorAll('.method').length," +
  "verdicts:Array.from(document.querySelectorAll('.method .badge')).map(function(b){return b.textContent.trim()})," +
  "limits:document.querySelectorAll('.limits li').length," +
  "sources:Array.from(document.querySelectorAll('.method code')).map(function(c){return c.textContent})," +
  "tracePlaceholder:(document.querySelector('.placeholder')||{}).textContent||null," +
  "uploader:!!document.querySelector('.dropzone')," +
  "toast:(document.querySelector('.toast')||{}).textContent||null" +
  "})");
ok("/how pipeline steps", how.pipeline >= 5, "steps=" + how.pipeline);
ok("/how methods with verdicts", how.methods >= 5 && (how.verdicts || []).length === how.methods, "methods=" + how.methods + " verdicts=" + JSON.stringify(how.verdicts));
ok("/how limits listed", how.limits >= 3, "limits=" + how.limits);
ok("/how uploader available", how.uploader === true, "dropzone");
ok("/how no error toast", how.toast === null, String(how.toast));
info("/how trace placeholder", String(how.tracePlaceholder).slice(0, 100));

const failed = checks.filter((c) => c.status === "FAIL");
return { script: "how-metrics", status: failed.length ? "FAIL" : "PASS", checks: checks };
