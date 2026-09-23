// e2e/error-state.js — поток «Ошибочные сценарии»: gateway недоступен.
// Запуск: СНАЧАЛА остановить gateway (порт 8080), затем:
//   lightpanda run apps/web/e2e/error-state.js
// Проверяет: экраны показывают понятное состояние «сервис недоступен», витрина не становится
// белым экраном, кнопки снова активны, необработанных JS-исключений нет.
const page = new Page();
const BASE = "http://127.0.0.1:3000";
const FIXTURE = "http://127.0.0.1:8099/96.13_19-08-2026_19-33-10.webp";

const checks = [];
const note = (name, status, detail) => checks.push({ name: name, status: status, detail: String(detail === undefined ? "" : detail) });
const ok = (name, cond, detail) => note(name, cond ? "PASS" : "FAIL", detail);
const info = (name, detail) => note(name, "INFO", detail);
const ev = (js) => { try { return page.evaluate(js); } catch (e) { return "__EVAL_ERR__:" + String(e); } };
const j = (js) => { const raw = ev(js); try { return JSON.parse(raw); } catch (e) { return { __raw: String(raw) }; } };
const waitFor = (script, ms) => { try { page.waitForScript({ script: script, timeout: ms || 20000 }); return true; } catch (e) { return false; } };

// ловим необработанные исключения страницы для каждого экрана
const HOOK = "window.__e2eErrs=[];window.addEventListener('error',function(e){window.__e2eErrs.push('error:'+e.message)});window.addEventListener('unhandledrejection',function(e){window.__e2eErrs.push('rejection:'+String(e.reason&&e.reason.message||e.reason))});";
const errs = () => j("JSON.stringify(window.__e2eErrs || [])");

// --- /metrics ---
await page.goto(BASE + "/metrics");
waitFor("document.readyState==='complete'", 15000);
ev(HOOK);
const metricsToast = waitFor("!!document.querySelector('.toast') || !!document.querySelector('.metric')", 25000);
const met = j("JSON.stringify({toast:(document.querySelector('.toast')||{}).textContent||null,metric:document.querySelectorAll('.metric').length,skel:document.querySelectorAll('.skeleton').length,h1:(document.querySelector('main h1')||{}).textContent||null,bodyLen:(document.body.innerText||'').trim().length})");
ok("/metrics shows service-unavailable toast", metricsToast && /недоступен/i.test(met.toast || ""), "toast=" + met.toast);
ok("/metrics no white screen", met.bodyLen > 80 && met.h1 !== null, "body chars=" + met.bodyLen + " h1=" + met.h1);
ok("/metrics no loading skeleton stuck", met.skel === 0, "skeletons=" + met.skel);
ok("/metrics no unhandled promise rejections", errs().length === 0, JSON.stringify(errs()));

// --- /catalog ---
await page.goto(BASE + "/catalog");
waitFor("document.readyState==='complete'", 15000);
ev(HOOK);
const catToast = waitFor("!!document.querySelector('.toast') || document.querySelectorAll('.wine-tile').length>0", 25000);
const cat = j("JSON.stringify({toast:(document.querySelector('.toast')||{}).textContent||null,tiles:document.querySelectorAll('.wine-tile').length,filters:document.querySelectorAll('.filter-grid select').length,h1:(document.querySelector('main h1')||{}).textContent||null,bodyLen:(document.body.innerText||'').trim().length})");
ok("/catalog shows service-unavailable toast", catToast && /каталог/i.test(cat.toast || ""), "toast=" + cat.toast);
ok("/catalog keeps filters usable", cat.filters === 4, "selects=" + cat.filters);
ok("/catalog no white screen", cat.bodyLen > 120 && cat.h1 !== null, "body chars=" + cat.bodyLen + " h1=" + cat.h1);
ok("/catalog no unhandled promise rejections", errs().length === 0, JSON.stringify(errs()));

// --- /sommelier: мастер проходит, submit честно сообщает об ошибке ---
await page.goto(BASE + "/sommelier");
waitFor("document.readyState==='complete' && !!document.querySelector('.steps')", 15000);
ev(HOOK);
ev("(function(){var b=Array.from(document.querySelectorAll('button')).find(function(x){return x.textContent.trim()==='Дальше'});if(b)b.click();return 1})()");
waitFor("!!document.getElementById('adv')", 8000);
ev("(function(){var b=Array.from(document.querySelectorAll('button')).find(function(x){return x.textContent.trim()==='Дальше'});if(b)b.click();return 1})()");
waitFor("!!document.getElementById('budget')", 8000);
ev("(function(){var b=Array.from(document.querySelectorAll('button')).find(function(x){return x.textContent.trim()==='Подобрать вино'});if(b)b.click();return 1})()");
const somToast = waitFor("!!document.querySelector('.toast')", 25000);
const som = j("JSON.stringify({toast:(document.querySelector('.toast')||{}).textContent||null,recs:document.querySelectorAll('.rec').length,steps:document.querySelectorAll('.steps li').length,submit:(function(){var b=Array.from(document.querySelectorAll('button')).find(function(x){return /Подобрать вино|Подбираю/.test(x.textContent)});return b?b.textContent.trim():null})(),bodyLen:(document.body.innerText||'').trim().length})");
ok("/sommelier shows service-unavailable toast", somToast && /gateway/i.test(som.toast || ""), "toast=" + som.toast);
ok("/sommelier wizard stays usable", som.steps === 3 && som.submit === "Подобрать вино", JSON.stringify(som).slice(0, 120));
ok("/sommelier no white screen", som.bodyLen > 120, "body chars=" + som.bodyLen);
ok("/sommelier no unhandled promise rejections", errs().length === 0, JSON.stringify(errs()));

// --- / (сканер): загрузка фото при мёртвом gateway ---
await page.goto(BASE + "/");
waitFor("document.readyState==='complete' && !!document.querySelector('input[type=file]')", 15000);
ev(HOOK);
ev("(function(){window.__e2eF='start';fetch(" + JSON.stringify(FIXTURE) + ").then(function(r){return r.blob()}).then(function(b){" +
  "var f=new File([b],'label.webp',{type:'image/webp'});var dt=new DataTransfer();dt.items.add(f);" +
  "var i=document.querySelector('input[type=file]');i.files=dt.files;i.dispatchEvent(new Event('change',{bubbles:true}));window.__e2eF='set'}).catch(function(e){window.__e2eF='ERR:'+e});return 1})()");
waitFor("window.__e2eF!=='start'", 15000);
const scanToast = waitFor("!!document.querySelector('.toast')", 40000);
const scan = j("JSON.stringify({toast:(document.querySelector('.toast')||{}).textContent||null,button:(function(){var b=document.querySelector('.card button.primary');return b?b.textContent:null})(),disabled:(function(){var b=document.querySelector('.card button.primary');return b?b.disabled:null})(),uploader:!!document.querySelector('.dropzone'),bodyLen:(document.body.innerText||'').trim().length})");
ok("/scan shows service-unavailable toast", scanToast && /недоступен/i.test(scan.toast || ""), "toast=" + scan.toast);
ok("/scan button re-enabled after failure", scan.disabled === false && /Снять или выбрать/.test(scan.button || ""), JSON.stringify(scan).slice(0, 120));
ok("/scan uploader still available", scan.uploader === true, "dropzone");
ok("/scan no white screen", scan.bodyLen > 120, "body chars=" + scan.bodyLen);
ok("/scan no unhandled promise rejections", errs().length === 0, JSON.stringify(errs()));

info("note", "скрипт предполагает остановленный gateway :8080; при живом gateway проверки toast ожидаемо упадут");

const failed = checks.filter((c) => c.status === "FAIL");
return { script: "error-state", status: failed.length ? "FAIL" : "PASS", checks: checks };
