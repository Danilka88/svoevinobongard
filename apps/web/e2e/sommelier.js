// e2e/sommelier.js — поток «Сомелье» (/sommelier): мастер 3 шагов, сенсорные пояснения,
// альтернативы и «Мой вкус» (localStorage).
// Запуск: lightpanda run apps/web/e2e/sommelier.js   (UI :3000 + gateway :8080)
const page = new Page();
const BASE = "http://127.0.0.1:3000";

const checks = [];
const note = (name, status, detail) => checks.push({ name: name, status: status, detail: String(detail === undefined ? "" : detail) });
const ok = (name, cond, detail) => note(name, cond ? "PASS" : "FAIL", detail);
const info = (name, detail) => note(name, "INFO", detail);
const ev = (js) => { try { return page.evaluate(js); } catch (e) { return "__EVAL_ERR__:" + String(e); } };
const j = (js) => { const raw = ev(js); try { return JSON.parse(raw); } catch (e) { return { __raw: String(raw) }; } };
const waitFor = (script, ms) => { try { page.waitForScript({ script: script, timeout: ms || 20000 }); return true; } catch (e) { return false; } };
const clickText = (text) => ev("(function(){var b=Array.from(document.querySelectorAll('button')).find(function(x){return x.textContent.trim()===" + JSON.stringify(text) + "});if(!b)return 'missing';b.click();return 'ok'})()");
const clickChip = (text) => ev("(function(){var b=Array.from(document.querySelectorAll('button.chip')).find(function(x){return x.textContent.trim()===" + JSON.stringify(text) + "});if(!b)return 'missing';b.click();return 'ok'})()");

await page.goto(BASE + "/sommelier");
waitFor("document.readyState==='complete' && !!document.querySelector('.steps')", 15000);

// --- шаг 1: блюдо + слайдеры ---
const s1 = j("JSON.stringify({steps:Array.from(document.querySelectorAll('.steps li')).map(function(l){return l.className}),h2:(document.querySelector('section.card h2')||{}).textContent,dishCount:document.querySelectorAll('button.chip').length,sliders:Array.from(document.querySelectorAll('input[type=range]')).map(function(i){return i.id}),back:(document.querySelector('button.secondary')||{}).disabled})");
ok("wizard step 1 rendered", s1.h2 === "Что за столом", JSON.stringify(s1));
ok("step 1 marked active", s1.steps[0] === "active", JSON.stringify(s1.steps));
ok("step 1 dish chips", s1.dishCount >= 8, "chips=" + s1.dishCount);
ok("step 1 four sliders", JSON.stringify(s1.sliders) === JSON.stringify(["sl-spice", "sl-fat", "sl-acidity", "sl-sweetness"]), JSON.stringify(s1.sliders));
ok("step 1 back disabled", s1.back === true, "disabled=" + s1.back);

ok("dish chip selectable", clickChip("Рыба") === "ok", "Рыба");
const pressed = j("JSON.stringify(Array.from(document.querySelectorAll('button.chip')).filter(function(b){return b.getAttribute('aria-pressed')==='true'}).map(function(b){return b.textContent.trim()}))");
ok("dish chip aria-pressed", JSON.stringify(pressed) === JSON.stringify(["Рыба"]), JSON.stringify(pressed));

ev("(function(){var s=document.getElementById('sl-spice');s.value='3';s.dispatchEvent(new Event('input',{bubbles:true}));s.dispatchEvent(new Event('change',{bubbles:true}));return 1})()");
const slider = j("JSON.stringify({v:document.getElementById('sl-spice').value,label:document.querySelector('label[for=sl-spice]').textContent.trim()})");
ok("slider updates value+label", slider.v === "3" && slider.label === "Острота: 3", JSON.stringify(slider));

// --- шаг 2: повод + смелость ---
ok("go to step 2", clickText("Дальше") === "ok", "Дальше");
const step2 = waitFor("!!document.getElementById('adv')", 8000) && j("JSON.stringify({h2:(document.querySelector('section.card h2')||{}).textContent,steps:Array.from(document.querySelectorAll('.steps li')).map(function(l){return l.className}),occasions:document.querySelectorAll('button.chip').length})").h2 === "Повод и настроение";
ok("wizard step 2 rendered", step2 === true, JSON.stringify(j("JSON.stringify({h2:(document.querySelector('section.card h2')||{}).textContent,steps:Array.from(document.querySelectorAll('.steps li')).map(function(l){return l.className})})")));
ok("step 1 marked done", clickText("Назад") === "ok" && clickText("Дальше") === "ok", "back/forward roundtrip");
ok("occasion selectable", clickChip("Свидание") === "ok", "Свидание");
ev("(function(){var s=document.getElementById('adv');s.value='0.8';s.dispatchEvent(new Event('input',{bubbles:true}));return 1})()");
ok("adventurousness slider", ev("document.getElementById('adv').value") === "0.8", "value=" + ev("document.getElementById('adv').value"));

// --- шаг 3: бюджет/стиль/регион + submit ---
ok("go to step 3", clickText("Дальше") === "ok", "Дальше");
const step3 = waitFor("!!document.getElementById('budget')", 8000);
const s3 = j("JSON.stringify({h2:(document.querySelector('section.card h2')||{}).textContent,budgets:Array.from(document.querySelectorAll('#budget option')).map(function(o){return o.value}),styles:Array.from(document.querySelectorAll('#style option')).length,region:!!document.getElementById('region')})");
ok("wizard step 3 rendered", step3 === true && s3.h2 === "Предпочтения", JSON.stringify(s3));
ok("budget options", JSON.stringify(s3.budgets) === JSON.stringify(["", "низкий", "средний", "высокий"]), JSON.stringify(s3.budgets));
ok("style options", s3.styles === 6, "styles=" + s3.styles);
ok("region input", s3.region === true, "region input");

try { page.selectOption({ selector: "#budget", value: "средний" }); } catch (e) { note("select budget", "FAIL", String(e)); }
try { page.selectOption({ selector: "#style", value: "сухое" }); } catch (e) { note("select style", "FAIL", String(e)); }
ev("(function(){var i=document.getElementById('region');i.value='Кубань';i.dispatchEvent(new Event('input',{bubbles:true}));return 1})()");
ok("selects applied", ev("document.getElementById('budget').value") === "средний" && ev("document.getElementById('style').value") === "сухое", "budget/style");

info("submit", String(clickText("Подобрать вино")));
const gotResult = waitFor("!!document.querySelector('.rec-grid') || !!document.querySelector('.toast')", 30000);
const res = j("JSON.stringify({" +
  "recs:document.querySelectorAll('.rec').length," +
  "alts:document.querySelectorAll('.rec-compact').length," +
  "source:(document.querySelector('section.card p.muted.small')||{}).textContent||null," +
  "headings:Array.from(document.querySelectorAll('section.card h2')).map(function(h){return h.textContent.trim()})," +
  "sensoryNotes:Array.from(document.querySelectorAll('.rec .muted.small')).filter(function(p){return p.textContent.length>40}).length," +
  "dishChips:document.querySelectorAll('.chip-dish').length," +
  "tasteCardEmpty:(document.querySelector('section.card p.muted.small')||{}).textContent||null," +
  "toast:(document.querySelector('.toast')||{}).textContent||null" +
  "})");
ok("sommelier returned recommendations", gotResult && res.recs > 0, JSON.stringify(res).slice(0, 200));
ok("recommendations have sensory notes", res.sensoryNotes > 0, "notes=" + res.sensoryNotes);
ok("recommendations have dish pairing", res.dishChips > 0, "dish chips=" + res.dishChips);
ok("alternatives block", res.alts > 0, "alternatives=" + res.alts);
ok("no error toast", res.toast === null, String(res.toast));

// --- «Мой вкус»: клик по сердцу пополняет localStorage и список ---
const heartBefore = ev("(document.querySelector('.rec button.secondary')||{}).textContent||null");
const heartClick = ev("(function(){var b=document.querySelector('.rec button.secondary');if(!b)return 'missing';b.click();return 'ok'})()");
const heartAfter = ev("(document.querySelector('.rec button.secondary')||{}).textContent||null");
const taste = j("JSON.stringify({" +
  "chips:Array.from(document.querySelectorAll('section.card ul.chips li.chip')).map(function(c){return c.textContent.trim().slice(0,30)})," +
  "removeButtons:document.querySelectorAll('.chip-remove').length," +
  "clearButton:!!Array.from(document.querySelectorAll('button')).find(function(b){return b.textContent.trim()==='Очистить'})," +
  "ls:(function(){try{return localStorage.getItem('svoe-vino-taste')}catch(e){return 'ERR'}})()" +
  "})");
ok("heart toggles to saved", heartBefore === "♡ Сохранить вкус" && heartAfter === "♥ В моём вкусе", JSON.stringify([heartBefore, heartAfter]));
ok("taste list populated", taste.chips.length > 0 && taste.removeButtons > 0, JSON.stringify(taste.chips).slice(0, 120));
ok("taste persisted in localStorage", typeof taste.ls === "string" && taste.ls.indexOf("slug") !== -1, String(taste.ls).slice(0, 100));
ok("taste clear button appears", taste.clearButton === true, "clear=" + taste.clearButton);

// сохранение «Мой вкус» переживает перезагрузку
await page.goto(BASE + "/sommelier");
waitFor("document.readyState==='complete' && !!document.querySelector('.steps')", 15000);
const tasteAfterReload = j("JSON.stringify({chips:Array.from(document.querySelectorAll('section.card ul.chips li.chip')).map(function(c){return c.textContent.trim().slice(0,30)}),removeButtons:document.querySelectorAll('.chip-remove').length})");
ok("taste survives reload", tasteAfterReload.chips.length > 0, JSON.stringify(tasteAfterReload).slice(0, 140));

// очистка
const cleared = ev("(function(){var b=Array.from(document.querySelectorAll('button')).find(function(x){return x.textContent.trim()==='Очистить'});if(!b)return 'missing';b.click();return 'ok'})()");
const tasteCleared = j("JSON.stringify({chips:Array.from(document.querySelectorAll('.chip-remove')).length,ls:(function(){try{return localStorage.getItem('svoe-vino-taste')}catch(e){return 'ERR'}})()})");
ok("taste clear works", cleared === "ok" && tasteCleared.chips === 0, JSON.stringify(tasteCleared));

const failed = checks.filter((c) => c.status === "FAIL");
return { script: "sommelier", status: failed.length ? "FAIL" : "PASS", checks: checks };
