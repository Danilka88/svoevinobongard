// e2e/learning.js — страница «Обучение и качество» (C-055).
//
// Проверяет: страница отвечает, показывает разделы механизма (консенсус, очередь, качество ответов
// моделей, рост датасета), помечает раздел модератора и корректно показывает пустые состояния.
// Запуск: lightpanda run apps/web/e2e/learning.js   (UI :3000 + gateway :8080)
const page = new Page();
const BASE = "http://127.0.0.1:3000";

const checks = [];
const note = (name, status, detail) => checks.push({ name: name, status: status, detail: String(detail === undefined ? "" : detail) });
const ok = (name, cond, detail) => note(name, cond ? "PASS" : "FAIL", detail);
const info = (name, detail) => note(name, "INFO", detail);
const ev = (js) => { try { return page.evaluate(js); } catch (e) { return "__EVAL_ERR__:" + String(e); } };
const j = (js) => { const raw = ev(js); try { return JSON.parse(raw); } catch (e) { return { __raw: String(raw) }; } };
const waitFor = (script, ms) => { try { page.waitForScript({ script: script, timeout: ms || 20000 }); return true; } catch (e) { return false; } };

await page.goto(BASE + "/learning");
waitFor("document.readyState==='complete' && document.querySelectorAll('main h1').length>0", 15000);
const loaded = waitFor("!document.querySelector('.skeleton') && (document.querySelectorAll('section.card').length>2 "
  + "|| !!document.querySelector('.unavailable'))", 25000);
const view = j("JSON.stringify({"
  + "status:(document.querySelector('main h1')||{}).textContent||null,"
  + "sections:Array.from(document.querySelectorAll('section.card')).map(function(s){return (s.querySelector('h2')||{}).textContent||''}),"
  + "hasConsensus:document.body.innerText.indexOf('Состояние консенсуса')!==-1,"
  + "hasQueue:document.body.innerText.indexOf('Очередь спорных этикеток')!==-1,"
  + "hasLlm:document.body.innerText.indexOf('Качество ответов моделей')!==-1,"
  + "hasDataset:document.body.innerText.indexOf('Рост датасета')!==-1,"
  + "moderator:document.body.innerText.indexOf('Раздел для модератора')!==-1,"
  + "rules:document.body.innerText.indexOf('независимых сессий')!==-1,"
  + "queueItems:document.querySelectorAll('.queue-item').length,"
  + "emptyQueue:document.body.innerText.indexOf('Очередь пуста')!==-1,"
  + "emptyDataset:document.body.innerText.indexOf('Датасет пока пуст')!==-1,"
  + "emptyReasons:document.body.innerText.indexOf('Пока нет голосов с причинами')!==-1,"
  + "unavailable:!!document.querySelector('.unavailable'),"
  + "toast:(document.querySelector('.toast')||{}).textContent||null,"
  + "metrics:document.querySelectorAll('.metric').length,"
  + "bodyLen:(document.body.innerText||'').trim().length})");

ok("/learning renders", view.bodyLen > 200, "body chars=" + view.bodyLen);
ok("/learning title", view.status === "Обучение и качество", "h1=" + view.status);
ok("/learning data loaded", loaded, "sections=" + JSON.stringify(view.sections).slice(0, 140));
ok("/learning section: consensus", view.hasConsensus === true, "consensus=" + view.hasConsensus);
ok("/learning section: moderation queue", view.hasQueue === true, "queue=" + view.hasQueue);
ok("/learning section: llm quality", view.hasLlm === true, "llm=" + view.hasLlm);
ok("/learning section: dataset growth", view.hasDataset === true, "dataset=" + view.hasDataset);
ok("/learning moderator badge", view.moderator === true, "badge=" + view.moderator);
ok("/learning trust rules shown", view.rules === true, "rules=" + view.rules);
ok("/learning counters shown", view.metrics >= 4, "metrics=" + view.metrics);
// Пустое состояние допустимо только когда данных действительно нет: либо карточки, либо явный текст.
ok("/learning queue state is explicit", view.queueItems > 0 || view.emptyQueue === true,
   "items=" + view.queueItems + " empty=" + view.emptyQueue);
ok("/learning no error toast", view.toast === null, String(view.toast));
info("queue items", String(view.queueItems));
info("sections", JSON.stringify(view.sections));

note("learning.js", "DONE", "checks=" + checks.length);
console.log(JSON.stringify({ script: "learning", status: "ok", checks: checks }, null, 0));
