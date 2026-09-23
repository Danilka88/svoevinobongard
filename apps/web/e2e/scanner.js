// e2e/scanner.js — поток «Сканер» (/) и «Как это работает» (/how).
// Запуск: lightpanda run apps/web/e2e/scanner.js
// Требует: UI :3000, gateway :8080, retrieval :8082, normalize :8081, ocr :8085,
//          фикстуры :8099 (python3 -m http.server 8099 --directory data/samples/real100).
//
// Lightpanda не умеет выбирать файл в <input type=file> из CLI, но умеет собрать File
// в самой странице: fetch фикстуры -> Blob -> File -> DataTransfer -> input.files -> change.
// Это ровно тот путь, которым пользуется пользователь (выбор файла), поэтому поток
// проверяется end-to-end, а не через прямой вызов API.
const page = new Page();
const BASE = "http://127.0.0.1:3000";
const FIXTURE = "http://127.0.0.1:8099/96.13_19-08-2026_19-33-10.webp"; // реальное фото этикетки
const SYNTHETIC_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAJUlEQVR4nGNgYGCQk5OzsbGJioqqqKiYNm3ali1bLl26xDC0JABjG07Bu/EhPgAAAABJRU5ErkJggg==";

const checks = [];
const note = (name, status, detail) => checks.push({ name: name, status: status, detail: String(detail === undefined ? "" : detail) });
const ok = (name, cond, detail) => note(name, cond ? "PASS" : "FAIL", detail);
const info = (name, detail) => note(name, "INFO", detail);
const ev = (js) => { try { return page.evaluate(js); } catch (e) { return "__EVAL_ERR__:" + String(e); } };
const j = (js) => { const raw = ev(js); try { return JSON.parse(raw); } catch (e) { return { __raw: String(raw) }; } };
const waitFor = (script, ms) => { try { page.waitForScript({ script: script, timeout: ms || 20000 }); return true; } catch (e) { return false; } };

// --- 1. загрузчик на главной: зона, кнопка, input с capture ---
await page.goto(BASE + "/");
waitFor("document.readyState==='complete' && !!document.querySelector('input[type=file]')", 15000);
const up = j("JSON.stringify({" +
  "input:!!document.querySelector('input[type=file]')," +
  "accept:(document.querySelector('input[type=file]')||{}).accept||''," +
  "capture:(document.querySelector('input[type=file]')||{}).getAttribute&&document.querySelector('input[type=file]').getAttribute('capture')," +
  "dropzone:!!document.querySelector('.dropzone')," +
  "dropzoneRole:(document.querySelector('.dropzone')||{}).getAttribute&&document.querySelector('.dropzone').getAttribute('role')," +
  "dropzoneTabindex:(document.querySelector('.dropzone')||{}).getAttribute&&document.querySelector('.dropzone').getAttribute('tabindex')," +
  "button:(document.querySelector('.card button.primary')||{}).textContent||null," +
  "hint:!!document.querySelector('#uploader-hint')" +
  "})");
ok("uploader file input", up.input === true, JSON.stringify(up));
ok("uploader accept=image/*", up.accept === "image/*", "accept=" + up.accept);
ok("uploader capture=environment", up.capture === "environment", "capture=" + up.capture);
ok("uploader dropzone role=button", up.dropzone === true && up.dropzoneRole === "button", "role=" + up.dropzoneRole);
ok("uploader keyboard reachable", up.dropzoneTabindex === "0", "tabindex=" + up.dropzoneTabindex);
ok("uploader primary button", up.button === "Снять или выбрать фото", "button=" + up.button);

// --- 2. реальное фото этикетки: fetch -> File -> change -> карточка вина ---
const setFile = ev("(function(){window.__e2eFile='start';" +
  "fetch(" + JSON.stringify(FIXTURE) + ").then(function(r){return r.blob()}).then(function(b){" +
  "var f=new File([b],'label.webp',{type:'image/webp'});var dt=new DataTransfer();dt.items.add(f);" +
  "var inp=document.querySelector('input[type=file]');inp.files=dt.files;" +
  "inp.dispatchEvent(new Event('change',{bubbles:true}));window.__e2eFile='set:'+b.size;" +
  "}).catch(function(e){window.__e2eFile='ERR:'+e});return 'kicked'})()");
const fileSet = waitFor("window.__e2eFile!=='start'", 20000);
const fileStatus = ev("window.__e2eFile");
ok("file selected via input", fileSet && String(fileStatus).indexOf("set:") === 0, String(fileStatus));
ok("busy state shown", ev("(function(){var b=document.querySelector('.card button.primary');return b?b.textContent:null})()") === "Ищу вино…", "button label while busy");

const scanned = waitFor("!!document.querySelector('.wine-hero') || !!document.querySelector('.state-badge') || !!document.querySelector('.toast')", 90000);
const card = j("JSON.stringify({" +
  "wineHero:!!document.querySelector('.wine-hero')," +
  "title:(document.querySelector('#wine-title')||{}).textContent||null," +
  "winery:(document.querySelector('.wine-sub')||{}).textContent||null," +
  "blocks:Array.from(document.querySelectorAll('h3.block-title')).map(function(h){return h.textContent})," +
  "stateBadge:(document.querySelector('.state-badge')||{}).textContent||null," +
  "toast:(document.querySelector('.toast')||{}).textContent||null," +
  "sommelierPanel:!!Array.from(document.querySelectorAll('h2')).find(function(h){return h.textContent==='Цифровой сомелье'})," +
  "similar:!!document.querySelector('#similar-title')," +
  "similarItems:document.querySelectorAll('.similar').length," +
  "enotourism:!!document.querySelector('#enotourism-title')" +
  "})");
ok("scan produced a result", scanned && (card.wineHero || card.stateBadge), JSON.stringify(card));
ok("scan no error toast", card.toast === null, String(card.toast));
ok("wine card rendered", card.wineHero === true, "title=" + card.title + " winery=" + card.winery);
ok("card has grape/dish/region blocks", (card.blocks || []).length >= 2, JSON.stringify(card.blocks));
ok("sommelier panel on card", card.sommelierPanel === true, "Цифровой сомелье");
ok("similar wines block", card.similar === true, "items=" + card.similarItems);
ok("enotourism block", card.enotourism === true, "Энотуризм");
info("scan result", "title=" + card.title + " | winery=" + card.winery + " | badge=" + card.stateBadge);

// --- 3. «честный отказ»: синтетический 8x8 PNG -> not_found, но не белый экран ---
await page.goto(BASE + "/");
waitFor("document.readyState==='complete' && !!document.querySelector('input[type=file]')", 15000);
ev("(function(){window.__e2eFile2='start';" +
  "var b=atob(" + JSON.stringify(SYNTHETIC_B64) + ");var arr=new Uint8Array(b.length);" +
  "for(var i=0;i<b.length;i++)arr[i]=b.charCodeAt(i);" +
  "var f=new File([arr],'synthetic.png',{type:'image/png'});var dt=new DataTransfer();dt.items.add(f);" +
  "var inp=document.querySelector('input[type=file]');inp.files=dt.files;" +
  "inp.dispatchEvent(new Event('change',{bubbles:true}));window.__e2eFile2='set';return 1})()");
waitFor("window.__e2eFile2==='set'", 10000);
const nfShown = waitFor("!!document.querySelector('.state-badge') || !!document.querySelector('.wine-hero') || !!document.querySelector('.toast')", 90000);
const nf = j("JSON.stringify({" +
  "badge:(document.querySelector('.state-badge')||{}).textContent||null," +
  "headline:(document.querySelector('#nf-title')||{}).textContent||null," +
  "candidates:document.querySelectorAll('.candidate').length," +
  "candidateNames:Array.from(document.querySelectorAll('.candidate-name')).slice(0,3).map(function(n){return n.textContent})," +
  "tips:document.querySelectorAll('.tips .chip').length," +
  "portal:(document.querySelector('.actions a.primary')||{}).getAttribute&&document.querySelector('.actions a.primary').getAttribute('href')," +
  "toast:(document.querySelector('.toast')||{}).textContent||null," +
  "bodyLen:(document.body.innerText||'').trim().length" +
  "})");
ok("not-found state rendered", nfShown && nf.badge !== null, JSON.stringify(nf));
ok("not-found has headline", typeof nf.headline === "string" && nf.headline.length > 5, String(nf.headline));
ok("not-found offers candidates", nf.candidates > 0, "candidates=" + nf.candidates + " " + JSON.stringify(nf.candidateNames));
ok("not-found offers retake tips", nf.tips >= 3, "tips=" + nf.tips);
ok("not-found links to portal search", typeof nf.portal === "string" && nf.portal.indexOf("vino-svoe.ru") !== -1, String(nf.portal));
ok("not-found no white screen", nf.bodyLen > 200, "body chars=" + nf.bodyLen);
info("not-found state", "badge=" + nf.badge + " | headline=" + nf.headline);

// --- 4. /how: живой трейс по тому же фото ---
await page.goto(BASE + "/how");
waitFor("document.readyState==='complete' && !!document.querySelector('input[type=file]')", 15000);
const howStatic = j("JSON.stringify({" +
  "pipeline:document.querySelectorAll('.pipe li').length," +
  "methods:document.querySelectorAll('.method').length," +
  "limits:document.querySelectorAll('.limits li').length," +
  "uploader:!!document.querySelector('.dropzone')" +
  "})");
ok("/how static pipeline", howStatic.pipeline >= 5, "steps=" + howStatic.pipeline);
ok("/how methods showcase", howStatic.methods >= 5, "methods=" + howStatic.methods);
ok("/how honest limits", howStatic.limits >= 3, "limits=" + howStatic.limits);
ev("(function(){window.__e2eHow='start';" +
  "fetch(" + JSON.stringify(FIXTURE) + ").then(function(r){return r.blob()}).then(function(b){" +
  "var f=new File([b],'label.webp',{type:'image/webp'});var dt=new DataTransfer();dt.items.add(f);" +
  "var inp=document.querySelector('input[type=file]');inp.files=dt.files;" +
  "inp.dispatchEvent(new Event('change',{bubbles:true}));window.__e2eHow='set'}).catch(function(e){window.__e2eHow='ERR:'+e});return 1})()");
waitFor("window.__e2eHow==='set'", 20000);
const traceShown = waitFor("!!document.querySelector('.timeline li') || !!document.querySelector('.toast')", 90000);
const trace = j("JSON.stringify({" +
  "stages:document.querySelectorAll('.timeline li').length," +
  "stageNames:Array.from(document.querySelectorAll('.stage-name')).map(function(n){return n.textContent})," +
  "flags:document.querySelectorAll('.flag-list li').length," +
  "methods:document.querySelectorAll('.chips li.chip').length," +
  "candidates:document.querySelectorAll('.k-list li').length," +
  "decision:(document.querySelector('section.card p.muted.small')||{}).textContent||null," +
  "toast:(document.querySelector('.toast')||{}).textContent||null" +
  "})");
ok("/how live trace rendered", traceShown && trace.stages > 0, JSON.stringify(trace));
ok("/how trace has stages", trace.stages >= 2, JSON.stringify(trace.stageNames));
ok("/how trace has flags", trace.flags >= 5, "flags=" + trace.flags);
ok("/how trace has candidates", trace.candidates >= 1, "top_k=" + trace.candidates);
ok("/how no error toast", trace.toast === null, String(trace.toast));
info("/how trace", "stages=" + JSON.stringify(trace.stageNames) + " | decision=" + String(trace.decision).slice(0, 80));

const failed = checks.filter((c) => c.status === "FAIL");
return { script: "scanner", status: failed.length ? "FAIL" : "PASS", checks: checks };
