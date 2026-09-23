// e2e/smoke.js — поток «Навигация/оболочка» (+, десктопная оболочка и a11y).
// Запуск: lightpanda run apps/web/e2e/smoke.js   (нужен UI на 127.0.0.1:3000)
// Проверяет: рендер 5 маршрутов, заголовки документа, пункты меню (сайдбар + таб-бар),
// активное состояние пункта, переход по ссылке без перезагрузки (SPA), базовую доступность.
const page = new Page();
const BASE = "http://127.0.0.1:3000";

const checks = [];
const note = (name, status, detail) => checks.push({ name: name, status: status, detail: String(detail === undefined ? "" : detail) });
const ok = (name, cond, detail) => note(name, cond ? "PASS" : "FAIL", detail);
const info = (name, detail) => note(name, "INFO", detail);
const ev = (js) => { try { return page.evaluate(js); } catch (e) { return "__EVAL_ERR__:" + String(e); } };
const j = (js) => { const raw = ev(js); try { return JSON.parse(raw); } catch (e) { return { __raw: String(raw) }; } };
const waitFor = (script, ms) => { try { page.waitForScript({ script: script, timeout: ms || 15000 }); return true; } catch (e) { return false; } };

const ROUTES = [
  { path: "/", title: "Своё Вино — сканер этикеток", h1: "Своё Вино" },
  { path: "/sommelier", title: "Сомелье — Своё Вино", h1: "Сомелье" },
  { path: "/how", title: "Как это работает — Своё Вино", h1: "Как это работает" },
  { path: "/catalog", title: "Каталог — Своё Вино", h1: "Каталог вин" },
  { path: "/metrics", title: "Метрики — Своё Вино", h1: "Метрики качества" },
  { path: "/learning", title: "Обучение — Своё Вино", h1: "Обучение и качество" },
];
// C-055: навигация «5 основных + Ещё». В сайдбаре на десктопе видны все страницы,
// в таб-баре — пять основных и кнопка «Ещё» с остальными (там же «Как это работает»).
const EXPECTED_SIDEBAR = ["/", "/catalog", "/sommelier", "/metrics", "/learning", "/how"];
const EXPECTED_TABBAR = ["/", "/catalog", "/sommelier", "/metrics", "/learning"];
const MORE_PATHS = ["/how"];

// каждый маршрут: рендер, заголовок, меню и активный пункт
for (const route of ROUTES) {
  const rendered = (await page.goto(BASE + route.path)) !== undefined;
  waitFor("document.readyState==='complete' && !!document.querySelector('main h1')", 15000);
  const view = j("JSON.stringify({" +
    "title:document.title," +
    "h1:Array.from(document.querySelectorAll('main h1')).map(function(h){return h.textContent.trim()})," +
    "nav:Array.from(document.querySelectorAll('.shell-nav .nav-link')).map(function(a){return a.getAttribute('href')})," +
    "tabbar:Array.from(document.querySelectorAll('.tabbar .tabbar-link')).map(function(a){return a.getAttribute('href')})," +
    "navActive:Array.from(document.querySelectorAll('.shell-nav .nav-link.active')).map(function(a){return a.getAttribute('href')})," +
    "tabActive:Array.from(document.querySelectorAll('.tabbar .tabbar-link[href].active')).map(function(a){return a.getAttribute('href')})," +
    "tabLinks:Array.from(document.querySelectorAll('.tabbar .tabbar-link[href]')).map(function(a){return a.getAttribute('href')})," +
    "moreButton:!!document.querySelector('.tabbar button.tabbar-link')," +
    "moreExpanded:(document.querySelector('.tabbar button.tabbar-link')||{}).getAttribute" +
      "?document.querySelector('.tabbar button.tabbar-link').getAttribute('aria-expanded'):null," +
    "moreActive:!!document.querySelector('.tabbar button.tabbar-link.active')," +
    "toast:(document.querySelector('.toast')||{}).textContent||null," +
    "bodyLen:(document.body.innerText||'').trim().length" +
    "})");
  ok(route.path + " render", rendered && view.bodyLen > 40, "body chars=" + view.bodyLen);
  ok(route.path + " title", view.title === route.title, "got: " + view.title);
  ok(route.path + " page h1", (view.h1 || []).indexOf(route.h1) !== -1, JSON.stringify(view.h1));
  // Сайдбар показывает все страницы: ни одна не потеряна.
  ok(route.path + " nav items (all " + EXPECTED_SIDEBAR.length + ")",
     JSON.stringify(view.nav) === JSON.stringify(EXPECTED_SIDEBAR), JSON.stringify(view.nav));
  ok(route.path + " nav has /how", (view.nav || []).indexOf("/how") !== -1, JSON.stringify(view.nav));
  // Таб-бар: пять основных ссылок и отдельная кнопка «Ещё».
  ok(route.path + " tabbar items (5)",
     JSON.stringify(view.tabLinks) === JSON.stringify(EXPECTED_TABBAR), JSON.stringify(view.tabLinks));
  ok(route.path + " tabbar more button", view.moreButton === true && view.moreExpanded === "false",
     "button=" + view.moreButton + " aria-expanded=" + view.moreExpanded);
  ok(route.path + " sidebar active", (view.navActive || []).indexOf(route.path) !== -1, JSON.stringify(view.navActive));
  // Активный пункт подсвечен в обоих режимах: в таб-баре для «Ещё»-страниц активна кнопка.
  const tabbarActive = MORE_PATHS.indexOf(route.path) === -1
    ? (view.tabActive || []).indexOf(route.path) !== -1
    : view.moreActive === true;
  ok(route.path + " tabbar active", tabbarActive,
     MORE_PATHS.indexOf(route.path) === -1 ? JSON.stringify(view.tabActive)
       : "moreActive=" + view.moreActive);
  ok(route.path + " no error toast", view.toast === null, String(view.toast));
}

// переход по боковому меню без полной перезагрузки: адрес и активный пункт обновляются
await page.goto(BASE + "/");
waitFor("document.readyState==='complete' && !!document.querySelector('.shell-nav .nav-link')", 10000);
const clicked = ev("(function(){var a=Array.from(document.querySelectorAll('.shell-nav .nav-link')).find(function(x){return x.getAttribute('href')==='/catalog'}); if(!a) return 'missing'; a.click(); return 'ok';})()");
const spaNavigated = waitFor("location.pathname==='/catalog' && !!document.querySelector('main .card')", 15000);
const spa = j("JSON.stringify({clicked:" + JSON.stringify(clicked) + ",path:location.pathname,active:Array.from(document.querySelectorAll('.shell-nav .nav-link.active')).map(function(a){return a.getAttribute('href')}),title:document.title})");
ok("SPA nav to /catalog", spaNavigated && spa.path === "/catalog", JSON.stringify(spa));
ok("SPA nav active state", JSON.stringify(spa.active) === JSON.stringify(["/catalog"]), JSON.stringify(spa.active));

// доступность: lang, landmarks, alt, имена кнопок, поля с подписями, порядок заголовков
await page.goto(BASE + "/");
waitFor("document.readyState==='complete'", 10000);
const a = j("JSON.stringify({" +
  "lang:document.documentElement.lang||''," +
  "main:!!document.querySelector('main')," +
  "navLabels:Array.from(document.querySelectorAll('nav[aria-label]')).map(function(n){return n.getAttribute('aria-label')})," +
  "header:!!document.querySelector('header'),footer:!!document.querySelector('footer')," +
  "imgNoAlt:Array.from(document.querySelectorAll('img')).filter(function(i){return !i.hasAttribute('alt')}).length," +
  "btnNoName:Array.from(document.querySelectorAll('button')).filter(function(b){return !(b.textContent||'').trim() && !b.getAttribute('aria-label')}).length," +
  "controlsNoLabel:Array.from(document.querySelectorAll('input:not([type=hidden]),select')).filter(function(i){var id=i.id;return !(id&&document.querySelector('label[for=\"'+id+'\"]'))&&!i.getAttribute('aria-label')}).length," +
  "h1:document.querySelectorAll('h1').length," +
  "headingSkips:(function(){var hs=Array.from(document.querySelectorAll('h1,h2,h3,h4')).map(function(h){return +h.tagName[1]});var s=0;for(var i=1;i<hs.length;i++){if(hs[i]-hs[i-1]>1)s++}return s})()" +
  "})");
ok("a11y html[lang]", a.lang.length > 0, "lang=" + JSON.stringify(a.lang));
ok("a11y main landmark", a.main === true, "main=" + a.main);
ok("a11y nav landmarks", (a.navLabels || []).length >= 2, JSON.stringify(a.navLabels));
ok("a11y images have alt", a.imgNoAlt === 0, "without alt=" + a.imgNoAlt);
ok("a11y buttons have name", a.btnNoName === 0, "without name=" + a.btnNoName);
ok("a11y form controls labelled", a.controlsNoLabel === 0, "without label=" + a.controlsNoLabel);
ok("a11y single h1", a.h1 === 1, "h1 count=" + a.h1 + " (оболочка + заголовок страницы)");
ok("a11y heading order", a.headingSkips === 0, "skips=" + a.headingSkips);
info("a11y h1 texts", JSON.stringify(j("JSON.stringify(Array.from(document.querySelectorAll('h1')).map(function(h){return h.textContent.trim()}))")));

const failed = checks.filter((c) => c.status === "FAIL");
return { script: "smoke", status: failed.length ? "FAIL" : "PASS", checks: checks };
