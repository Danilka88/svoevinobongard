// e2e/theme.js — поток «Тема»: переключение light/dark, aria-label, сохранение в localStorage,
// сохранение при SPA-переходе и при полной перезагрузке.
// Запуск: lightpanda run apps/web/e2e/theme.js   (UI :3000)
const page = new Page();
const BASE = "http://127.0.0.1:3000";

const checks = [];
const note = (name, status, detail) => checks.push({ name: name, status: status, detail: String(detail === undefined ? "" : detail) });
const ok = (name, cond, detail) => note(name, cond ? "PASS" : "FAIL", detail);
const ev = (js) => { try { return page.evaluate(js); } catch (e) { return "__EVAL_ERR__:" + String(e); } };
const j = (js) => { const raw = ev(js); try { return JSON.parse(raw); } catch (e) { return { __raw: String(raw) }; } };
const waitFor = (script, ms) => { try { page.waitForScript({ script: script, timeout: ms || 10000 }); return true; } catch (e) { return false; } };
const state = () => j("JSON.stringify({" +
  "theme:document.documentElement.getAttribute('data-theme')," +
  "stored:(function(){try{return localStorage.getItem('svoe-vino-theme')}catch(e){return 'ERR'}})()," +
  "label:(document.querySelector('.theme-toggle span')||{}).textContent||null," +
  "aria:(document.querySelector('.theme-toggle')||{}).getAttribute&&document.querySelector('.theme-toggle').getAttribute('aria-label')," +
  "token:(function(){try{return getComputedStyle(document.documentElement).getPropertyValue('--accent').trim()}catch(e){return '?'}})()" +
  "})");

await page.goto(BASE + "/");
waitFor("document.readyState==='complete' && !!document.querySelector('.theme-toggle')", 15000);
const initial = state();
ok("theme attribute present", initial.theme === "light" || initial.theme === "dark", "theme=" + initial.theme);
ok("theme persisted on mount", initial.stored === initial.theme, "stored=" + initial.stored);
ok("theme toggle label", typeof initial.label === "string" && initial.label.length > 0, "label=" + initial.label);

// переключаем в тёмную
ev("(function(){document.querySelector('.theme-toggle').click();return 1})()");
const toDark = waitFor("document.documentElement.getAttribute('data-theme')!=='" + initial.theme + "'", 8000);
const dark = state();
ok("toggle switches theme", toDark && dark.theme !== initial.theme, initial.theme + " -> " + dark.theme);
ok("theme written to localStorage", dark.stored === dark.theme, "stored=" + dark.stored);
ok("theme aria-label updated", dark.aria !== initial.aria && /светл/i.test(dark.aria || ""), "aria=" + dark.aria);
ok("theme accent token changed", dark.token !== initial.token, initial.token + " -> " + dark.token);

// SPA-переход не сбрасывает тему
ev("(function(){var a=Array.from(document.querySelectorAll('.nav-link')).find(function(x){return x.getAttribute('href')==='/catalog'});if(a)a.click();return 1})()");
waitFor("location.pathname==='/catalog'", 12000);
const afterNav = state();
ok("theme survives SPA navigation", afterNav.theme === dark.theme, "theme=" + afterNav.theme);

// полная перезагрузка
await page.goto(BASE + "/metrics");
waitFor("document.readyState==='complete' && !!document.querySelector('.theme-toggle')", 15000);
const afterReload = state();
ok("theme survives full reload", afterReload.theme === dark.theme, "theme=" + afterReload.theme + " stored=" + afterReload.stored);

// переключаем обратно
ev("(function(){document.querySelector('.theme-toggle').click();return 1})()");
const back = waitFor("document.documentElement.getAttribute('data-theme')==='" + initial.theme + "'", 8000);
const restored = state();
ok("toggle switches back", back && restored.theme === initial.theme, "theme=" + restored.theme);
ok("back-toggle persisted", restored.stored === restored.theme, "stored=" + restored.stored);

// чистим, чтобы не влиять на другие прогоны
ev("(function(){try{localStorage.removeItem('svoe-vino-theme')}catch(e){}return 1})()");

const failed = checks.filter((c) => c.status === "FAIL");
return { script: "theme", status: failed.length ? "FAIL" : "PASS", checks: checks };
