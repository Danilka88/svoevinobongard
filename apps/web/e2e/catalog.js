// e2e/catalog.js — поток «Каталог» (/catalog): фасетные фильтры, сброс, «показать ещё»,
// карточка вина в нижней шторке (открытие/закрытие по кнопке и по фону), пустое состояние.
// Текстовый поиск (поле «Поиск») вынесен в отдельный e2e/catalog-search.js — он известен как сломанный.
// Запуск: lightpanda run apps/web/e2e/catalog.js   (UI :3000 + gateway :8080)
const page = new Page();
const BASE = "http://127.0.0.1:3000";

const checks = [];
const note = (name, status, detail) => checks.push({ name: name, status: status, detail: String(detail === undefined ? "" : detail) });
const ok = (name, cond, detail) => note(name, cond ? "PASS" : "FAIL", detail);
const info = (name, detail) => note(name, "INFO", detail);
const ev = (js) => { try { return page.evaluate(js); } catch (e) { return "__EVAL_ERR__:" + String(e); } };
const j = (js) => { const raw = ev(js); try { return JSON.parse(raw); } catch (e) { return { __raw: String(raw) }; } };
const waitFor = (script, ms) => { try { page.waitForScript({ script: script, timeout: ms || 20000 }); return true; } catch (e) { return false; } };
const snap = () => j("JSON.stringify({total:(document.querySelector('.badge')||{}).textContent||null,tiles:document.querySelectorAll('.wine-tile').length,hasReset:!!Array.from(document.querySelectorAll('button')).find(function(x){return x.textContent.trim()==='Сбросить'}),more:(function(){var b=Array.from(document.querySelectorAll('button')).find(function(x){return /Показать ещё/.test(x.textContent)});return b?b.textContent.trim():null})(),first:(document.querySelector('.tile-title')||{}).textContent||null,placeholder:(document.querySelector('.placeholder')||{}).textContent||null,toast:(document.querySelector('.toast')||{}).textContent||null})");
const waitTotalChange = (prev) => waitFor("((document.querySelector('.badge')||{}).textContent||'')!==" + JSON.stringify(prev), 25000);

await page.goto(BASE + "/catalog");
waitFor("document.readyState==='complete' && document.querySelectorAll('.wine-tile').length>0", 25000);

// --- витрина ---
const initial = snap();
ok("catalog showcase renders", initial.tiles === 24, "tiles=" + initial.tiles + " total=" + initial.total);
ok("catalog total reported", initial.total === "Найдено: 2103", String(initial.total));
ok("catalog load-more offered", /Показать ещё/.test(initial.more || ""), String(initial.more));
ok("catalog no error toast", initial.toast === null, String(initial.toast));

// --- фильтры ---
const facets = j("JSON.stringify({region:Array.from(document.querySelectorAll('#f-region option')).map(function(o){return o.value}),category:Array.from(document.querySelectorAll('#f-cat option')).map(function(o){return o.value}),grapeCount:document.querySelectorAll('#f-grape option').length,wineryCount:document.querySelectorAll('#f-winery option').length})");
ok("catalog region facets", (facets.region || []).length >= 5, JSON.stringify(facets.region));
ok("catalog category facets", (facets.category || []).length >= 4, JSON.stringify(facets.category));
ok("catalog grape/winery facets", facets.grapeCount > 10 && facets.wineryCount > 10, "grape=" + facets.grapeCount + " winery=" + facets.wineryCount);

let pivot = initial.total;
try { page.selectOption({ selector: "#f-region", value: "Крым" }); } catch (e) { note("region select", "FAIL", String(e)); }
const regionChanged = waitTotalChange(pivot);
const region = snap();
ok("region filter applies", regionChanged && region.total !== pivot, pivot + " -> " + region.total + " tiles=" + region.tiles);

pivot = region.total;
try { page.selectOption({ selector: "#f-cat", value: "Красное" }); } catch (e) { note("category select", "FAIL", String(e)); }
const catChanged = waitTotalChange(pivot);
const combined = snap();
ok("category filter applies", catChanged, pivot + " -> " + combined.total);
ok("filters combine (AND)", combined.total && combined.total !== "Найдено: 2103", "total=" + combined.total);

const resetState = ev("(function(){var b=Array.from(document.querySelectorAll('button')).find(function(x){return x.textContent.trim()==='Сбросить'});return b?'shown':'missing'})()");
ok("reset button appears with filters", resetState === "shown", String(resetState));

// --- пустое состояние: последовательно сужаем фильтры ---
// Сначала сбрасываем, иначе значения селектов остаются от предыдущих шагов.
ev("(function(){var b=Array.from(document.querySelectorAll('button')).find(function(x){return x.textContent.trim()==='Сбросить'});if(b)b.click();return 1})()");
waitFor("((document.querySelector('.badge')||{}).textContent||'')==='Найдено: 2103'", 25000);
try { page.selectOption({ selector: "#f-cat", value: "Красное" }); } catch (e) {}
waitFor("((document.querySelector('.badge')||{}).textContent||'')==='Найдено: 795'", 25000);
try { page.selectOption({ selector: "#f-region", value: "Дальневосточная зона" }); } catch (e) {}
const narrowed = waitFor("((document.querySelector('.badge')||{}).textContent||'')==='Найдено: 1'", 25000);
const empty = snap();
ok("filters narrow to a single item", narrowed, "total=" + String(ev("(document.querySelector('.badge')||{}).textContent")));
ok("narrowed result has no error toast", empty.toast === null, String(empty.toast));
// Пустое состояние через фасеты недостижимо: опции пересчитываются по отфильтрованному набору,
// поэтому у каждой опции count>=1 и любая последовательность выборов даёт >=1 результат.
// Единственный путь к нулю — поле поиска, а оно сломано (см. e2e/catalog-search.js, BUG-001).
info("empty-state reachability", "placeholder 'Ничего не найдено' недостижим через фасетные фильтры; доступен только через сломанный поиск (BUG-001)");
const stale = j("JSON.stringify({grapeValue:document.getElementById('f-grape').value,grapeListed:Array.from(document.querySelectorAll('#f-grape option')).some(function(o){return o.value===document.getElementById('f-grape').value}),regionOptions:Array.from(document.querySelectorAll('#f-region option')).map(function(o){return o.value})})");
info("facet recompute (UX)", "после сужения опции пересчитаны: regions=" + JSON.stringify(stale.regionOptions) + " grape-facet=" + JSON.stringify(stale.grapeValue));

// --- сброс ---
const prevTotal = empty.total;
ev("(function(){var b=Array.from(document.querySelectorAll('button')).find(function(x){return x.textContent.trim()==='Сбросить'});if(b)b.click();return 1})()");
const resetBack = waitFor("((document.querySelector('.badge')||{}).textContent||'')==='Найдено: 2103' && document.querySelectorAll('.wine-tile').length>0", 25000);
ok("reset restores full catalog", resetBack, JSON.stringify(snap()));

// --- «показать ещё» ---
ev("(function(){var b=Array.from(document.querySelectorAll('button')).find(function(x){return /Показать ещё/.test(x.textContent)});if(b)b.click();return 1})()");
const grew = waitFor("document.querySelectorAll('.wine-tile').length>24", 25000);
const more = snap();
ok("load more appends page 2", grew && more.tiles === 48, "tiles=" + more.tiles);
ok("load-more counter decreases", /Показать ещё \(20/.test(more.more || ""), String(more.more));

// --- карточка в шторке ---
ev("(function(){var b=document.querySelector('.wine-tile .tile-hit');if(b)b.click();return 1})()");
const sheetOpen = waitFor("!!document.querySelector('.sheet-backdrop') && (!!document.querySelector('.wine-title') || !!document.querySelector('.sheet .placeholder'))", 25000);
const sheet = j("JSON.stringify({" +
  "open:!!document.querySelector('.sheet-backdrop')," +
  "role:(document.querySelector('.sheet')||{}).getAttribute&&document.querySelector('.sheet').getAttribute('role')," +
  "ariaModal:(document.querySelector('.sheet')||{}).getAttribute&&document.querySelector('.sheet').getAttribute('aria-modal')," +
  "title:(document.querySelector('.wine-title')||{}).textContent||null," +
  "blocks:Array.from(document.querySelectorAll('.sheet h3.block-title')).map(function(h){return h.textContent})," +
  "chips:document.querySelectorAll('.sheet ul.chips li.chip').length," +
  "close:!!Array.from(document.querySelectorAll('.sheet button')).find(function(b){return b.textContent.trim()==='Закрыть'})," +
  "sommelierHref:(document.querySelector('.sheet a.primary')||{}).getAttribute&&document.querySelector('.sheet a.primary').getAttribute('href')," +
  "wineryHref:(document.querySelector('.sheet a.secondary')||{}).getAttribute&&document.querySelector('.sheet a.secondary').getAttribute('href')" +
  "})");
ok("card sheet opens", sheetOpen && sheet.open === true, JSON.stringify(sheet).slice(0, 180));
ok("sheet is modal dialog", sheet.role === "dialog" && sheet.ariaModal === "true", "role=" + sheet.role + " aria-modal=" + sheet.ariaModal);
ok("sheet shows wine details", typeof sheet.title === "string" && sheet.title.length > 1, "title=" + sheet.title);
ok("sheet blocks (grapes/dishes/region)", (sheet.blocks || []).length >= 1, JSON.stringify(sheet.blocks));
ok("sheet links to sommelier with slug", /^\/sommelier\?slug=/.test(sheet.sommelierHref || ""), String(sheet.sommelierHref));

// закрытие кнопкой
ev("(function(){var b=Array.from(document.querySelectorAll('.sheet button')).find(function(x){return x.textContent.trim()==='Закрыть'});if(b)b.click();return 1})()");
ok("sheet closes via button", waitFor("!document.querySelector('.sheet-backdrop')", 8000), "open=" + ev("!!document.querySelector('.sheet-backdrop')"));

// закрытие кликом по фону
ev("(function(){var b=document.querySelector('.wine-tile .tile-hit');if(b)b.click();return 1})()");
waitFor("!!document.querySelector('.sheet-backdrop .sheet')", 20000);
ev("(function(){var b=document.querySelector('.sheet-backdrop');if(b)b.click();return 1})()");
ok("sheet closes via backdrop", waitFor("!document.querySelector('.sheet-backdrop')", 8000), "open=" + ev("!!document.querySelector('.sheet-backdrop')"));

info("catalog totals seen", "initial=" + initial.total + " region=Крым=" + region.total + " +Красное=" + combined.total + " empty=" + empty.total);
const failed = checks.filter((c) => c.status === "FAIL");
return { script: "catalog", status: failed.length ? "FAIL" : "PASS", checks: checks };
