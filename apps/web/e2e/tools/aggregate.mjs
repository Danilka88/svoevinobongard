#!/usr/bin/env node
/**
 * aggregate.mjs — собирает apps/web/e2e/results/*.json в единый JSON-отчёт.
 *
 * Использование (из корня svoe-vino-scanner):
 *   node apps/web/e2e/tools/aggregate.mjs [reports/e2e_ui_report.json]
 *
 * Пишет только машинную часть (статусы, проверки, консоль). Человекочитаемая
 * часть отчёта (баги, ограничения) добавляется агентом поверх.
 */
import { readFileSync, writeFileSync, readdirSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';

const root = process.cwd();
const resultsDir = join(root, 'apps/web/e2e/results');
const outPath = process.argv[2] || join(root, 'reports/e2e_ui_report.json');

const readJson = (p) => {
  try { return JSON.parse(readFileSync(p, 'utf8')); } catch { return null; }
};

const flows = [];
for (const file of existsSync(resultsDir) ? readdirSync(resultsDir).sort() : []) {
  if (!file.endsWith('.json')) continue;
  if (['console.json', 'analysis.json'].includes(file)) continue; // не кейсы
  const data = readJson(join(resultsDir, file));
  if (!data || !Array.isArray(data.checks)) continue;
  const checks = data.checks;
  flows.push({
    flow: data.script || file.replace(/\.json$/, ''),
    status: data.status || 'ERROR',
    checks: checks.length,
    pass: checks.filter((c) => c.status === 'PASS').length,
    fail: checks.filter((c) => c.status === 'FAIL').length,
    info: checks.filter((c) => c.status === 'INFO').length,
    failures: checks.filter((c) => c.status === 'FAIL').map((c) => ({ name: c.name, detail: c.detail })),
    details: checks,
  });
}

const consoleCapture = readJson(join(resultsDir, 'console.json'));
const analysis = readJson(join(resultsDir, 'analysis.json')) || {};
const report = {
  generated_at: new Date().toISOString(),
  ui_base: process.env.UI_URL || 'http://127.0.0.1:3000',
  lightpanda: {
    cli: 'lightpanda run (PandaScript)',
    viewport: '1920x1080 (фиксированный, мобильный вьюпорт недоступен)',
  },
  totals: {
    flows: flows.length,
    pass: flows.filter((f) => f.status === 'PASS').length,
    fail: flows.filter((f) => f.status === 'FAIL').length,
    checks: flows.reduce((n, f) => n + f.checks, 0),
    failed_checks: flows.reduce((n, f) => n + f.fail, 0),
  },
  /** Человекочитаемая аналитика (bugs/limitations/coverage) — из results/analysis.json. */
  ...analysis,
  flows,
  console: consoleCapture
    ? {
        routes: Object.keys(consoleCapture.routes || {}),
        console_lines: consoleCapture.summary?.console ?? 0,
        js_errors: consoleCapture.summary?.errors ?? 0,
        details: consoleCapture.summary?.errorLines ?? [],
      }
    : { note: 'сбор консоли не выполнялся (lightpanda serve не запущен)' },
};

writeFileSync(outPath, JSON.stringify(report, null, 2) + '\n');
console.log('wrote ' + outPath);
console.log(`flows=${report.totals.flows} pass=${report.totals.pass} fail=${report.totals.fail} checks=${report.totals.checks} failed=${report.totals.failed_checks}`);
