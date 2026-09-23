#!/usr/bin/env bash
# e2e/run_all.sh — прогон всех PandaScript-кейсов UI и сбор результатов.
#
# Использование (из корня svoe-vino-scanner):
#   bash apps/web/e2e/run_all.sh
#
# Предусловия (поднимаются отдельно, скрипт их только проверяет):
#   UI       http://127.0.0.1:3000        (node .output/server/index.mjs)
#   gateway  http://127.0.0.1:8080
#   фикстуры http://127.0.0.1:8099        (python3 -m http.server 8099 --directory data/samples/real100)
#   CDP      ws://127.0.0.1:9222          (lightpanda serve --port 9222) — для сбора консоли
#
# Результаты: apps/web/e2e/results/<script>.json (по одному на кейс).
# Примечание: catalog-search.js идёт последним: исторически он вызывал нагрузку на gateway
# (BUG-001, поиск ускорен в C-046/C-047 — кейс проходит). После него gateway перезапускается,
# если задан GW_RESTART_CMD.
set -u
export PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
E2E="$ROOT/apps/web/e2e"
OUT="$E2E/results"
mkdir -p "$OUT"

UI_URL="${UI_URL:-http://127.0.0.1:3000}"
FIXTURES_URL="${FIXTURES_URL:-http://127.0.0.1:8099}"
CDP_PORT="${CDP_PORT:-9222}"

echo "== prerequisites =="
curl -s -o /dev/null -m 5 "$UI_URL/" && echo "  UI        ok" || { echo "  UI        НЕДОСТУПЕН ($UI_URL)"; exit 1; }
curl -s -o /dev/null -m 5 "http://127.0.0.1:8080/v1/health" && echo "  gateway   ok" || echo "  gateway   недоступен (error-state.js ожидаемо упадёт)"
curl -s -o /dev/null -m 5 "$FIXTURES_URL/96.13_19-08-2026_19-33-10.webp" && echo "  fixtures  ok" || echo "  fixtures  недоступны (scanner.js не сможет загрузить фото)"
curl -s -o /dev/null -m 5 "http://127.0.0.1:$CDP_PORT/json/version" && echo "  lightpanda cdp ok" || echo "  lightpanda cdp недоступен (сбор консоли пропущен)"

run_case () {
  local name="$1"; shift
  echo "== $name =="
  lightpanda run "$E2E/$name.js" "$@" 2>/dev/null | grep -v '^\$time' > "$OUT/$name.json.raw"
  if [ -s "$OUT/$name.json.raw" ]; then
    mv "$OUT/$name.json.raw" "$OUT/$name.json"
    node -e "
      const fs=require('fs');
      const j=JSON.parse(fs.readFileSync('$OUT/$name.json','utf8'));
      const bad=j.checks.filter(c=>c.status==='FAIL');
      console.log('  '+j.status+'  checks='+j.checks.length+' failed='+bad.length);
      for (const c of bad) console.log('    FAIL '+c.name+' | '+String(c.detail).slice(0,110));
    "
  else
    echo "  НЕТ ВЫВОДА (скрипт не выполнился)"
    echo "{\"script\":\"$name\",\"status\":\"ERROR\",\"checks\":[],\"error\":\"no output\"}" > "$OUT/$name.json"
  fi
}

run_case smoke
run_case theme
run_case fixes
run_case how-metrics
run_case catalog
run_case sommelier
run_case scanner
run_case images
run_case learning

# сбор консоли/исключений по маршрутам (CDP)
if curl -s -o /dev/null -m 5 "http://127.0.0.1:$CDP_PORT/json/version"; then
  echo "== console capture (cdp) =="
  node "$E2E/tools/cdp_console.mjs" "$UI_URL" / /sommelier /how /catalog /metrics > "$OUT/console.json" 2>"$OUT/console.err" \
    && node -e "
      const j=require('$OUT/console.json');
      console.log('  console lines='+j.summary.console+' errors='+j.summary.errors);
    " || echo "  сбор консоли не удался (см. console.err)"
fi

# ожидаемая регрессия поиска — последней
run_case catalog-search

if [ -n "${GW_RESTART_CMD:-}" ]; then
  echo "== restart gateway after catalog-search (BUG-001) =="
  eval "$GW_RESTART_CMD" && echo "  gateway restarted"
fi

echo "== done: $OUT =="
