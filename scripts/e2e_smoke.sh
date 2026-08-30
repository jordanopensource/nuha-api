#!/usr/bin/env bash
# End-to-end smoke test for the Nuha 2.0 stack (one api service + the models
# volume), driven through the api's published port and the fetch service.
#
# Proves the frozen public contract AND the runtime add/remove story:
# brings the api up with the dev override, installs every dialect but the last
# (sorted) via the fetch service, restarts, and runs the per-dialect contract
# checks; then asserts the missing dialect 400s, fetch-adds it, restarts,
# asserts 200; fetch-removes it, restarts, asserts 400; and finally adds it
# back so the stack is left complete. No rebuild anywhere: the image never
# changes, only the volume does.
#
# It asserts on STATUS CODES (the frozen contract), not on model output, so it
# is deterministic regardless of what the models predict. Needs curl + python3
# (to read the dialect files). Usage:
#   scripts/e2e_smoke.sh            # build + up + test + leave running
#   KEEP=0 scripts/e2e_smoke.sh     # tear the stack down at the end
#   BASE=http://host:8000 NOUP=1 scripts/e2e_smoke.sh   # contract checks only,
#                                   # against an already-running complete stack
set -uo pipefail

cd "$(dirname "$0")/.."

BASE="${BASE:-http://localhost:8000}"
COMPOSE=(docker compose -f compose.yml -f compose.dev.yml)
FAILED=0

dialects() { ls app/dialects/*.json | sed 's#.*/##; s#\.json$##'; }

# First declared language of a dialect (its alias if one exists, else the
# canonical code), read from the dialect file so no language is hardcoded.
lang_for() {
  python3 -c "
import json, sys
langs = json.load(open('app/dialects/' + sys.argv[1] + '.json'))['languages']
code = sorted(langs)[0]
print((langs[code].get('aliases') or [code])[0])" "$1"
}

# check <label> <expected-status> <curl-args...>
check() {
  local label="$1" want="$2"; shift 2
  local got
  got="$(curl -sS -o /dev/null -w '%{http_code}' "$@" 2>/dev/null)"
  if [ "$got" = "$want" ]; then
    printf '  ok   %-48s %s\n' "$label" "$got"
  else
    printf '  FAIL %-48s want=%s got=%s\n' "$label" "$want" "$got"
    FAILED=1
  fi
}

# The volume's installed set, via the fetch service (the volume's only reader
# outside the api).
installed() {
  "${COMPOSE[@]}" run --rm fetch list 2>/dev/null | sed -n 's/^installed[^:]*: //p' | tr -d ','
}

fetch_cmd() { "${COMPOSE[@]}" run --rm fetch "$@" >/dev/null 2>&1; }

# Restart the api (a fresh startup scan) and wait until /ready answers 200.
restart_and_wait() {
  "${COMPOSE[@]}" restart api >/dev/null 2>&1 || { echo "  restart failed"; FAILED=1; return; }
  for i in $(seq 1 120); do
    if "${COMPOSE[@]}" exec -T api \
        python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/ready',timeout=3).status==200 else 1)" \
        >/dev/null 2>&1; then
      echo "  api ready (${i}s)"; return
    fi
    [ "$i" = 120 ] && { echo "  api NOT ready after 120s"; FAILED=1; }
    sleep 1
  done
}

# One full per-dialect contract block.
contract_for() {
  local d="$1" lang
  echo "-- $d --"
  lang="$(lang_for "$d")"
  check "$d single"            200 -X POST "$BASE/$d/classify"       -H 'Content-Type: application/json' -d '{"text":"مرحبا كيف حالك"}'
  check "$d declared lang"     200 -X POST "$BASE/$d/classify"       -H 'Content-Type: application/json' -d "{\"text\":\"مرحبا\",\"lang\":\"$lang\"}"
  check "$d single bad lang"   422 -X POST "$BASE/$d/classify"       -H 'Content-Type: application/json' -d '{"text":"مرحبا","lang":"zzz"}'
  check "$d empty text"        422 -X POST "$BASE/$d/classify"       -H 'Content-Type: application/json' -d '{"text":""}'
  check "$d batch"             200 -X POST "$BASE/$d/classify/batch" -H 'Content-Type: application/json' -d '{"texts":["مرحبا","اهلا"]}'
  check "$d batch empty elem"  200 -X POST "$BASE/$d/classify/batch" -H 'Content-Type: application/json' -d '{"texts":["مرحبا",""]}'
  check "$d batch empty list"  422 -X POST "$BASE/$d/classify/batch" -H 'Content-Type: application/json' -d '{"texts":[]}'
}

LAST="$(dialects | sort | tail -n1)"

if [ "${NOUP:-0}" != "1" ]; then
  echo "== building + starting the api (dev override) =="
  "${COMPOSE[@]}" up -d --build api || { echo "compose up failed"; exit 1; }

  echo "== installing every dialect but '$LAST' =="
  have="$(installed)"
  for d in $(dialects); do
    [ "$d" = "$LAST" ] && continue
    case " $have " in
      *" $d "*) echo "  $d already installed" ;;
      *) echo "  fetching $d ..."; fetch_cmd add "$d" || { echo "  fetch $d FAILED"; FAILED=1; } ;;
    esac
  done
  case " $have " in
    *" $LAST "*) echo "  removing leftover '$LAST' from an earlier run"; fetch_cmd remove "$LAST" ;;
  esac

  echo "== restarting the api to pick the volume up =="
  restart_and_wait
fi

echo "== public contract per installed dialect =="
for d in $(dialects); do
  if [ "${NOUP:-0}" != "1" ] && [ "$d" = "$LAST" ]; then continue; fi
  contract_for "$d"
done

echo "== edge cases =="
check "invalid dialect -> 400" 400 -X POST "$BASE/nope/classify"     -H 'Content-Type: application/json' -d '{"text":"x"}'
check "bare /classify -> 404"  404 -X POST "$BASE/classify"          -H 'Content-Type: application/json' -d '{"text":"x"}'
# 11 MiB body, over the default 10 MiB MAX_BODY_SIZE cap. From a file so curl
# sends a Content-Length and the api refuses it up front, unread.
oversize="$(mktemp)"
head -c $((11 * 1024 * 1024)) /dev/zero | tr '\0' 'a' > "$oversize"
check "oversize body -> 413"   413 -X POST "$BASE/$(dialects | head -n1)/classify" -H 'Content-Type: application/json' --data-binary "@$oversize"
rm -f "$oversize"
check "health"                 200 "$BASE/health"

echo "== invalid-dialect body =="
body="$(curl -sS -X POST "$BASE/nope/classify" -H 'Content-Type: application/json' -d '{"text":"x"}')"
echo "  body: $body"
case "$body" in
  *'Invalid dialect'*) echo "  ok   invalid-dialect body shape" ;;
  *) echo "  FAIL invalid-dialect body shape"; FAILED=1 ;;
esac

if [ "${NOUP:-0}" != "1" ]; then
  echo "== runtime add/remove story ('$LAST') =="
  check "$LAST before fetch -> 400" 400 -X POST "$BASE/$LAST/classify" -H 'Content-Type: application/json' -d '{"text":"مرحبا"}'

  echo "  fetching $LAST ..."
  fetch_cmd add "$LAST" || { echo "  fetch $LAST FAILED"; FAILED=1; }
  restart_and_wait
  check "$LAST after fetch -> 200"  200 -X POST "$BASE/$LAST/classify" -H 'Content-Type: application/json' -d '{"text":"مرحبا كيف حالك"}'
  contract_for "$LAST"

  echo "  removing $LAST ..."
  fetch_cmd remove "$LAST" || { echo "  remove $LAST FAILED"; FAILED=1; }
  restart_and_wait
  check "$LAST after remove -> 400" 400 -X POST "$BASE/$LAST/classify" -H 'Content-Type: application/json' -d '{"text":"مرحبا"}'

  echo "  adding $LAST back (leave the stack complete) ..."
  fetch_cmd add "$LAST" || { echo "  re-fetch $LAST FAILED"; FAILED=1; }
  restart_and_wait
  check "$LAST restored -> 200"     200 -X POST "$BASE/$LAST/classify" -H 'Content-Type: application/json' -d '{"text":"مرحبا كيف حالك"}'
fi

if [ "${KEEP:-1}" = "0" ] && [ "${NOUP:-0}" != "1" ]; then
  echo "== tearing down =="
  "${COMPOSE[@]}" down
fi

if [ "$FAILED" = "0" ]; then echo "E2E SMOKE: PASS"; else echo "E2E SMOKE: FAIL"; fi
exit "$FAILED"
