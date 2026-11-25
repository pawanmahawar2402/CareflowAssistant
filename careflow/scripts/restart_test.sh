#!/bin/bash
# Kill-and-restart test for M3.C5 (memory survives a restart) and M5.C4/C5 (paused run resumes after a kill).
# Usage:  PYTHON=/path/to/python scripts/restart_test.sh     (defaults to `python`)
# Output: eval_results/restart_test.log
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python}"
LOG="$ROOT/eval_results/restart_test.log"
mkdir -p "$ROOT/eval_results"
cd "$ROOT"
U=pt-restart-01
B=http://127.0.0.1:8011
start(){ PYTHONPATH="$ROOT" "$PY" scripts/serve_stub.py >> "$ROOT/runtime/restart_server.log" 2>&1 & echo $! > "$ROOT/runtime/server.pid"
         for i in $(seq 1 60); do curl -s "$B/health" >/dev/null && break; sleep 1; done; }
kill_hard(){ kill -9 "$(cat "$ROOT/runtime/server.pid")"; sleep 1; }
post(){ curl -s -XPOST "$B/chat" -H 'content-type: application/json' -d "$1"; echo; }
mkdir -p "$ROOT/runtime"; : > "$LOG"; : > "$ROOT/runtime/restart_server.log"
{
echo "== run 1: fresh server"; start; curl -s "$B/health"; echo
echo "-- referral status for patient $U (episode is written)"; post "{\"message\":\"Where is my referral REF-0003?\",\"session_id\":\"s-a\",\"user_id\":\"$U\"}"
echo "-- emergency pauses for staff"; post '{"message":"I have crushing chest pain","session_id":"hitl-1"}'
echo "== KILL -9 the server"; kill_hard
echo "== run 2: new process, same data stores"; start
echo "-- new session, same patient: what did we do last time?"; post "{\"message\":\"What did we do last time?\",\"session_id\":\"s-b\",\"user_id\":\"$U\"}"
echo "-- staff approves the case paused in run 1"; curl -s -XPOST "$B/review" -H 'content-type: application/json' -d '{"session_id":"hitl-1","approved":true,"note":"confirmed"}'; echo
echo "-- approve again: must return the same ticket"; curl -s -XPOST "$B/review" -H 'content-type: application/json' -d '{"session_id":"hitl-1","approved":true,"note":"confirmed"}'; echo
echo "-- privacy delete, then recall"; curl -s -XDELETE "$B/users/$U/memory"; echo
post "{\"message\":\"What did we do last time?\",\"session_id\":\"s-c\",\"user_id\":\"$U\"}"
kill_hard
echo "== done"
} >> "$LOG" 2>&1
echo "wrote $LOG"
