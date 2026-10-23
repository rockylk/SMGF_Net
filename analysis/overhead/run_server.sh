#!/usr/bin/env bash
# Section 5.6 V17 native GPU server one-command launcher
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export S56_CODE_ROOT="${S56_CODE_ROOT:-/home/admin1/lkfiles}"
export S56_RESULT_ROOT="${S56_RESULT_ROOT:-/home/admin1/lkfiles/RESULT}"
export S56_PCAP_ROOT="${S56_PCAP_ROOT:-/home/admin1/dataset/aes-128-gcm/aes-128-gcm}"
export MKL_THREADING_LAYER=GNU
unset MKL_SERVICE_FORCE_INTEL || true
PYTHON_BIN="${S56_PYTHON:-python}"
mkdir -p "$HERE/results/logs" "$HERE/results/report"
LOG="$HERE/results/logs/oneclick.log"
PIDFILE="$HERE/results/logs/oneclick.pid"
if [[ -f "$PIDFILE" ]]; then
  LAST_PID="$(cat "$PIDFILE")"
  if [[ "$LAST_PID" =~ ^[0-9]+$ ]] && kill -0 "$LAST_PID" 2>/dev/null; then
    echo "[Section56] already running PID=$LAST_PID"
    echo "LOG=$LOG"
    exit 0
  fi
fi
command -v "$PYTHON_BIN" >/dev/null || { echo "ERROR: python executable unavailable ($PYTHON_BIN)" >&2; exit 2; }
echo "[Section56] Model-specific sibling-folder source/checkpoint preflight:"
"$PYTHON_BIN" "$HERE/server_layout_check.py" || exit 2
echo "[Section56] Existing previous-run measurements (read-only):"
"$PYTHON_BIN" "$HERE/diagnose_previous.py" --root "$HERE" --output "$HERE/results/previous_run_diagnosis.json" || true
echo "STARTED_AT=$(date -Is)" >> "$LOG"
echo "CODE=$S56_CODE_ROOT RESULT=$S56_RESULT_ROOT" >> "$LOG"
nohup "$PYTHON_BIN" -u "$HERE/server_oneclick.py" \
   --code-root "$S56_CODE_ROOT" --result-root "$S56_RESULT_ROOT" \
   >> "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PIDFILE"
printf '\n[Section56] server-only GPU experiment launched PID=%s\n' "$PID"
printf 'LOG: %s\n' "$LOG"
printf 'STATUS: %s\n' "$HERE/results/run_status.json"
printf 'TABLE: %s\n' "$HERE/results/report/TableVIII_V17_status.csv"
printf 'FIGURE: %s\n' "$HERE/results/report/Fig7_V17_latency_robustness.pdf"
printf '\nWatch: tail -f "%s"\n' "$LOG"
printf 'Check: bash "%s/show_results.sh"\n\n' "$HERE"
