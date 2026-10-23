#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "=== COUNTS OF ALREADY COMPLETED GPU PROFILES ==="
python "$HERE/diagnose_previous.py" --root "$HERE" --output "$HERE/results/previous_run_diagnosis.json" || true
echo "=== Section 5.6 server run ==="
echo "PACKAGE: $HERE"
echo "=== Main log (latest 45 lines) ==="
if [[ -f "$HERE/results/logs/oneclick.log" ]]; then
 tail -n 45 "$HERE/results/logs/oneclick.log"
else
 echo "NOT STARTED: bash run_server.sh"
fi
echo "=== STATUS ==="
if [[ -f "$HERE/results/run_status.json" ]]; then
 cat "$HERE/results/run_status.json"
else
 echo "No run_status.json yet"
fi
echo "=== TABLE VIII ==="
if [[ -f "$HERE/results/report/TableVIII_V17_status.csv" ]]; then
 cat "$HERE/results/report/TableVIII_V17_status.csv"
else
 echo "Table not generated yet"
fi
echo "=== FIG.7 ==="
find "$HERE/results/report" -maxdepth 1 -type f -iname '*Fig7*' -print 2>/dev/null || true
echo "=== FLOP WARNING ==="
echo "PyTorch profiler operator FLOPs are subtotal, not verified full model FLOPs."
