#!/usr/bin/env bash
# Submit a validated kernel version once no submission is pending (limit: 1 pending per team).
set -u
cd "$(dirname "$0")/.."
export KAGGLE_API_TOKEN="$(cat .kaggle/access_token)"
K=.venv/bin/kaggle
KERNEL=${1:-axizcc/arc3-b2-algebra}; VERSION=${2:-2}
MSG=${3:-"B2-algebra v2: B0 + bar/prompt fixes + solved-level memory + algebra sandbox tools (input-path resolver)"}
log() { echo "$(date -u +%H:%M:%S) $*"; }
while true; do
  s=$($K competitions submissions arc-prize-2026-arc-agi-3 2>&1)
  if echo "$s" | grep -q "SubmissionStatus"; then
    if ! echo "$s" | grep -q "PENDING"; then
      log "no pending submission"; echo "$s" | sed -n 3,3p | cut -c1-200; break
    fi
  fi
  sleep 300
done
out=$($K competitions submit arc-prize-2026-arc-agi-3 -k $KERNEL -v $VERSION -f submission.parquet -m "$MSG" 2>&1 | tail -2)
log "submit: $out"
sleep 30
$K competitions submissions arc-prize-2026-arc-agi-3 2>&1 | sed -n 3,5p | cut -c1-200
