#!/usr/bin/env bash
# Push the algebra notebook (minimal commit), validate the commit, then submit it once the
# next daily submission slot opens (2026-10-13 00:00 UTC). Network errors are retried;
# only explicit Kaggle states drive decisions.
set -u
cd "$(dirname "$0")/.."
export KAGGLE_API_TOKEN="$(cat .kaggle/access_token)"
K=.venv/bin/kaggle
KERNEL=axizcc/arc3-b2-algebra
log() { echo "$(date -u +%H:%M:%S) $*"; }

# 1. wait for a free GPU session slot (max 2 concurrent) and push
while true; do
  a=$($K kernels status axizcc/arc3-pair-b2c 2>&1 | tail -1)
  b=$($K kernels status axizcc/arc3-pair-nudge 2>&1 | tail -1)
  if echo "$a$b" | grep -q -E "COMPLETE|ERROR|CANCEL"; then
    out=$($K kernels push -p kaggle/b2/push/b2-algebra 2>&1 | tail -1)
    log "push: $out"
    echo "$out" | grep -q "successfully pushed" && break
  fi
  sleep 300
done

# 2. wait for the commit run to finish
while true; do
  s=$($K kernels status $KERNEL 2>&1 | tail -1)
  echo "$s" | grep -q "COMPLETE" && { log "commit COMPLETE"; break; }
  echo "$s" | grep -q -E "ERROR|CANCEL" && { log "commit FAILED: $s"; exit 2; }
  sleep 120
done

# 3. validate the commit log and outputs
mkdir -p runs/kaggle/b2_algebra_commit
$K kernels output $KERNEL -p runs/kaggle/b2_algebra_commit --file-pattern '^(submission\.parquet|.*\.log)$' >/dev/null 2>&1
LOGF=$(ls runs/kaggle/b2_algebra_commit/*.log 2>/dev/null | head -1)
python3 - "$LOGF" <<'EOF' || { log "validation FAILED - not submitting"; exit 3; }
import json, sys
raw = open(sys.argv[1]).read()
try:
    text = ''.join(e.get('data', '') for e in json.loads(raw))
except Exception:
    text = raw
checks = {
    'server ready': 'READY after' in text,
    'patches installed': "B2 install: {'installed': ['bar_fix', 'solved_memory', 'tools', 'prompt_fixes']" in text,
    'no traceback': 'Traceback (most recent call last)' not in text,
    'b2 errors 0': "'errors': 0" in text,
}
print(checks)
sys.exit(0 if all(checks.values()) else 1)
EOF
test -s runs/kaggle/b2_algebra_commit/submission.parquet || { log "no submission.parquet - not submitting"; exit 4; }

# 4. wait for the next daily slot, then submit
until [ "$(date -u +%Y%m%d)" -ge 20261013 ]; do sleep 120; done
out=$($K competitions submit arc-prize-2026-arc-agi-3 -k $KERNEL -v 1 -f submission.parquet \
      -m "B2-algebra v1: B0 + bar/prompt fixes + solved-level memory + algebra sandbox tools" 2>&1 | tail -2)
log "submit: $out"
sleep 20
$K competitions submissions arc-prize-2026-arc-agi-3 2>&1 | sed -n 3,4p | cut -c1-170
