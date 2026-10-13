#!/usr/bin/env bash
# B3 frozen-candidate qualification (docs/qualification_b3.md). Reproduce: bash scripts/b3_qualify.sh
set -eu
cd "$(dirname "$0")/.."
OUT=runs/b3/qual
COMMON="--games all --seeds 0,1,2,3,4 --max-actions 3000 --game-seconds 600 --workers 10"
mkdir -p $OUT
for spec in "B1 explorer {}" "B3nocache b3 {\"plan_cache\":\"none\"}" "B3c b3 {\"plan_cache\":\"projected\"}"; do
  set -- $spec
  echo "== $1"; .venv/bin/python -m arc3.runner --agent "$2" $COMMON --config "$3" --out "$OUT/$1" > "$OUT/$1.log" 2>&1
  grep -E '"total_levels"|"errors"' "$OUT/$1.log" | tr -d '\n'; echo
done
