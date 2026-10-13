#!/usr/bin/env bash
# Registered B3 development comparison (docs/prereg_b3.md). Reproduce: bash scripts/b3_dev_runs.sh
set -eu
cd "$(dirname "$0")/.."
OUT=runs/b3/dev
COMMON="--games all --seeds 0,1,2 --max-actions 3000 --game-seconds 600 --workers 10"
run() { echo "== $1"; .venv/bin/python -m arc3.runner --agent "$2" $COMMON --config "$3" --out "$OUT/$1" > "$OUT/$1.log" 2>&1; tail -12 "$OUT/$1.log" | grep -E '"mean_score"|"total_levels"|"errors"' | tr -d '\n'; echo; }
mkdir -p $OUT
run B1  explorer '{}'
run V1 b3_ablation '{"plan_cache":"none"}'
run V2 b3_ablation '{"plan_cache":"projected"}'
run V3 b3_ablation '{"plan_cache":"dependency"}'
run V4 b3_ablation '{"plan_cache":"dependency","temporal":true}'
run V4b b3_ablation '{"plan_cache":"dependency","temporal":true,"goal_reuse":true}'
run V5 b3_ablation '{"plan_cache":"dependency","temporal":true,"experiments":"shared"}'
run V5s b3_ablation '{"plan_cache":"dependency","temporal":true,"experiments":"separate"}'
