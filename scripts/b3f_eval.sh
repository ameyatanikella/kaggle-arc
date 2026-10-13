#!/usr/bin/env bash
# Registered B3 / B3+Franzen / Franzen comparison (docs/prereg_b3_franzen.md). LLM arms run sequentially
# (one Ollama KV cache); the CPU-only B3 arm runs alongside. Reproduce: bash scripts/b3f_eval.sh
set -u
cd "$(dirname "$0")/.."
PY=build/b0src/ARC3-Inference/.venv/bin/python
GAMES=vc33,r11l,sp80,lp85,tu93,ls20,ft09,cn04
COMMON="--games $GAMES --minutes 20 --max-actions 3000 --seed 0 --base-url http://localhost:11434/v1 --model qwen3.5:35b"
mkdir -p runs/b3f/eval
$PY kaggle/local/b3_franzen.py --arm b3 $COMMON --out runs/b3f/eval/b3 > runs/b3f/eval/b3.log 2>&1 &
$PY kaggle/local/b3_franzen.py --arm b3_franzen $COMMON --out runs/b3f/eval/b3_franzen > runs/b3f/eval/b3_franzen.log 2>&1
$PY kaggle/local/b3_franzen.py --arm franzen $COMMON --out runs/b3f/eval/franzen > runs/b3f/eval/franzen.log 2>&1
wait
echo ALL DONE
