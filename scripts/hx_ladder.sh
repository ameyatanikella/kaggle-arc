#!/usr/bin/env bash
# HX CPU-host ladder (docs/prereg_hx.md). usage: bash scripts/hx_ladder.sh tune|holdout [arm ...]
# tune    = families B+D (8 games), seeds 0-4; holdout = families A+C+E (17 games), seeds 0-9.
# Same budget for every arm: 3000 actions, 600 s per game, 10 workers.
set -u
cd "$(dirname "$0")/.."
SPLIT=${1:-tune}; shift || true
if [ "$SPLIT" = tune ]; then
  GAMES=lp85,vc33,g50t,ls20,re86,tr87,tu93,wa30; SEEDS=0,1,2,3,4
else
  GAMES=ft09,cn04,bp35,s5i5,cd82,dc22,r11l,su15,sb26,lf52,ar25,ka59,m0r0,sc25,sk48,sp80,tn36; SEEDS=0,1,2,3,4,5,6,7,8,9
fi
declare_arm() {
  case "$1" in
    C0) echo '{}';;
    C1) echo '{"events":true}';;
    C2) echo '{"events":true,"algebra":true}';;
    C3) echo '{"events":true,"algebra":true,"memory":true}';;
    C4) echo '{"events":true,"algebra":true,"memory":true,"models":true}';;
    C5) echo '{"events":true,"algebra":true,"memory":true,"models":true,"probes":true}';;
    Aa) echo '{"events":true,"homing":true}';;
    Ab) echo '{"events":true,"goals":true}';;
    LOO_algebra) echo '{"events":true,"memory":true,"models":true,"probes":true}';;
    LOO_models) echo '{"events":true,"algebra":true,"memory":true,"probes":true}';;
    LOO_probes) echo '{"events":true,"algebra":true,"memory":true,"models":true}';;
    LOO_memory) echo '{"events":true,"algebra":true,"models":true,"probes":true}';;
    *) echo "unknown arm $1" >&2; exit 1;;
  esac
}
ARMS=${*:-C0 C1 C2 C3 C4 C5 Aa Ab}
mkdir -p runs/hx/$SPLIT
for arm in $ARMS; do
  CFG=$(declare_arm "$arm")
  echo "== $SPLIT $arm $CFG"
  .venv/bin/python -m arc3.runner --agent hx --games "$GAMES" --seeds "$SEEDS" --workers 10 \
     --max-actions 3000 --game-seconds 600 --config "$CFG" --out "runs/hx/$SPLIT/$arm" > "runs/hx/$SPLIT/$arm.log" 2>&1
  tail -12 "runs/hx/$SPLIT/$arm.log" | grep -E '"total_levels"|"mean_score"|"errors"'
done
echo LADDER DONE
