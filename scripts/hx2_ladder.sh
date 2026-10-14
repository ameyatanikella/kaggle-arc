#!/usr/bin/env bash
# HX2 CPU-host ablation (docs/prereg_hx2.md). usage: caffeinate -i bash scripts/hx2_ladder.sh tune|holdout [arm ...]
set -u
cd "$(dirname "$0")/.."
SPLIT=${1:-tune}; shift || true
if [ "$SPLIT" = tune ]; then
  GAMES=lp85,vc33,g50t,ls20,re86,tr87,tu93,wa30; SEEDS=0,1,2,3,4
else
  GAMES=ft09,cn04,bp35,s5i5,cd82,dc22,r11l,su15,sb26,lf52,ar25,ka59,m0r0,sc25,sk48,sp80,tn36; SEEDS=0,1,2,3,4,5,6,7,8,9
fi
cfg() {
  case "$1" in
    A) echo '{"events":true}';;
    B) echo '{"events":true,"ledger":true}';;
    C) echo '{"events":true,"ledger":true,"history":true}';;
    D) echo '{"events":true,"ledger":true,"history":true,"algebra":true,"memory":true,"models":true,"probes":true,"certify":true}';;
    *) echo "unknown arm $1" >&2; exit 1;;
  esac
}
ARMS=${*:-A B C D}
mkdir -p runs/hx2/$SPLIT
for arm in $ARMS; do
  echo "== $SPLIT $arm $(cfg $arm)"
  .venv/bin/python -m arc3.runner --agent hx --games "$GAMES" --seeds "$SEEDS" --workers 8 \
     --max-actions 3000 --game-seconds 600 --config "$(cfg $arm)" --out "runs/hx2/$SPLIT/$arm" > "runs/hx2/$SPLIT/$arm.log" 2>&1
  tail -12 "runs/hx2/$SPLIT/$arm.log" | grep -E '"total_levels"|"errors"'
done
echo LADDER DONE
