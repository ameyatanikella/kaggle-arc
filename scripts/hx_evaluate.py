"""Evaluate HX CPU-host arms against the pre-registered rule (docs/prereg_hx.md).

For a pair (base, arm): per game, levels averaged over seeds; paired delta per game; 90% bootstrap
CI over games (10,000 resamples, seed 0). RETAIN if mean delta > 0 and CI low >= 0 and no new error
class and median decision time <= 2x; REJECT if CI high < 0; otherwise NEUTRAL.

usage: python scripts/hx_evaluate.py runs/hx/holdout C1:C0 C2:C1 C3:C2 C4:C3 C5:C4 Aa:C1 Ab:C1 C2:C1 ...
       (each item is ARM:BASE)
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np


def load(d: Path) -> dict:
    rs = json.loads((d / "results.json").read_text())
    per = {}
    for r in rs:
        g = per.setdefault(r["game_id"], {"levels": {}, "score": {}, "actions": {}, "err": 0, "dec": [], "stats": []})
        g["levels"][r["seed"]] = r.get("levels_completed", 0)
        g["score"][r["seed"]] = r.get("score", 0.0)
        g["err"] += 1 if r.get("error") else 0
        st = r.get("agent_stats") or {}
        dec = max(1, st.get("decisions", 1))
        g["dec"].append(r.get("agent_seconds", 0.0) / dec)
        g["stats"].append(st)
    return per


def compare(base: dict, arm: dict, boot: int = 10000) -> dict:
    games = sorted(set(base) & set(arm))
    d, ds = [], []
    for g in games:
        seeds = sorted(set(base[g]["levels"]) & set(arm[g]["levels"]))
        d.append(np.mean([arm[g]["levels"][s] - base[g]["levels"][s] for s in seeds]))
        ds.append(np.mean([arm[g]["score"][s] - base[g]["score"][s] for s in seeds]))
    d = np.array(d)
    rng = random.Random(0)
    bs = [d[[rng.randrange(len(d)) for _ in d]].mean() for _ in range(boot)]
    lo, hi = np.percentile(bs, [5, 95])
    err_b = sum(base[g]["err"] for g in games)
    err_a = sum(arm[g]["err"] for g in games)
    tb = np.median([x for g in games for x in base[g]["dec"]])
    ta = np.median([x for g in games for x in arm[g]["dec"]])
    verdict = "NEUTRAL"
    if d.mean() > 0 and lo >= 0 and err_a <= err_b and ta <= 2 * max(tb, 1e-9):
        verdict = "RETAIN"
    elif hi < 0:
        verdict = "REJECT"
    lv_b = sum(sum(base[g]["levels"].values()) for g in games)
    lv_a = sum(sum(arm[g]["levels"].values()) for g in games)
    return {"games": len(games), "levels_base": lv_b, "levels_arm": lv_a, "delta_per_game": round(float(d.mean()), 3),
            "ci90": [round(float(lo), 3), round(float(hi), 3)], "score_delta_per_game": round(float(np.mean(ds)), 3),
            "errors": [err_b, err_a], "decision_ms_median": [round(tb * 1000, 2), round(ta * 1000, 2)],
            "verdict": verdict,
            "per_game": {g: round(float(x), 2) for g, x in zip(games, d) if x != 0}}


def main():
    root = Path(sys.argv[1])
    out = {}
    for item in sys.argv[2:]:
        arm, base = item.split(":")
        r = compare(load(root / base), load(root / arm))
        out[item] = r
        print(f"{arm:>12s} vs {base:<4s} levels {r['levels_base']:4d} -> {r['levels_arm']:4d}  "
              f"d/game {r['delta_per_game']:+.3f} CI90 [{r['ci90'][0]:+.3f}, {r['ci90'][1]:+.3f}]  "
              f"score d {r['score_delta_per_game']:+.3f}  err {r['errors']}  dec ms {r['decision_ms_median']}  "
              f"=> {r['verdict']}  {r['per_game']}")
    (root / "evaluation.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
