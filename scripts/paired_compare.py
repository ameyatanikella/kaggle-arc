"""Paired comparison of two harness arms from TAAF benchmark.json outputs.

usage: python scripts/paired_compare.py <armA_dir> <armB_dir> [--boot 10000]

Game-level pairing: each game's score is averaged over that arm's passes; the
paired delta is B - A per game; the bootstrap resamples GAMES (not runs).
These are descriptive intervals on contaminated public games, not evidence of
hidden-set generalization.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np


def load(d: Path) -> dict:
    b = json.loads((d / "benchmark.json").read_text())
    per: dict = {}
    for r in b["game_runs"]:
        g = r["game_id"]
        x = per.setdefault(g, {"score": [], "levels": [], "tokens": [], "actions": [], "won": []})
        x["score"].append(float(r.get("final_score") or 0.0))
        x["levels"].append(int(r.get("levels_completed") or 0))
        x["tokens"].append(int(r.get("final_generated_tokens") or 0))
        apl = r.get("actions_per_level") or []
        x["actions"].append(int(sum(apl)) if isinstance(apl, list) else 0)
        x["won"].append(r.get("state") == "won")
    return per


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--boot", type=int, default=10000)
    a = p.parse_args()
    A, B = load(Path(a.a)), load(Path(a.b))
    games = sorted(set(A) & set(B))
    rows = []
    for g in games:
        rows.append((g, np.mean(A[g]["score"]), np.mean(B[g]["score"]), np.mean(A[g]["levels"]), np.mean(B[g]["levels"])))
    d = np.array([r[2] - r[1] for r in rows])
    dl = np.array([r[4] - r[3] for r in rows])
    rng = random.Random(0)
    boots = []
    for _ in range(a.boot):
        idx = [rng.randrange(len(d)) for _ in range(len(d))]
        boots.append(d[idx].mean())
    lo, hi = np.percentile(boots, [5, 95])
    print(f"games={len(games)}  meanA={np.mean([r[1] for r in rows]):.2f}  meanB={np.mean([r[2] for r in rows]):.2f}")
    print(f"paired delta (B-A) = {d.mean():+.2f}  90% bootstrap CI [{lo:+.2f}, {hi:+.2f}]  "
          f"P(delta>0)={np.mean(np.array(boots) > 0):.2f}")
    print(f"levels delta per game = {dl.mean():+.3f}; wins/ties/losses by game = "
          f"{int((d > 0).sum())}/{int((d == 0).sum())}/{int((d < 0).sum())}")
    for g, sa, sb, la, lb in rows:
        print(f"  {g:16s} A={sa:6.2f} B={sb:6.2f} d={sb - sa:+7.2f}  lv {la:.1f}->{lb:.1f}")


if __name__ == "__main__":
    main()
