"""Shadow evaluation of the HX2 hypothesis ledger on REAL LLM trajectories (B0 / Franzen M2 runs).

Each run's recorded actions are replayed on the local engine and fed to HXSession(ledger=True),
which never acts. Measured per run and pooled:
  * forecast ledger: per model coverage, accuracy, EFFECT accuracy (only actions that changed the
    board), change precision; failure causes;
  * goal ledger: hypotheses proposed, refuted as sufficient by a non-winning state, transfer (did a
    still-'sufficient?' hypothesis from an earlier level hold at the next win?);
  * hidden state: conflicts explained / unexplained by the fixed history features; UNDO forecasts;
  * camera shifts detected; report characters the triggers would add.
usage: python scripts/hx2_shadow.py [--source franzen_v3|b0_pair_b2c] [--workers 12]
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCES = {"franzen_v3": ROOT / "data/franzen_v3_output/benchmark.json",
           "b0_pair_b2c": ROOT / "runs/kaggle/pair_b2c/arm_b0/benchmark.json"}
AMAP = {"RESET": 0, "ACTION1": 1, "ACTION2": 2, "ACTION3": 3, "ACTION4": 4, "ACTION5": 5, "ACTION6": 6, "ACTION7": 7}


def one(job):
    os.environ.pop("ONLY_RESET_LEVELS", None)
    os.environ["OPERATION_MODE"] = "offline"
    logging.disable(logging.CRITICAL)
    import arc_agi
    from arc3.env import CompetitionEnv, ActionRefused
    from arc3.hx.session import HXSession
    run = job["run"]
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE,
                         environments_dir=str(ROOT / "data/environment_files"), logger=logging.getLogger("x"))
    env = CompetitionEnv(arc, run["game_id"])
    hx = HXSession(ledger=True)
    hx.start(env.obs.grid, 0)
    for h in run["history"]:
        a = AMAP[h["action"]["id"]]
        d = h["action"].get("data") or {}
        o0 = env.obs
        pre = o0.grid.copy()
        try:
            obs = env.step(a, d.get("x"), d.get("y"))
        except ActionRefused:
            continue
        act = (6, int(d["x"]), int(d["y"])) if a == 6 else (a,)
        hx.record(act, obs.frames, obs.state, o0.levels_completed, obs.levels_completed,
                  legal=o0.available_actions, pre_grid=pre)
    led = hx.request("ledger", {"limit": 3})
    return {"game_id": run["game_id"], "pass": job["pass"], "levels": env.obs.levels_completed,
            "recorded_levels": run.get("levels_completed"), "forecasts": led["forecasts"],
            "goal_counts": led["goals"]["counts"], "hidden": {k: v for k, v in led["hidden_state"].items() if k != "examples"},
            "camera_shifts": int(hx.stats.get("camera_shifts", 0)), "report_chars": int(hx.stats.get("report_chars", 0)),
            "triggers": {k: v for k, v in hx.stats.items() if k.startswith("trigger_")},
            "ft": {m: hx.fl.tot[m] for m in hx.fl.tot}, "fail": {m: dict(c) for m, c in hx.fl.failures.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="franzen_v3", choices=list(SOURCES))
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    d = json.loads(SOURCES[a.source].read_text())
    jobs = []
    for r in d["game_runs"]:
        m = re.search(r"_p(\d+)\.html$", str(r.get("solver_analysis_html") or ""))
        jobs.append({"run": r, "pass": int(m.group(1)) if m else 0})
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=4) as pool:
        out = list(pool.imap_unordered(one, jobs))
    path = ROOT / f"runs/hx/hx2_shadow_{a.source}.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    import numpy as np
    print(f"replay fidelity {sum(r['levels'] == r['recorded_levels'] for r in out)}/{len(out)}")
    tot = {}
    for r in out:
        for m, rec in r["ft"].items():
            t = tot.setdefault(m, {k: 0 for k in rec})
            for k, v in rec.items():
                t[k] += v
    acts = sum(r["forecasts"]["actions"] for r in out)
    print(f"scored actions {acts}")
    for m, t in tot.items():
        if t["n"]:
            print(f"  {m:8s} forecasts {t['n']:6d} (coverage {t['n'] / acts:.3f}) accuracy {t['right'] / t['n']:.3f} "
                  f"effect accuracy {t['eff_right'] / max(1, t['eff_n']):.3f} (n={t['eff_n']}) "
                  f"change precision {t['chg_did'] / max(1, t['chg_n']):.3f} (n={t['chg_n']})")
    fails = {}
    for r in out:
        for m, c in r["fail"].items():
            for k, v in c.items():
                fails.setdefault(m, {}).setdefault(k, 0)
                fails[m][k] += v
    print("failure causes", fails)
    g = {}
    for r in out:
        for k, v in r["goal_counts"].items():
            g[k] = g.get(k, 0) + v
    print("goal ledger", g)
    hid = {}
    for r in out:
        for k, v in r["hidden"].items():
            hid[k] = hid.get(k, 0) + v
    print("hidden state", hid, "camera shifts", sum(r["camera_shifts"] for r in out),
          "report chars/run median", np.median([r["report_chars"] for r in out]))


if __name__ == "__main__":
    main()
