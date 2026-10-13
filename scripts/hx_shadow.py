"""Shadow evaluation of HX components on REAL LLM trajectories (B0 / Franzen M2 runs).

Each run's recorded action history is replayed on the local engine (deterministic), which rebuilds
the faithful event log the harness would have produced. HX then runs in shadow mode (it never acts):
  * goal transfer: candidates inferred from completed levels are re-tested at the next level start;
    when that next level was completed, did its win make an open candidate true (hit@1, hit@3)?
  * contextual algebra: before each observed transition, the algebra's prediction (if any) is
    compared with the outcome (precision; coverage = predictions / transitions);
  * triggers: level boundaries, interior no-op streaks (>= 3), prediction failures, same-state /
    same-action conflicts (hidden-state evidence);
  * report size: characters of the reports the LLM host would have injected.
Public games (contaminated): this measures component behaviour, not score.

usage: python scripts/hx_shadow.py [--source franzen_v3|b0_pair_b2c] [--limit N] [--workers 10]
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCES = {
    "franzen_v3": ROOT / "data/franzen_v3_output/benchmark.json",
    "b0_pair_b2c": ROOT / "runs/kaggle/pair_b2c/arm_b0/benchmark.json",
}
AMAP = {"RESET": 0, "ACTION1": 1, "ACTION2": 2, "ACTION3": 3, "ACTION4": 4, "ACTION5": 5, "ACTION6": 6, "ACTION7": 7}


def shadow_one(job: dict) -> dict:
    os.environ.pop("ONLY_RESET_LEVELS", None)
    os.environ["OPERATION_MODE"] = "offline"
    logging.disable(logging.CRITICAL)
    import numpy as np
    import arc_agi
    from arc3.env import CompetitionEnv, ActionRefused
    from arc3.hx.events import EventLog, MOVE, WIN, DEATH
    from arc3.hx.goals import GoalModel, describe
    from arc3.hx.algebra import Algebra
    from arc3.hx.context import GridContext
    from arc3.perception import VolatileMask

    run = job["run"]
    gid = run["game_id"]
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE,
                            environments_dir=str(ROOT / "data/environment_files"), logger=logging.getLogger("x"))
    env = CompetitionEnv(arcade, gid)
    log = EventLog()
    log.start(env.obs.grid, 0)
    vm = VolatileMask()
    ctx = GridContext(log, vm)
    alg = Algebra(ctx=ctx.ctx, inventory=ctx.inventory)
    goals = GoalModel(mask_fn=lambda: vm.mask)
    res = {"game_id": gid, "pass": job["pass"], "levels_recorded": run.get("levels_completed"), "refused": 0}
    noop_streak, streaks, pred_fail, conflicts = 0, 0, 0, 0
    alg_level, n_alg = -1, 0
    pred_tot = {"predictions": 0, "pred_checked": 0, "pred_correct": 0}
    report_chars = 0
    goal_tests = []                 # (level, open candidates at start, completed?, hit rank)
    pending_open = None
    t0 = time.time()
    for h in run["history"]:
        a = AMAP[h["action"]["id"]]
        dat = h["action"].get("data") or {}
        x, y = dat.get("x"), dat.get("y")
        obs0 = env.obs
        pre = obs0.grid.copy()
        try:
            obs = env.step(a, x, y)
        except ActionRefused:
            res["refused"] += 1
            continue
        act = (6, int(x), int(y)) if a == 6 else (a,)
        st = log.record(act, obs.frames, obs.state, obs0.levels_completed, obs.levels_completed,
                        legal=obs0.available_actions, pre_grid=pre)
        if st.kind == MOVE:
            old_mask = vm.mask.copy()
            vm.update(pre, obs.grid)
            if alg_level != st.level or not np.array_equal(old_mask, vm.mask):
                # level change or HUD mask growth: rebuild this level's algebra under the new keys
                alg = Algebra(ctx=ctx.ctx, inventory=ctx.inventory)
                alg_level = st.level
                for p0, a0, p1, _ in log.transitions(level=st.level):
                    if (p0, a0, p1) != (st.pre, act, st.post):
                        alg.observe(ctx.key(p0), a0, ctx.key(p1))
                n_alg += 1
            k0, k1 = ctx.key(st.pre), ctx.key(st.post)
            p = alg.predict(k0, act)
            if p is not None:
                alg.record_prediction(k0, act, p)
            info = alg.observe(k0, act, k1)
            if p is not None:
                pred_tot["predictions"] += 1
                pred_tot["pred_checked"] += 1
                pred_tot["pred_correct"] += int(p[0] == k1)
                pred_fail += int(p[0] != k1)
            if info["conflict"]:
                conflicts += 1
            if k0 == k1:
                noop_streak += 1
                if noop_streak == 3:
                    streaks += 1
            else:
                noop_streak = 0
        elif st.kind == DEATH:
            goals.observe_death(pre, log.grid(st.terminal))
            noop_streak = 0
        elif st.kind == WIN:
            noop_streak = 0
            lv = st.level
            visited = [log.grid(s.pre) for s in log.level_steps(lv)]
            explain = goals.observe_win(lv, log.grid(log.level_start.get(lv)), visited, log.grid(st.terminal))
            if pending_open is not None and pending_open[0] == lv:
                open_facts = [f for f, _ in pending_open[1]]
                rank = next((i for i, f in enumerate(open_facts) if f in explain), None)
                from arc3.hx.goals import facts as _facts
                vis_f = [_facts(g, vm.mask) for g in visited[:-1]] if len(visited) > 1 else []
                fpr = [round(sum(1 for fs in vis_f if f in fs) / len(vis_f), 3) if vis_f else None
                       for f in open_facts[:3]]
                goal_tests.append({"level": lv, "n_open": len(open_facts), "rank": rank,
                                   "n_hit": sum(1 for f in open_facts if f in explain), "fpr_top3": fpr,
                                   "open": [describe(f) for f in open_facts],
                                   "already_true": len(pending_open[2]), "explain": [describe(f) for f in explain[:3]]})
            start_report = goals.at_level_start(lv + 1, log.grid(st.init))
            pending_open = (lv + 1, start_report["open"], start_report["already_true"])
            report_chars += 120 + 60 * (len(start_report["open"]) + len(start_report["already_true"])) + 80 * min(3, len(explain))
    if pending_open is not None and pending_open[1]:
        goal_tests.append({"level": pending_open[0], "n_open": len(pending_open[1]), "rank": None,
                           "already_true": len(pending_open[2]), "completed": False})
    res.update({
        "levels_replayed": env.obs.levels_completed, "steps": len(log.steps),
        "transitions": sum(1 for s in log.steps if s.kind == MOVE), "noop_streak_triggers": streaks,
        "pred": pred_tot, "algebra_rebuilds": n_alg, "pred_fail_triggers": pred_fail, "conflicts": conflicts,
        "relations_live": sum(1 for r in alg.relations(limit=10000) if r["counter"] == 0),
        "goal": dict(goals.n), "goal_tests": goal_tests, "report_chars": report_chars,
        "seconds": round(time.time() - t0, 2),
    })
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="franzen_v3", choices=list(SOURCES))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = json.loads(SOURCES[a.source].read_text())
    import re
    jobs = []
    for r in d["game_runs"]:
        m = re.search(r"_p(\d+)\.html$", str(r.get("solver_analysis_html") or ""))
        jobs.append({"run": r, "pass": int(m.group(1)) if m else 0})
    if a.limit:
        jobs = jobs[:a.limit]
    ctx = mp.get_context("spawn")
    out = []
    with ctx.Pool(a.workers, maxtasksperchild=4) as pool:
        for r in pool.imap_unordered(shadow_one, jobs):
            out.append(r)
            print(f"{r['game_id'][:4]} p{r['pass']} lv {r['levels_replayed']}/{r['levels_recorded']} steps {r['steps']} "
                  f"pred {r['pred']['pred_checked']}/{r['pred']['pred_correct']} conf {r['conflicts']} "
                  f"goal tests {len(r['goal_tests'])} {r['seconds']}s", flush=True)
    path = Path(a.out or ROOT / f"runs/hx/shadow_{a.source}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=str))
    summarize(out)


def summarize(out):
    import numpy as np
    fid = sum(r["levels_replayed"] == r["levels_recorded"] for r in out)
    print(f"\nreplay fidelity: {fid}/{len(out)} runs reproduce the recorded level count")
    chk = sum(r["pred"]["pred_checked"] for r in out)
    cor = sum(r["pred"]["pred_correct"] for r in out)
    tr = sum(r["transitions"] for r in out)
    print(f"algebra: {chk} predictions checked on {tr} LLM transitions (coverage {chk / max(1, tr):.3f}), "
          f"precision {cor / max(1, chk):.3f}; conflicts {sum(r['conflicts'] for r in out)}")
    tests = [t for r in out for t in r["goal_tests"] if t.get("completed", True) and "n_hit" in t]
    done = [t for t in tests if t.get("completed", True) is not False]
    h1 = sum(1 for t in done if t["rank"] == 0)
    h3 = sum(1 for t in done if t["rank"] is not None and t["rank"] < 3)
    anyh = sum(1 for t in done if t["rank"] is not None)
    rnd = sum(t["n_hit"] / t["n_open"] for t in done if t["n_open"])
    f1 = [t["fpr_top3"][0] for t in done if t.get("fpr_top3") and t["fpr_top3"][0] is not None]
    hitf = [t["fpr_top3"][0] for t in done if t["rank"] == 0 and t.get("fpr_top3") and t["fpr_top3"][0] is not None]
    print(f"top-1 open fact true in earlier non-winning states of the next level: median {np.median(f1) if f1 else None:.3f}, "
          f"share with fpr==0 {np.mean([x == 0 for x in f1]) if f1 else 0:.2f}; for hits: median {np.median(hitf) if hitf else None}, "
          f"fpr==0 share {np.mean([x == 0 for x in hitf]) if hitf else 0:.2f} (n={len(hitf)})")
    print(f"random-choice baseline hit@1 (expected) {rnd:.1f} of {len(done)}")
    print(f"goal transfer on completed next levels: n={len(done)} hit@1 {h1} hit@3 {h3} any {anyh}; "
          f"open per test median {np.median([t['n_open'] for t in done]) if done else 0}")
    print(f"triggers per run (median): no-op streaks {np.median([r['noop_streak_triggers'] for r in out])}, "
          f"prediction failures {np.median([r['pred_fail_triggers'] for r in out])}; report chars per run median "
          f"{np.median([r['report_chars'] for r in out])}")


if __name__ == "__main__":
    main()
