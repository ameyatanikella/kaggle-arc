"""DIAG stage 2: shadow mode on development traces (docs/prereg_diag.md). Checkers only observe.

Traces: B0's recorded level-1 actions (data/franzen_v3_output/transcripts, first pass with
level-1 actions), replayed on the offline engine; automatic RESET after GAME OVER starts a new
attempt. Observations come from frames only (no engine internals):
  * mover components (linear over Q), per attempt and coordinate: the object that moves under
    a directional key, tracked by exact shape; position = p0 + sum_a n_a * v_a over EFFECTIVE
    moves (a move is effective iff the tracked object moved: the wall guard, read from the
    clean frames); identity 'uncertain' when several objects share the shape.
  * toggle components (linear over GF(2)), per attempt and interior cell: cell changed vs the
    attempt's first frame = parity of presses of the buttons that flip it; button = bounding box
    of the object under the click; applicable only while the attempt contains clicks only.
    Cells that take > 2 colours are unsupported (binary model). Cells with identical value
    sequences are one class (checked once, multiplicity recorded).
Metrics M1-M5 as pre-registered. Fault injection: Q -> add a nonzero integer in [-5, 5];
GF(2) -> flip; k in {1,2,3}, 30 trials per component instance, fixed seeds.

usage: python scripts/diag_shadow.py [--games all] [--instances 4] [--trials 30]
"""
import argparse
import json
import random
import re
import sys
import time
import zlib
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from bfs_on_games import ENV_DIR, arc_agi, list_games, quiet  # noqa: E402
from exhausted_witness import TOK, TR, parse_act  # noqa: E402
from arcengine import GameAction  # noqa: E402

from arc3.algebra.linalg import solve  # noqa: E402
from arc3.diag import structure as S  # noqa: E402
from arc3.diag.checkers import AlgebraicChecker, Fitter, OrdinaryChecker  # noqa: E402
from arc3.diag.ledger import Component, Observation  # noqa: E402
from arc3.perception import background_color, components  # noqa: E402

EDGE = 3


def level1_actions(gid, cap=200):
    for f in sorted(TR.glob(f"{gid}_p*.txt")):
        acts = []
        for line in f.read_text(errors="replace").splitlines():
            if not line.startswith("[action] "):
                continue
            body = line[9:]
            if body.startswith("batch of"):
                txt = body.split("executed:", 1)[1].split("|")[0]
                parsed = [parse_act(m.group(0)) for m in TOK.finditer(txt)]
                done = "stop_reason: level_completed" in body or "stop_reason: run_complete" in body
            else:
                head, _, res = body.partition(" -> ")
                parsed = [parse_act(head)]
                done = res.startswith(("LEVEL COMPLETED", "RUN COMPLETE"))
            if any(p is None for p in parsed):
                break
            acts += parsed
            if done or len(acts) >= cap:
                break
        if acts:
            return f.stem.split("_")[-1], acts[:cap]
    return None, []


def replay(arc, gid, acts):
    w = arc.make(gid)
    f0 = np.array(w.observation_space.frame[-1], dtype=np.int16)
    steps = []                                   # (action, frame_before, frame_after, attempt, n_frames)
    attempt, prev = 0, f0
    for a in acts:
        if a[0] == "click":
            r = w.step(GameAction.ACTION6, data={"x": a[1], "y": a[2]})
        else:
            r = w.step(GameAction.from_id(a[1]))
        if r is None:
            break
        f = np.array(r.frame[-1], dtype=np.int16) if r.frame else prev
        st = r.state.name if hasattr(r.state, "name") else str(r.state)
        if r.levels_completed > 0:
            break                                # the after-frame is the NEXT level: not an observation
        steps.append({"a": a, "before": prev, "after": f, "attempt": attempt, "nframes": len(r.frame or [])})
        if st == "GAME_OVER":
            r = w.step(GameAction.RESET)
            attempt += 1
            f = np.array(r.frame[-1], dtype=np.int16) if r and r.frame else f0
        prev = f
    return f0, steps


def interior(shape):
    m = np.zeros(shape, bool)
    m[EDGE:-EDGE, EDGE:-EDGE] = True
    return m


# ------------------------------------------------------------------ component builders
def sig(grid, c):
    return (c.color, (grid[c.y0:c.y1 + 1, c.x0:c.x1 + 1] == c.color).tobytes(), c.y1 - c.y0, c.x1 - c.x0)


def build_movers(f0, steps):
    inside = interior(f0.shape)
    bg = background_color(f0)
    mover = None
    for s in steps:
        if s["a"][0] == "key" and s["a"][1] in (1, 2, 3, 4):
            d = (s["before"] != s["after"]) & inside
            if not d.any():
                continue
            for c in sorted(components(s["before"], skip_colors=(bg,)), key=lambda c: -c.size):
                if not d[c.y0:c.y1 + 1, c.x0:c.x1 + 1].any() or c.size < 2:
                    continue
                g = sig(s["before"], c)
                if any(sig(s["after"], e) == g and (e.y0, e.x0) != (c.y0, c.x0)
                       for e in components(s["after"], skip_colors=(bg,))):
                    mover = g
                    break
            if mover:
                break
    if mover is None:
        return []
    obs, last, counts, att = [], None, [0, 0, 0, 0], -1
    for t, s in enumerate(steps):
        if s["attempt"] != att:
            att, counts, last = s["attempt"], [0, 0, 0, 0], None
            cs = [e for e in components(s["before"], skip_colors=(bg,)) if sig(s["before"], e) == mover]
            if len(cs) >= 1:
                e = min(cs, key=lambda e: (e.y0, e.x0))
                last = (e.y0, e.x0)
                ident = "certain" if len(cs) == 1 else "uncertain"
                for q, v in (("x", e.x0), ("y", e.y0)):
                    obs.append(Observation(f"a{att}t{t}s{q}", q, v, {"t": t, "attempt": att, "n": tuple(counts),
                                                                      "identity": ident}))
        cs = [e for e in components(s["after"], skip_colors=(bg,)) if sig(s["after"], e) == mover]
        if not cs:
            continue
        e = min(cs, key=lambda e: abs(e.y0 - last[0]) + abs(e.x0 - last[1])) if last else cs[0]
        pos = (e.y0, e.x0)
        a = s["a"]
        if a[0] == "key" and a[1] in (1, 2, 3, 4) and last is not None and pos != last:
            counts[a[1] - 1] += 1
        other_moved = last is not None and pos != last and not (a[0] == "key" and a[1] in (1, 2, 3, 4))
        last = pos
        ident = "certain" if len(cs) == 1 else "uncertain"
        for q, v in (("x", e.x0), ("y", e.y0)):
            obs.append(Observation(f"a{att}t{t}{q}", q, v, {"t": t, "attempt": att, "n": tuple(counts),
                                                           "identity": ident, "other_moved": other_moved}))
    comps = []
    for att in sorted({o.context["attempt"] for o in obs}):
        for q in ("x", "y"):
            def mk(att=att, q=q):
                return Component(
                    name=f"mover_{q}_a{att}", version=1, kind="linear", field="q",
                    params=["p0", "v_up", "v_down", "v_left", "v_right"],
                    explains=lambda o, q=q, att=att: o.quantity == q and o.context["attempt"] == att,
                    applies=lambda o: ((o.context["identity"] == "certain"), "identity uncertain"),
                    row=lambda ctx: [1, *ctx["n"]], design_key=lambda ctx: ctx["n"],
                    context_signature=lambda ctxs, att=att: ("attempt", att),
                    applicability_text="same attempt; tracked identity certain; effective moves only")
            comps.append(mk())
    return [(c, [o for o in obs if c.explains(o)]) for c in comps]


def build_toggles(f0, steps, max_classes=400):
    inside = interior(f0.shape)
    bg = background_color(f0)
    out = []
    for att in sorted({s["attempt"] for s in steps}):
        ss = [(t, s) for t, s in enumerate(steps) if s["attempt"] == att]
        start = ss[0][1]["before"]
        buttons, presses, rows, frames = {}, [], [], []
        for t, s in ss:
            a = s["a"]
            if a[0] != "click":
                break                                    # applicability: clicks-only prefix
            x, y = a[1], a[2]
            key = ("bg",)
            for c in components(s["before"], skip_colors=(bg,)):
                if c.y0 <= y <= c.y1 and c.x0 <= x <= c.x1 and s["before"][y, x] == c.color:
                    key = (c.y0, c.x0, c.y1, c.x1)
                    break
            buttons.setdefault(key, len(buttons))
            presses.append(buttons[key])
            frames.append((t, s["after"]))
        if len(frames) < 4:
            continue
        nb = len(buttons)
        par = [0] * nb
        designs = [tuple([1] + par)]
        cells_frames = [start]
        ts = [ss[0][0] - 0.5]
        for (t, f), b in zip(frames, presses):
            par[b] ^= 1
            designs.append(tuple([1] + par))
            cells_frames.append(f)
            ts.append(t)
        stack = np.stack(cells_frames)
        changed = (stack != start[None]).any(axis=0) & inside
        classes = {}
        for yy, xx in zip(*np.nonzero(changed)):
            col = stack[:, yy, xx]
            if len(set(col.tolist())) > 2:
                classes.setdefault("unsupported", []).append((yy, xx))
                continue
            seq = tuple(int(v != start[yy, xx]) for v in col)
            classes.setdefault(seq, []).append((yy, xx))
        unsupported = len(classes.pop("unsupported", []))
        for k, (seq, cells) in enumerate(list(classes.items())[:max_classes]):
            obs = [Observation(f"a{att}c{k}s{i}", f"cell{k}", v, {"step": i, "d": designs[i], "attempt": att})
                   for i, v in enumerate(seq)]
            comp = Component(name=f"toggle_a{att}_c{k}", version=1, kind="linear", field="gf2",
                             params=["s0"] + [f"btn{j}" for j in range(nb)],
                             explains=lambda o, k=k: o.quantity == f"cell{k}",
                             applies=lambda o: (True, ""), row=lambda ctx: list(ctx["d"]),
                             design_key=lambda ctx: ctx["d"],
                             context_signature=lambda ctxs, att=att: ("attempt", att, "clicks-only"),
                             applicability_text="same attempt, clicks-only prefix, 2-colour cell")
            out.append((comp, obs, len(cells), unsupported))
    return out


# ------------------------------------------------------------------ metrics
def checkable_flags(comp, obs):
    elig = [o for o in obs if comp.applies(o)[0]]
    rows = [comp.row(o.context) for o in elig]
    flags = []
    for i in range(len(rows)):
        flags.append(i > 0 and S.rank(rows[:i + 1], comp.field) == S.rank(rows[:i], comp.field))
    return elig, flags


def oracle_consistent(comp, obs):
    elig = [o for o in obs if comp.applies(o)[0]]
    if not elig:
        return False
    if comp.field == "gf2":
        from arc3.algebra.modular import gf2_solve_all
        return gf2_solve_all([comp.row(o.context) for o in elig], [int(o.value) & 1 for o in elig]) is not None
    return solve([comp.row(o.context) for o in elig], [Fraction(o.value) for o in elig]) is not None


def corrupt(o, rng, field):
    if field == "gf2":
        return Observation(o.id, o.quantity, 1 - (int(o.value) & 1), o.context)
    d = rng.choice([x for x in range(-5, 6) if x != 0])
    return Observation(o.id, o.quantity, o.value + d, o.context)


def outcome(e, truth):
    truth = sorted(truth)
    if e.verdict == "contradiction-localized":
        return "exact" if sorted(e.diagnoses[0]) == truth else "confident_wrong"
    if e.verdict == "contradiction-ambiguous":
        return "correctly_ambiguous" if truth in [sorted(d) for d in e.diagnoses] else "ambiguous_wrong"
    if e.verdict == "consistent":
        return "missed_undetectable" if set(truth) <= set(e.unverifiable) else "missed"
    return "inadequate" if e.verdict == "hypothesis-inadequate" else e.verdict


def flagged(e):
    if e.verdict == "contradiction-localized":
        return set(e.diagnoses[0])
    if e.verdict == "contradiction-ambiguous":
        return set.intersection(*[set(d) for d in e.diagnoses])
    return set()


def repaired_accuracy(comp, check_obs, e, holdout):
    drop = set()
    if e.verdict.startswith("contradiction"):
        drop = set().union(*[set(d) for d in e.diagnoses])
    keep = [o for o in check_obs if o.id not in drop and comp.applies(o)[0]]
    hold = [o for o in holdout if comp.applies(o)[0]]
    if not keep or not hold:
        return None
    F = Fitter(comp, keep + hold, seed=3)
    idx = list(range(len(keep)))
    preds = []
    for s in range(2):                           # two random particular solutions: determinedness
        F.rng.seed(s)
        th = F.fit(idx)
        if th is None:
            return {"determined": 0, "correct": 0, "n": len(hold), "fit_failed": 1}
        preds.append([F.predict(th, len(keep) + j) for j in range(len(hold))])
    det = [j for j in range(len(hold)) if preds[0][j] == preds[1][j]]
    ok = sum(1 for j in det if F.agrees(preds[0][j], len(keep) + j))
    return {"determined": len(det), "correct": ok, "n": len(hold)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", default="all")
    p.add_argument("--instances", type=int, default=4)
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--out", default=str(ROOT / "runs" / "diag" / "shadow.json"))
    a = p.parse_args()
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(ENV_DIR), logger=quiet())
    games = list_games() if a.games == "all" else [g for g in list_games() if g.split("-")[0] in a.games.split(",")]
    res = {}
    for gid in games:
        t_game = time.perf_counter()
        pas, acts = level1_actions(gid)
        if not acts:
            res[gid] = {"note": "no level-1 actions in transcripts"}
            continue
        f0, steps = replay(arc, gid, acts)
        insts = [(c, o, 1, 0) for c, o in build_movers(f0, steps)] + build_toggles(f0, steps)
        g = {"pass": pas, "actions": len(acts), "steps": len(steps),
             "attempts": len({s["attempt"] for s in steps}), "components": {}, "M1": {}, "M2": [], "M3": [],
             "M5": {}}
        # ---- M1 encounter (per step, any component) + online contradiction events
        step_elig, step_check = set(), set()
        n_unsupported = 0
        contra = {"B_detect_events": 0}
        for comp, obs, mult, uns in insts:
            n_unsupported = max(n_unsupported, uns)
            elig, flags = checkable_flags(comp, obs)
            for o, fl in zip(elig, flags):
                key = (o.context.get("attempt"), o.context.get("t", o.context.get("step")))
                step_elig.add(key)
                if fl:
                    step_check.add(key)
            g["components"].setdefault(comp.name.split("_")[0], 0)
            g["components"][comp.name.split("_")[0]] += 1
        g["M1"] = {"eligible_steps": len(step_elig), "checkable_steps": len(step_check),
                   "rate": len(step_check) / max(1, len(step_elig)), "unsupported_cells": n_unsupported}
        # ---- final verdicts on real (uninjected) data: real contradictions
        Bc, Cc = OrdinaryChecker(), AlgebraicChecker()
        real = {}
        for comp, obs, mult, uns in insts:
            eb, ec = Bc.check(comp, obs), Cc.check(comp, obs)
            real.setdefault(f"B:{eb.verdict}", 0)
            real[f"B:{eb.verdict}"] += 1
            real.setdefault(f"C:{ec.verdict}", 0)
            real[f"C:{ec.verdict}"] += 1
        g["real_verdicts"] = real
        # ---- choose clean instances (oracle-consistent, enough data) for M2-M4
        rng = random.Random(zlib.crc32(gid.encode()))
        clean = [(c, o) for c, o, m, u in insts if oracle_consistent(c, o)
                 and sum(1 for x in o if c.applies(x)[0]) >= len(c.params) + 3]
        rng.shuffle(clean)
        movers = [x for x in clean if x[0].name.startswith("mover")]
        toggles = [x for x in clean if x[0].name.startswith("toggle")]
        chosen = (movers[:2] + toggles)[:a.instances]
        cache = S.StructureCache(enabled=True)
        C = AlgebraicChecker(cache=cache)
        Cn = AlgebraicChecker(cache=S.StructureCache(enabled=False))
        B = OrdinaryChecker()
        times = {"B": [], "C": [], "Cnocache": []}
        for comp, obs in chosen:
            elig = [o for o in obs if comp.applies(o)[0]]
            nh = max(1, len(elig) // 5)
            check_obs, hold = elig[:-nh], elig[-nh:]
            # M3 false alarms on the clean check set
            eb, ec = B.check(comp, check_obs), C.check(comp, check_obs)
            g["M3"].append({"comp": comp.name, "B": eb.verdict, "C": ec.verdict, "n": len(check_obs),
                            "C_unverifiable": len(ec.unverifiable), "B_unverifiable": len(eb.unverifiable)})
            for k in (1, 2, 3):
                if len(check_obs) <= k + 1:
                    continue
                trng = random.Random(1000 * k + len(check_obs))
                for trial in range(a.trials):
                    pick = set(trng.sample(range(len(check_obs)), k))
                    data = [corrupt(o, trng, comp.field) if i in pick else o for i, o in enumerate(check_obs)]
                    truth = [check_obs[i].id for i in pick]
                    t0 = time.perf_counter()
                    eb = B.check(comp, data)
                    t1 = time.perf_counter()
                    ec = C.check(comp, data)
                    t2 = time.perf_counter()
                    ecn = Cn.check(comp, data)
                    t3 = time.perf_counter()
                    times["B"].append(t1 - t0)
                    times["C"].append(t2 - t1)
                    times["Cnocache"].append(t3 - t2)
                    assert ecn.verdict == ec.verdict and ecn.diagnoses == ec.diagnoses
                    row = {"comp": comp.name, "field": comp.field, "k": k, "n": len(check_obs)}
                    for name, e in (("B", eb), ("C", ec)):
                        fl = flagged(e)
                        row[name] = {"outcome": outcome(e, truth), "verdict": e.verdict,
                                     "n_diag": len(e.diagnoses), "tp": len(fl & set(truth)), "fp": len(fl - set(truth)),
                                     "repair": repaired_accuracy(comp, data, e, hold),
                                     "fallback": e.fallback}
                    # suggested reobservation: does replacing that observation with its true value resolve?
                    sug = [s for s in ec.suggestions if s["type"] == "reobserve"]
                    if sug:
                        oid = sug[0]["obs"]
                        fixed = [check_obs[i] if check_obs[i].id == oid else d for i, d in enumerate(data)]
                        e2 = C.check(comp, fixed)
                        rem = [x for x in truth if x != oid]
                        row["C"]["reobserve"] = {"was_faulty": oid in truth, "after": outcome(e2, rem) if rem else e2.verdict}
                    g["M2"].append(row)
        g["M5"] = {k: {"n": len(v), "median_ms": 1000 * float(np.median(v)) if v else None,
                       "p95_ms": 1000 * float(np.percentile(v, 95)) if v else None} for k, v in times.items()}
        g["M5"]["cache"] = dict(cache.stats, memory_bytes=cache.memory_bytes(), entries=len(cache.store))
        g["seconds"] = time.perf_counter() - t_game
        res[gid] = g
        print(f"{gid[:4]} steps={g['steps']:>3} att={g['attempts']} comps={g['components']} "
              f"M1={g['M1']['checkable_steps']}/{g['M1']['eligible_steps']} trials={len(g['M2'])} "
              f"t={g['seconds']:.0f}s", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
