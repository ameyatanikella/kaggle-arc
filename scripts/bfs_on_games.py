"""bfs_plan and zm_solve on the REAL public games (level 1, offline engine).

Analysis tooling only. It uses the engine itself as the simulator (deep-copied game
objects), which no agent can do in competition. It therefore measures a CEILING: what
the planning tools return when the model is exactly right. It also checks one assumption
the model would make when writing its own key function: that the visible frame (minus
the step counter) identifies the state.

D. bfs_plan, level 1. Actions = available keys (1-5) plus, in click games, the clicks
   that change the frame. key = frame with the edge band removed (step counters live there).
   Each returned plan is replayed on a fresh instance: did it complete level 1?
   Plan length is compared with the level-1 human baseline.
E. zm_solve, modular mechanics. For action pairs (a, b) where k presses of a return the
   interior frame to the start (period m) and a·b is the identity, every residue r is
   reached by pressing a r times. zm_solve on the Z_m model picks the cheapest presses, and
   the plan is replayed and compared with the target frame.
F. Visible-Markov check (math_contract principle 14) on the engine: two histories with the
   same interior frame, same next action, different next frame => hidden state; a frame-keyed
   BFS can then report 'exhausted' for reachable goals.

usage: python scripts/bfs_on_games.py [--games all] [--time 120] [--nodes 30000]
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "kaggle" / "b2"))
os.environ["OPERATION_MODE"] = "offline"
os.environ.pop("ONLY_RESET_LEVELS", None)

import arc_agi  # noqa: E402
from arcengine import ActionInput, GameAction  # noqa: E402

import sandbox_tools as T  # noqa: E402
from algebra_on_games import Game, interior, quiet  # noqa: E402
from arc3.perception import background_color, components  # noqa: E402
from arc3.runner import ENV_DIR, list_games  # noqa: E402


def baseline_l1(gid):
    name, ver = gid.split("-")
    meta = json.loads((ENV_DIR / name / ver / "metadata.json").read_text())
    return meta["baseline_actions"][0]


def effective_clicks(g, limit=30):
    bg = background_color(g.f0)
    inside = interior(g.f0.shape)
    seen, out = set(), []
    for c in sorted(components(g.f0, skip_colors=(bg,)), key=lambda c: (c.size, c.y0, c.x0)):
        if (c.cx, c.cy) in seen:
            continue
        seen.add((c.cx, c.cy))
        f, info = g.play([("click", c.cx, c.cy)])
        if info["level_up"] or ((f != g.f0) & inside).sum() > 4:
            out.append(("click", c.cx, c.cy))
        if len(out) >= limit:
            break
    return out


class Node:
    __slots__ = ("game", "frame", "won", "lost")

    def __init__(self, game, frame, won=False, lost=False):
        self.game, self.frame, self.won, self.lost = game, frame, won, lost


def engine_step(node, act):
    if node.won or node.lost:
        return None
    g2 = copy.deepcopy(node.game)
    if act[0] == "click":
        ai = ActionInput(id=GameAction.ACTION6, data={"x": int(act[1]), "y": int(act[2])})
    else:
        ai = ActionInput(id=GameAction.from_id(act[1]), data={})
    r = g2.perform_action(ai, raw=True)
    f = np.array(r.frame[-1], dtype=np.int16) if r.frame else node.frame
    st = r.state.name if hasattr(r.state, "name") else str(r.state)
    return Node(g2, f, won=r.levels_completed > 0, lost=(st == "GAME_OVER"))


def bfs_experiment(arc, g, time_limit, max_nodes):
    w = arc.make(g.gid)
    start = Node(w._game, g.f0)
    acts = [("key", a) for a in (1, 2, 3, 4, 5) if a in g.avail]
    if 6 in g.avail:
        acts += effective_clicks(g)
    inside = interior(g.f0.shape)
    t0 = time.time()
    r = T.bfs_plan(start, engine_step, lambda n: n.won, acts, max_nodes=max_nodes,
                   key=lambda n: (n.won, n.lost, n.frame[inside].tobytes()), time_limit=time_limit)
    g.acts = acts
    out = {"actions": len(acts), "nodes": r["nodes"], "exhausted": r["exhausted"],
           "seconds": round(time.time() - t0, 1), "baseline": baseline_l1(g.gid)}
    if r["plan"] is not None:
        _, info = g.play(r["plan"])
        out.update({"plan_len": len(r["plan"]), "replay_completes_level": info["level_up"],
                    "plan": [list(a) for a in r["plan"]]})
    return out


def markov_conflicts(arc, g, acts, max_dups=150, time_limit=60.0):
    """Principle 14 on the engine: explore by frame-deduplicated BFS; whenever a second history
    reaches an already-seen interior frame, step both histories with every action and compare.
    Any mismatch proves that the visible frame does not determine the state, so a frame-keyed
    BFS (as a model-written simulator would use) can prune reachable states."""
    inside = interior(g.f0.shape)
    key = lambda n: n.frame[inside].tobytes()
    start = Node(arc.make(g.gid)._game, g.f0)
    first = {key(start): start}
    q = [start]
    dups = conflicts = 0
    example = None
    t_end = time.time() + time_limit
    while q and dups < max_dups and time.time() < t_end:
        nq = []
        for n in q:
            for a in acts:
                m = engine_step(n, a)
                if m is None or m.won or m.lost:
                    continue
                k = key(m)
                if k not in first:
                    first[k] = m
                    nq.append(m)
                    continue
                dups += 1
                o = first[k]
                for b in acts:
                    x, y = engine_step(m, b), engine_step(o, b)
                    if x is None or y is None:
                        continue
                    if not np.array_equal(x.frame[inside], y.frame[inside]) or x.won != y.won or x.lost != y.lost:
                        conflicts += 1
                        example = example or {"action": list(b)}
                        break
                if dups >= max_dups or time.time() > t_end:
                    break
            if dups >= max_dups or time.time() > t_end:
                break
        q = nq
    return {"frames_seen": len(first), "duplicate_histories_checked": dups, "conflicts": conflicts,
            "example": example}


def zm_experiment(g, pairs, max_period=12):
    inside = interior(g.f0.shape)
    res = []
    for a, b in pairs:
        m = None
        for k in range(2, max_period + 1):
            f, info = g.play([a] * k)
            if info["level_up"] or info["game_over"]:
                break
            if np.array_equal(f[inside], g.f0[inside]):
                m = k
                break
        if m is None:
            continue
        fab, _ = g.play([a, b])
        if not np.array_equal(fab[inside], g.f0[inside]):
            continue
        ok = cheaper_via_b = 0
        for r in range(1, m):
            tgt, _ = g.play([a] * r)
            sol = T.zm_solve([[1, m - 1]], [r], m, cost=[1, 1])
            u = sol["presses"]
            got, _ = g.play([a] * u[0] + [b] * u[1])
            ok += int(np.array_equal(got[inside], tgt[inside]))
            cheaper_via_b += int(u[1] > 0)
            assert sol["cost"] == min(r, m - r)
        res.append({"a": list(a), "b": list(b), "period": m, "targets": m - 1, "replay_ok": ok,
                    "plans_using_inverse_button": cheaper_via_b})
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", default="all")
    p.add_argument("--time", type=float, default=120.0)
    p.add_argument("--nodes", type=int, default=30000)
    p.add_argument("--out", default=str(ROOT / "runs" / "algebra_on_games" / "bfs_zm.json"))
    a = p.parse_args()
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(ENV_DIR), logger=quiet())
    games = list_games() if a.games == "all" else [x for x in list_games() if x.split("-")[0] in a.games.split(",")]
    results = {}
    for gid in games:
        g = Game(arc, gid)
        r = {"bfs": bfs_experiment(arc, g, a.time, a.nodes)}
        r["markov"] = markov_conflicts(arc, g, g.acts)
        keys = [("key", k) for k in (1, 2, 3, 4) if k in g.avail]
        r["zm"] = zm_experiment(g, [(x, y) for x in keys for y in keys if x != y])
        results[gid] = r
        b = r["bfs"]
        print(f"{gid:16s} bfs: acts={b['actions']:>2} nodes={b['nodes']:>6} {b['seconds']:>6}s "
              f"exhausted={b['exhausted']!s:5} plan={b.get('plan_len', '-')!s:>3} "
              f"replay_ok={b.get('replay_completes_level', '-')!s:5} baseline={b['baseline']} | "
              f"zm: {[(z['period'], z['replay_ok'], z['targets']) for z in r['zm']]} | markov conflicts "
              f"{r['markov']['conflicts']}/{r['markov']['duplicate_histories_checked']}", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main()
