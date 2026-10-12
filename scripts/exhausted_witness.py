"""Witnesses for the 'exhausted' bfs_plan results in docs/algebra_on_games.md section 4.

An 'exhausted' result means only: no plan exists in the SUPPLIED search graph (start-state
action set, state key, goal test). This script looks for evidence about which part of the
graph was inadequate, game by game, using B0's own winning level-1 action segments
(data/franzen_v3_output/transcripts) as known solutions.

For each game where B0 completed level 1 at least once:
  1. Extract the segment from the last level start (start or post-GAME-OVER reset) up to the
     first level completion, then replay it on a fresh instance. Only replay-verified
     segments are used.
  2. Rerun bfs_plan with action set A' = A (the original graph's actions) plus the segment's
     distinct actions, with two state keys: 'masked' (interior frame, as in section 4) and
     'raw' (every frame the engine returned for the step, unmasked).
     * Plan found with A' under the masked key: the omitted actions are SUFFICIENT for the
       graph to contain a winning path (a witness for "action set incomplete").
     * A' still 'exhausted' under a key although the segment is a winning path made only of A'
       actions: the key merged a state on that path with a different state (a witness for
       "state merging" under that key). 'raw' succeeding where 'masked' fails = the merging is
       caused by the mask, not by information absent from the observation.
  3. Visible-Markov check (as in section 4) with the RAW key over the original action set:
     conflicts there show state that is not determined by the full observation.

usage: python scripts/exhausted_witness.py [--games g50t,ka59,...] [--time 120]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from bfs_on_games import (ENV_DIR, Game, T, arc_agi, effective_clicks,  # noqa: E402
                          interior, list_games, quiet)
from bfs_on_games import Node as _SlotNode  # noqa: E402


class Node(_SlotNode):
    __slots__ = ("raw",)                    # the full, unmasked observation of the last step
from arcengine import ActionInput, GameAction  # noqa: E402

TR = ROOT / "data" / "franzen_v3_output" / "transcripts"
KEYS = {"UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4, "SPACE": 5, "UNDO": 7}
TOK = re.compile(r"MOUSE\(row=(\d+), col=(\d+)\)|\b(UP|DOWN|LEFT|RIGHT|SPACE|UNDO)\b")


def parse_act(s):
    m = TOK.fullmatch(s.strip())
    if not m:
        return None
    if m.group(1):
        return ("click", int(m.group(2)), int(m.group(1)))      # x = col, y = row
    return ("key", KEYS[m.group(3)])


def level1_segments(gid):
    """Replay-candidate segments: actions since the last level start up to level-1 completion."""
    segs = []
    for f in sorted(TR.glob(f"{gid}_p*.txt")):
        seg, done = [], False
        for line in f.read_text(errors="replace").splitlines():
            if not line.startswith("[action] "):
                continue
            body = line[len("[action] "):]
            if body.startswith("batch of"):
                acts = body.split("executed:", 1)[1].split("|")[0]
                parsed = [parse_act(m.group(0)) for m in TOK.finditer(acts)]
                n_exec = re.search(r"(\d+) executed", body)
                if n_exec and int(n_exec.group(1)) != len(parsed):
                    parsed = [None]                # listing truncated or unparseable: stop here
                stop = re.search(r"stop_reason: (\w+)", body)
                stop = stop.group(1) if stop else ""
            else:
                head, _, res = body.partition(" -> ")
                parsed = [parse_act(head)]
                stop = ("level_completed" if res.startswith(("LEVEL COMPLETED", "RUN COMPLETE"))
                        else "game_over" if res.startswith("GAME OVER") else "")
            if any(p is None for p in parsed):
                break
            seg += parsed
            if stop in ("level_completed", "run_complete"):
                done = True
                break
            if stop == "game_over":
                seg = []                       # automatic RESET: level restarts
        if done:
            segs.append({"pass": f.stem.split("_")[-1], "actions": seg})
    return segs


def raw_step(node, act):
    if node.won or node.lost:
        return None
    import copy
    g2 = copy.deepcopy(node.game)
    if act[0] == "click":
        ai = ActionInput(id=GameAction.ACTION6, data={"x": int(act[1]), "y": int(act[2])})
    else:
        ai = ActionInput(id=GameAction.from_id(act[1]), data={})
    r = g2.perform_action(ai, raw=True)
    frames = [np.array(fr, dtype=np.int16) for fr in r.frame] if r.frame else [node.frame]
    st = r.state.name if hasattr(r.state, "name") else str(r.state)
    n = Node(g2, frames[-1], won=r.levels_completed > 0, lost=(st == "GAME_OVER"))
    n.raw = b"".join(f.tobytes() for f in frames)
    return n


def run_bfs(arc, g, acts, keyname, time_limit, max_nodes):
    inside = interior(g.f0.shape)
    start = Node(arc.make(g.gid)._game, g.f0)
    start.raw = g.f0.tobytes()
    if keyname == "masked":
        key = lambda n: (n.won, n.lost, n.frame[inside].tobytes())
    else:
        key = lambda n: (n.won, n.lost, n.raw)
    r = T.bfs_plan(start, raw_step, lambda n: n.won, acts, max_nodes=max_nodes, key=key,
                   time_limit=time_limit)
    out = {"nodes": r["nodes"], "exhausted": r["exhausted"], "plan_len": None, "replay_ok": None}
    if r["plan"] is not None:
        _, info = g.play(r["plan"])
        out.update({"plan_len": len(r["plan"]), "replay_ok": info["level_up"]})
    return out


def raw_markov(arc, g, acts, max_dups=150, time_limit=60.0):
    start = Node(arc.make(g.gid)._game, g.f0)
    start.raw = g.f0.tobytes()
    first = {start.raw: start}
    q, dups, conflicts, t_end = [start], 0, 0, time.time() + time_limit
    while q and dups < max_dups and time.time() < t_end:
        nq = []
        for n in q:
            for a in acts:
                m = raw_step(n, a)
                if m is None or m.won or m.lost:
                    continue
                if m.raw not in first:
                    first[m.raw] = m
                    nq.append(m)
                    continue
                dups += 1
                o = first[m.raw]
                for b in acts:
                    x, y = raw_step(m, b), raw_step(o, b)
                    if x is not None and y is not None and (x.raw != y.raw or x.won != y.won):
                        conflicts += 1
                        break
                if dups >= max_dups:
                    break
            if dups >= max_dups:
                break
        q = nq
    return {"duplicate_histories_checked": dups, "conflicts": conflicts}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", default="g50t,ka59,lf52,sb26,sc25,su15,tn36")
    p.add_argument("--time", type=float, default=120.0)
    p.add_argument("--nodes", type=int, default=20000)
    p.add_argument("--out", default=str(ROOT / "runs" / "algebra_on_games" / "exhausted_witness.json"))
    a = p.parse_args()
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(ENV_DIR), logger=quiet())
    wanted = a.games.split(",")
    results = {}
    for gid in [x for x in list_games() if x.split("-")[0] in wanted]:
        g = Game(arc, gid)
        A = [("key", k) for k in (1, 2, 3, 4, 5) if k in g.avail]
        if 6 in g.avail:
            A += effective_clicks(g)
        r = {"original_actions": len(A), "raw_markov_original_actions": raw_markov(arc, g, A) if A else None}
        segs = level1_segments(gid)
        verified = []
        for s in segs:
            _, info = g.play(s["actions"])
            s["replay_completes_level"] = info["level_up"]
            if info["level_up"]:
                verified.append(s)
        r["b0_segments"] = [{"pass": s["pass"], "len": len(s["actions"]), "replay_ok": s["replay_completes_level"]}
                            for s in segs]
        if verified:
            s = min(verified, key=lambda s: len(s["actions"]))
            seg_acts = list(dict.fromkeys(s["actions"]))
            missing = [x for x in seg_acts if x not in A]
            A2 = A + missing
            r["witness_segment"] = {"pass": s["pass"], "len": len(s["actions"]),
                                    "distinct_actions": len(seg_acts), "not_in_original_set": len(missing),
                                    "missing_examples": [list(x) for x in missing[:6]]}
            r["bfs_augmented"] = {k: run_bfs(arc, g, A2, k, a.time, a.nodes) for k in ("masked", "raw")}
        results[gid] = r
        print(gid, json.dumps({k: v for k, v in r.items() if k != "b0_segments"}), flush=True)
        Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
