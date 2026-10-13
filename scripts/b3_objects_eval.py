"""Component O on recorded traces: commit-to-nearest vs retained correspondence alternatives.

Traces: B0's recorded level-1 actions (as in scripts/diag_shadow.py), replayed offline; frames only.
For the object shape that moves under a directional key, every instance of that exact shape is
tracked. Steps where the number of instances changes restart both trackers.
Registered criterion: usable, model-consistent observations, and contradictions.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from diag_shadow import ENV_DIR, arc_agi, level1_actions, list_games, quiet, replay, sig  # noqa: E402
from arc3.agents.b3.correspond import Retained, commit_nearest  # noqa: E402
from arc3.perception import background_color, components  # noqa: E402
import numpy as np  # noqa: E402


def mover_sig(f0, steps):
    bg = background_color(f0)
    for s in steps:
        if s["a"][0] == "key" and s["a"][1] in (1, 2, 3, 4):
            d = s["before"] != s["after"]
            if not d.any():
                continue
            for c in sorted(components(s["before"], skip_colors=(bg,)), key=lambda c: -c.size):
                if c.size < 2 or not d[c.y0:c.y1 + 1, c.x0:c.x1 + 1].any():
                    continue
                g = sig(s["before"], c)
                if any(sig(s["after"], e) == g and (e.y0, e.x0) != (c.y0, c.x0)
                       for e in components(s["after"], skip_colors=(bg,))):
                    return g, bg
    return None, bg


def main():
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(ENV_DIR), logger=quiet())
    out = {}
    for gid in list_games():
        _, acts = level1_actions(gid)
        if not acts:
            continue
        f0, steps = replay(arc, gid, acts)
        g, bg = mover_sig(f0, steps)
        if g is None:
            continue
        inst = lambda f: sorted((e.y0, e.x0) for e in components(f, skip_colors=(bg,)) if sig(f, e) == g)
        r = {"commit_usable_consistent": 0, "commit_contradictions": 0, "retain_usable": 0,
             "retain_contradictions": 0, "observations": 0, "k_max": 0}
        prev = inst(steps[0]["before"]) if steps else []
        learned = {}
        tracker = None
        for s in steps:
            cur = inst(s["after"])
            a = s["a"][1] if s["a"][0] == "key" else 6
            if not cur or len(cur) != len(prev):
                prev, learned, tracker = cur, {}, None
                continue
            k = len(cur)
            r["k_max"] = max(r["k_max"], k)
            if tracker is None:
                tracker = Retained(k)
                tracker.start(prev)
                cprev = list(prev)
            r["observations"] += k
            # commit-to-nearest: implied displacement per identity vs what it learned before
            cnew = commit_nearest(cprev, cur)
            for j in range(k):
                d = (cnew[j][0] - cprev[j][0], cnew[j][1] - cprev[j][1])
                key = (a, j)
                if d != (0, 0) and key in learned and learned[key] != d:
                    r["commit_contradictions"] += 1
                else:
                    r["commit_usable_consistent"] += 1
                    if d != (0, 0):
                        learned[key] = d
            cprev = cnew
            before = tracker.contradictions
            res = tracker.update(a, cur)
            r["retain_contradictions"] += tracker.contradictions - before
            r["retain_usable"] += sum(p is not None for p in res)
            prev = cur
        if r["observations"]:
            out[gid] = r
            print(gid[:4], r, flush=True)
    tot = {k: sum(v[k] for v in out.values()) for k in next(iter(out.values())) if k != "k_max"}
    amb = {g: v for g, v in out.items() if v["k_max"] >= 2}
    tot_amb = {k: sum(v[k] for v in amb.values()) for k in tot} if amb else {}
    print("ALL", tot)
    print("AMBIGUOUS GAMES (k>=2):", sorted(g[:4] for g in amb), tot_amb)
    p = ROOT / "runs" / "b3" / "objects_eval.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"per_game": out, "total": tot, "ambiguous_total": tot_amb}, indent=1))


if __name__ == "__main__":
    main()
