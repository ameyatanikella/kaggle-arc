"""Classify B3c level-1 failures (runs/b3/qual/B3c) against B0's replay-verified winning paths.

For each game and seed where B3c did not complete level 1, using B0's shortest verified level-1
segment (data/franzen_v3_output/transcripts) as the reference solution:
  * action_not_in_candidates: a step of the reference is a click whose target object (connected
    component under the click) is not hit by any B3c candidate at tier <= 2 in that frame
    (keys and simple actions are always candidates).
  * perception_mask: B3c's learned HUD mask hides pixels that the reference solution changes.
  * hidden_state: B3c observed transition conflicts (the same key and action led to different
    keys) in this run.
  * reached_depth: the deepest reference step whose pre-action state B3c visited (masked keys of
    every frame B3c saw), as a fraction of the solution length.
  * wasted_actions: actions B3c spent in "reset to reach frontier" + "fallback" (no frontier
    reachable from where it was).
Primary category, in order of decisiveness:
  1. action set: some reference action is not in the candidate set;
  2. perception: the mask hides solution pixels;
  3. exploration: the reference path is coverable but B3c never reached its later states;
     subdivided into frontier_remaining_at_cap and exhausted_looping (by wasted actions).
"Wrong goals" does not apply: B3 has no goal model. "Search failure" cannot occur within the known
graph (BFS is exact), so it is reported as 0.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from diag_shadow import ENV_DIR, arc_agi, list_games, quiet, replay  # noqa: E402
from exhausted_witness import level1_segments  # noqa: E402
from arc3.perception import VolatileMask, background_color, components  # noqa: E402

Q = ROOT / "runs" / "b3" / "qual" / (sys.argv[1] if len(sys.argv) > 1 else "B3c")


def mask_from_hex(h):
    return np.unpackbits(np.frombuffer(bytes.fromhex(h), np.uint8))[:4096].reshape(64, 64).astype(bool)


def candidate_points(g, mask, tier=2, max_clicks=48):
    bg = background_color(g)
    comps = [c for c in components(g, skip_colors=(bg,)) if not mask[c.cy, c.cx]]
    comps.sort(key=lambda c: (c.size, c.y0, c.x0))
    pts = set()
    for c in comps:
        if len(pts) < max_clicks:
            pts.add((c.cx, c.cy))
    if tier >= 1:
        pts |= {(x, y) for y in range(2, 62, 4) for x in range(2, 62, 4) if not mask[y, x]}
    if tier >= 2:
        for c in comps:
            ys, xs = np.nonzero(g[c.y0:c.y1 + 1, c.x0:c.x1 + 1] == c.color)
            pts |= {(int(x) + c.x0, int(y) + c.y0) for y, x in zip(ys[::2], xs[::2])}
    return pts


def comp_label(g):
    """Pixel -> component id (non-background components only; background pixels -> -1)."""
    bg = background_color(g)
    lab = -np.ones(g.shape, int)
    for i, c in enumerate(components(g, skip_colors=(bg,))):
        sub = g[c.y0:c.y1 + 1, c.x0:c.x1 + 1] == c.color
        # components() is 4-connected; restrict to this component's pixels by flood fill from a member
        ys, xs = np.nonzero(sub)
        for y, x in zip(ys, xs):
            if lab[c.y0 + y, c.x0 + x] == -1:
                lab[c.y0 + y, c.x0 + x] = i
    return lab


def covered(g, mask, x, y):
    pts = candidate_points(g, mask)
    if (x, y) in pts:
        return True
    lab = comp_label(g)
    t = lab[y, x]
    if t < 0:                                   # background click: only exact grid hits count
        return False
    return any(lab[py, px] == t for px, py in pts)


def replay_wins(arc, gid, acts):
    from arcengine import GameAction
    w = arc.make(gid)
    for a in acts:
        r = w.step(GameAction.ACTION6, data={"x": a[1], "y": a[2]}) if a[0] == "click" else w.step(GameAction.from_id(a[1]))
        if r is None:
            return False
        if r.levels_completed > 0:
            return True
        st = r.state.name if hasattr(r.state, "name") else str(r.state)
        if st == "GAME_OVER":
            return False
    return False


def main():
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(ENV_DIR), logger=quiet())
    out, tally = {}, Counter()
    for gid in list_games():
        runs = sorted(Q.glob(f"seed*/games/{gid}.result.json"))
        failed = []
        for p in runs:
            r = json.loads(p.read_text())
            if r.get("levels_completed", 0) == 0:
                failed.append((p, r))
        if not failed:
            continue
        segs = [s for s in level1_segments(gid)]
        ref = None
        if segs:
            seg = min(segs, key=lambda s: len(s["actions"]))
            f0, steps = replay(arc, gid, seg["actions"])
            if steps and len(steps) == len(seg["actions"]) - 1 or (steps and True):
                ref = (seg["actions"], f0, steps)
        g_out = {"failed_seeds": len(failed), "seeds": len(runs), "reference_len": len(ref[0]) if ref else None,
                 "per_seed": []}
        for p, r in failed:
            s = r["agent_stats"]
            mask = mask_from_hex(s["hud_mask_hex"]) if s.get("hud_mask_hex") else np.zeros((64, 64), bool)
            vm = VolatileMask()
            vm.mask = mask
            frames = np.load(p.with_name(gid + ".frames.npz"))["frames"]
            visited = {vm.key(f.astype(np.uint8)) for f in frames}
            cr = s.get("choice_reasons", {})
            wasted = cr.get("reset to reach frontier", 0) + cr.get("fallback", 0)
            rec = {"seed": r["seed"], "wasted_actions": wasted, "conflicts": s.get("graph_conflicts", 0),
                   "nodes": s.get("graph_nodes", 0), "distinct_frames_seen": len(visited),
                   "choice_reasons": cr}
            if ref:
                acts, f0, steps = ref
                pre = [f.astype(np.uint8) for f in [f0] + [st["after"] for st in steps]]
                n = min(len(acts), len(pre))
                # loop-free reference in B3c's masked-key space
                keys = [vm.key(pre[i]) for i in range(n)]
                path, first = [], {}
                for i in range(n):
                    k = keys[i]
                    if k in first:
                        path = path[:first[k]]
                        first = {keys[j]: m for m, j in enumerate(path)}
                    first[k] = len(path)
                    path.append(i)
                L = [acts[i] for i in path]
                wins = replay_wins(arc, gid, L)
                missing = [i for i in path if acts[i][0] == "click" and not covered(pre[i], mask, acts[i][1], acts[i][2])]
                depth = max([m + 1 for m, i in enumerate(path) if keys[i] in visited], default=0)
                rec.update(loopfree_len=len(L), loopfree_replay_wins=wins, missing_action_steps=missing,
                           reached_depth=depth, solution_len=len(acts))
                if missing:
                    cat = "action_set"
                elif not wins:
                    cat = "perception_or_hidden_state"
                elif depth >= len(L):
                    cat = "reached_final_state_completing_action_untried"
                elif wasted > 1000:
                    cat = "exploration_exhausted_looping"
                else:
                    cat = "exploration_frontier_at_cap"
            else:
                cat = "no_reference_solution"
            if rec["conflicts"] >= 5:
                rec["hidden_state_flag"] = True
            rec["category"] = cat
            tally[cat] += 1
            g_out["per_seed"].append(rec)
        out[gid] = g_out
        cats = Counter(x["category"] for x in g_out["per_seed"])
        ex = g_out["per_seed"][0]
        print(f"{gid[:4]} failed {len(failed)}/{len(runs)} ref_len={g_out['reference_len']} cats={dict(cats)} "
              f"| s{ex['seed']}: loop-free {ex.get('loopfree_len')}/{ex.get('solution_len')} wins={ex.get('loopfree_replay_wins')} "
              f"depth {ex.get('reached_depth')} missing {ex.get('missing_action_steps')} wasted {ex['wasted_actions']} conflicts {ex['conflicts']} "
              f"frames {ex['distinct_frames_seen']}", flush=True)
    print("TALLY (failed game-seeds):", dict(tally))
    (ROOT / "runs" / "b3" / "qual" / ("failure_diag.json" if Q.name == "B3c" else f"failure_diag_{Q.name}.json")).write_text(json.dumps({"per_game": out, "tally": tally}, indent=1, default=str))


if __name__ == "__main__":
    main()
