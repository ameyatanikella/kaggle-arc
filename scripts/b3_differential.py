"""Differential check of B3's state abstraction on recorded states (no agent decisions).

B0's recorded level-1 actions (data/franzen_v3_output/transcripts) are replayed offline. Frames
pass through B3's perception (VolatileMask) and TemporalGraph with temporal refinement off
(frame-keyed, as B1) and on. For every step whose (node, action) edge was already known and
deterministic, the predicted successor is compared with the observed one. Reported:
  prediction accuracy, change precision/recall, conflicts, refinements, unresolved alternatives.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from diag_shadow import ENV_DIR, arc_agi, level1_actions, list_games, quiet, replay  # noqa: E402
from arc3.agents.b3.graph import TemporalGraph  # noqa: E402
from arc3.perception import VolatileMask  # noqa: E402


def run(steps, temporal):
    vm = VolatileMask()
    for s in steps:                                      # learn the HUD mask from the trace first
        vm.update(s["before"], s["after"])
    g = TemporalGraph(temporal)
    hist, att = (), -1
    r = {"pred": 0, "correct": 0, "tp": 0, "fp": 0, "fn": 0}
    for s in steps:
        if s["attempt"] != att:
            att, hist = s["attempt"], ()
        a = s["a"]
        mk0 = g.mk_id(vm.key(s["before"]))
        mk1 = g.mk_id(vm.key(s["after"]))
        node = g.node(mk0, hist)
        pred = g.det_successor(node, a)
        res = g.add(mk0, hist, a, mk1, False)
        changed = mk1 != mk0
        if pred is not None:
            r["pred"] += 1
            r["correct"] += int(pred == res["post"])
            pc = pred[0] != mk0
            r["tp"] += int(pc and changed)
            r["fp"] += int(pc and not changed)
            r["fn"] += int(changed and not pc)
        hist = (hist + ((a, changed),))[-7:]
    r.update(conflicts=g.conflicts, refinements=g.refinements, unresolved=g.unresolved, nodes=len(g.edges))
    return r


def main():
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(ENV_DIR), logger=quiet())
    out = {}
    for gid in list_games():
        _, acts = level1_actions(gid)
        if not acts:
            continue
        _, steps = replay(arc, gid, acts)
        if len(steps) < 3:
            continue
        out[gid] = {"steps": len(steps), "frame_keyed": run(steps, False), "temporal": run(steps, True)}
        f, t = out[gid]["frame_keyed"], out[gid]["temporal"]
        print(f"{gid[:4]} steps={len(steps):3d} | frame-keyed acc {f['correct']}/{f['pred']} conflicts {f['conflicts']} "
              f"| temporal acc {t['correct']}/{t['pred']} conflicts {t['conflicts']} refine {t['refinements']} "
              f"unresolved {t['unresolved']}", flush=True)
    tot = {m: {k: sum(v[m][k] for v in out.values()) for k in ("pred", "correct", "tp", "fp", "fn", "conflicts",
                                                              "refinements", "unresolved")}
           for m in ("frame_keyed", "temporal")}
    for m, t in tot.items():
        prec = t["tp"] / max(1, t["tp"] + t["fp"])
        rec = t["tp"] / max(1, t["tp"] + t["fn"])
        print(f"TOTAL {m}: accuracy {t['correct']}/{t['pred']} = {t['correct'] / max(1, t['pred']):.3f}; "
              f"change precision {prec:.3f} recall {rec:.3f}; conflicts {t['conflicts']} refinements "
              f"{t['refinements']} unresolved {t['unresolved']}")
    p = ROOT / "runs" / "b3" / "differential.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"per_game": out, "total": tot}, indent=1))


if __name__ == "__main__":
    main()
