"""Baseline record for the HX study (step 1): the published harness run unchanged.

Sources (all existing artifacts; no new GPU run):
  * Franzen v3 public notebook output: data/franzen_v3_output/benchmark.json (4 passes x 25 games)
  * our B0 arm of pair-b2c (first position in its session): runs/kaggle/pair_b2c/arm_b0/benchmark.json (2 passes)

Per run: score (harness final_score, the official formula), levels completed, actions per
completed level vs human baseline, generated tokens, elapsed wall time, terminal state, and a
failure-mode label for the level the run ended on. Failure labels are computed from counts only:
  won            - every level cleared
  stall_think    - unfinished level, >= 30K tokens spent on it and tokens/action > 1200
                   (deliberates instead of acting)
  stall_act      - unfinished level, >= 30K tokens and tokens/action <= 1200 (acts but no progress)
  starved        - unfinished level with < 30K tokens (the scheduler gave it little compute)
usage: python scripts/hx_baseline.py [--out runs/hx/baseline.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arc3.scoring import game_score  # noqa: E402

SOURCES = {
    "franzen_v3": ROOT / "data/franzen_v3_output/benchmark.json",
    "b0_pair_b2c": ROOT / "runs/kaggle/pair_b2c/arm_b0/benchmark.json",
}


def run_record(r: dict) -> dict:
    base = list(r.get("base_actions_per_level") or [])
    apl = list(r.get("actions_per_level") or [])
    done = int(r.get("levels_completed") or 0)
    hist = r.get("history") or []
    toks = [int(h.get("generated_tokens") or 0) for h in hist]
    total_tokens = sum(toks)
    note = str(r.get("solver_note") or "")
    if note.startswith("tokens="):
        try:
            total_tokens = max(total_tokens, int(note.split("=", 1)[1]))
        except ValueError:
            pass
    # tokens per level: actions are consumed in order, actions_per_level[i] of them on level i
    per_level_tokens, i, k = [], 0, 0
    for lv, n in enumerate(apl):
        per_level_tokens.append(sum(toks[k:k + int(n)]))
        k += int(n)
    tail_tokens = sum(toks[k:])          # tokens after the last counted action (no action followed)
    completed_actions = [int(x) for x in apl[:done]]
    rescore = game_score(base, completed_actions) if base else None
    rec = {
        "game_id": r["game_id"], "state": r.get("state"), "n_levels": int(r.get("number_of_levels") or len(base)),
        "levels_completed": done, "score": float(r.get("final_score") or 0.0), "score_replica": rescore,
        "actions_total": int(sum(int(x) for x in apl)), "actions_per_completed_level": completed_actions,
        "baseline_per_completed_level": base[:done],
        "ratio_per_completed_level": [round(a / b, 3) if b else None for a, b in zip(completed_actions, base)],
        "tokens_total": total_tokens, "wall_seconds": float(r.get("final_wallclock_seconds") or 0.0),
    }
    if done < rec["n_levels"]:
        fl_actions = int(apl[done]) if done < len(apl) else 0
        fl_tokens = (per_level_tokens[done] if done < len(per_level_tokens) else 0) + tail_tokens
        rec["final_level"] = done + 1
        rec["final_level_actions"] = fl_actions
        rec["final_level_baseline"] = base[done] if done < len(base) else None
        rec["final_level_tokens"] = fl_tokens
        tpa = fl_tokens / max(1, fl_actions)
        rec["final_level_tokens_per_action"] = round(tpa, 1)
        if fl_tokens < 30000:
            rec["failure"] = "starved"
        elif tpa > 1200:
            rec["failure"] = "stall_think"
        else:
            rec["failure"] = "stall_act"
    else:
        rec["failure"] = "won"
    return rec


def summarize(recs: list[dict], n_passes: int) -> dict:
    games = sorted({r["game_id"] for r in recs})
    per_game = {}
    for g in games:
        rs = [r for r in recs if r["game_id"] == g]
        per_game[g] = {
            "score_mean": round(float(np.mean([r["score"] for r in rs])), 2),
            "score_best": round(float(max(r["score"] for r in rs)), 2),
            "levels": [r["levels_completed"] for r in rs], "n_levels": rs[0]["n_levels"],
            "tokens_k": [round(r["tokens_total"] / 1000) for r in rs],
            "failures": [r["failure"] for r in rs],
        }
    ratios = [x for r in recs for x in r["ratio_per_completed_level"] if x]
    by_pass = {}
    for idx, r in enumerate(recs):
        by_pass.setdefault(r.get("pass_index", 0), []).append(r["score"])
    fails = {}
    for r in recs:
        fails[r["failure"]] = fails.get(r["failure"], 0) + 1
    return {
        "runs": len(recs), "games": len(games), "passes": n_passes,
        "mean_score": round(float(np.mean([r["score"] for r in recs])), 2),
        "best_pass_per_game_mean": round(float(np.mean([per_game[g]["score_best"] for g in games])), 2),
        "pass_means": {k: round(float(np.mean(v)), 2) for k, v in sorted(by_pass.items())},
        "levels_total": int(sum(r["levels_completed"] for r in recs)),
        "levels_per_run": round(float(np.mean([r["levels_completed"] for r in recs])), 2),
        "wins": int(sum(r["state"] == "won" for r in recs)),
        "actions_to_baseline_ratio_median": round(float(np.median(ratios)), 3) if ratios else None,
        "actions_per_completed_level_median": float(np.median([a for r in recs for a in r["actions_per_completed_level"]] or [0])),
        "tokens_per_run_median_k": round(float(np.median([r["tokens_total"] for r in recs])) / 1000, 1),
        "wall_seconds_per_run_median": round(float(np.median([r["wall_seconds"] for r in recs])), 0),
        "failure_modes": fails,
        "final_level_tokens_median_k": round(float(np.median([r["final_level_tokens"] for r in recs if "final_level_tokens" in r] or [0])) / 1000, 1),
        "replica_max_abs_diff": round(max(abs((r["score_replica"] or 0) - r["score"]) for r in recs), 4),
        "per_game": per_game,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs/hx/baseline.json"))
    a = ap.parse_args()
    out = {}
    for name, path in SOURCES.items():
        d = json.loads(path.read_text())
        n_passes = int(d.get("n_passes") or 1)
        recs = []
        counts: dict = {}
        for r in d["game_runs"]:
            rec = run_record(r)
            m = re.search(r"_p(\d+)\.html$", str(r.get("solver_analysis_html") or ""))
            rec["pass_index"] = int(m.group(1)) if m else counts.get(r["game_id"], 0)
            counts[r["game_id"]] = counts.get(r["game_id"], 0) + 1
            recs.append(rec)
        out[name] = {"source": str(path.relative_to(ROOT)), "start": d.get("start_time"), "end": d.get("end_time"),
                     "summary": summarize(recs, n_passes), "runs": recs}
        s = out[name]["summary"]
        print(f"== {name}: {s['runs']} runs ({s['passes']} passes) mean {s['mean_score']} "
              f"best-pass/game {s['best_pass_per_game_mean']} pass means {s['pass_means']}")
        print(f"   levels {s['levels_total']} ({s['levels_per_run']}/run), wins {s['wins']}, "
              f"median actions/baseline {s['actions_to_baseline_ratio_median']}, tokens/run {s['tokens_per_run_median_k']}K, "
              f"wall/run {s['wall_seconds_per_run_median']}s, replica max diff {s['replica_max_abs_diff']}")
        print(f"   failure modes {s['failure_modes']}, final-level tokens median {s['final_level_tokens_median_k']}K")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
