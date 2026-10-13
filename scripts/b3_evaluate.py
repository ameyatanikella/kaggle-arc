"""Evaluate the registered B3 development comparison (runs/b3/dev/<version>) against docs/prereg_b3.md."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "runs" / "b3" / "dev"
VERSIONS = ["B1", "V1", "V2", "V3", "V4", "V4b", "V5", "V5s"]
T = ("t_perception", "t_inference", "t_search", "t_validation", "t_experiments", "t_policy", "t_cache")


def load(v):
    p = DEV / v / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


def summarize(rs):
    seeds = sorted({r["seed"] for r in rs})
    per_seed = {s: [r for r in rs if r["seed"] == s] for s in seeds}
    out = {"runs": len(rs), "errors": sum(1 for r in rs if r.get("error")),
           "refusals": sum(len(r.get("refused", [])) for r in rs),
           "levels": sum(r.get("levels_completed", 0) for r in rs),
           "mean_score": float(np.mean([np.mean([r.get("score", 0) for r in v]) for v in per_seed.values()])),
           "levels_per_seed": [sum(r.get("levels_completed", 0) for r in v) for v in per_seed.values()],
           "stop_reasons": dict(sorted({k: sum(1 for r in rs if r.get("stop_reason") == k)
                                        for k in {r.get("stop_reason") for r in rs}}.items()))}
    apl, contrib = [], []
    for r in rs:
        la = r.get("official_level_actions") or []
        sc = r.get("level_scores") or []
        for i in range(r.get("levels_completed", 0)):
            if i < len(la):
                apl.append(la[i])
            if i < len(sc) and sc[i] > 0:
                contrib.append((r["game_id"][:4], r["seed"], i + 1, sc[i]))
    out["actions_per_completed_level_median"] = float(np.median(apl)) if apl else None
    out["completions_contributing_score"] = sorted(contrib)
    st = defaultdict(float)
    for r in rs:
        for k, v in (r.get("agent_stats") or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                st[k] += v
    out["time_split_s"] = {k: round(st.get(k, 0.0), 3) for k in T}
    out["decision_time_s"] = round(sum(st.get(k, 0.0) for k in T if k != "t_cache"), 3)
    for k in ("decisions", "bfs_calls", "bfs_expansions", "plans_started", "plan_steps_followed", "plan_breaks",
              "cache_built", "cache_hits", "cache_misses", "cache_invalidated", "cache_validation_steps",
              "cache_rejected_on_validation", "cache_preserved_on_refine", "neg_built", "neg_hits", "neg_invalidated",
              "cache_dup_words", "predictions", "pred_correct", "pred_change_tp", "pred_change_fp", "pred_change_fn",
              "graph_conflicts", "graph_refinements", "graph_unresolved", "graph_rebuilds", "allowance_returns",
              "tier_expansions", "fallback_actions", "resets", "exp_started", "exp_valid_verified", "exp_valid_assumed",
              "exp_distinguishing", "exp_hypotheses_eliminated", "exp_novel_obs", "exp_fatal", "exp_overhead_actions",
              "rules_learned", "rules_used_ok", "rules_used_fail", "goal_plans_stored", "goal_reuse_started",
              "goal_reuse_steps_ok", "goal_reuse_completed", "goal_reuse_invalidated"):
        out[k] = st.get(k, 0)
    out["expansions_per_decision"] = out["bfs_expansions"] / max(1, out["decisions"])
    out["prediction_accuracy"] = out["pred_correct"] / max(1, out["predictions"])
    out["change_precision"] = out["pred_change_tp"] / max(1, out["pred_change_tp"] + out["pred_change_fp"])
    out["change_recall"] = out["pred_change_tp"] / max(1, out["pred_change_tp"] + out["pred_change_fn"])
    return out


def per_game_levels(rs):
    d = defaultdict(dict)
    for r in rs:
        d[r["game_id"][:4]][r["seed"]] = r.get("levels_completed", 0)
    return d


def main():
    R = {v: load(v) for v in VERSIONS}
    S = {v: summarize(rs) for v, rs in R.items() if rs}
    for v, s in S.items():
        print(f"{v:4s} levels={s['levels']:3d} {s['levels_per_seed']} score={s['mean_score']:.3f} "
              f"apl_med={s['actions_per_completed_level_median']} exp/dec={s['expansions_per_decision']:.2f} "
              f"dec_time={s['decision_time_s']:.1f}s cache(b/h/inv)={s['cache_built']:.0f}/{s['cache_hits']:.0f}/"
              f"{s['cache_invalidated']:.0f} neg(b/h/inv)={s['neg_built']:.0f}/{s['neg_hits']:.0f}/{s['neg_invalidated']:.0f} "
              f"pred_acc={s['prediction_accuracy']:.3f} chg_prec={s['change_precision']:.3f} errors={s['errors']}")
    G = {v: per_game_levels(rs) for v, rs in R.items() if rs}
    crit = {}
    if "V1" in S and "V3" in S:
        a, b = S["V1"], S["V3"]
        red = 1 - b["expansions_per_decision"] / max(1e-9, a["expansions_per_decision"])
        crit["1_plan_reuse_V3_vs_V1"] = {"expansion_reduction": round(red, 3),
                                          "decision_time": (a["decision_time_s"], b["decision_time_s"]),
                                          "levels": (a["levels"], b["levels"]), "score": (a["mean_score"], b["mean_score"]),
                                          "pass": red >= 0.5 and b["decision_time_s"] < a["decision_time_s"]
                                          and b["levels"] >= a["levels"] - 1 and b["mean_score"] >= a["mean_score"] - 0.5}
    if "V2" in S and "V3" in S:
        a, b = S["V2"], S["V3"]
        cost = lambda s: s["time_split_s"]["t_cache"] + s["time_split_s"]["t_search"]
        crit["1b_unique_advantage_V3_vs_V2"] = {"cache+search_s": (round(cost(a), 3), round(cost(b), 3)),
                                                "expansions": (a["bfs_expansions"], b["bfs_expansions"]),
                                                "levels": (a["levels"], b["levels"]),
                                                "advantage": (cost(b) < cost(a) or b["bfs_expansions"] < a["bfs_expansions"])
                                                and b["levels"] >= a["levels"]}
    for new, old, key in (("V4", "V3", "2_temporal_V4_vs_V3"), ("V4b", "V4", "2b_goal_reuse_V4b_vs_V4")):
        if new in S and old in S:
            a, b = S[old], S[new]
            hidden = sum(sum(G[new].get(g, {}).values()) - sum(G[old].get(g, {}).values()) for g in ("g50t", "sc25"))
            crit[key] = {"levels": (a["levels"], b["levels"]), "hidden_state_games_level_delta": hidden,
                         "pass": (b["levels"] >= a["levels"] + 1 or hidden > 0) and b["levels"] >= a["levels"] - 1}
    for new in ("V5", "V5s"):
        if new in S and "V4" in S:
            a, b = S["V4"], S[new]
            crit[f"3_experiments_{new}_vs_V4"] = {"levels": (a["levels"], b["levels"]),
                                                  "score": (round(a["mean_score"], 3), round(b["mean_score"], 3)),
                                                  "retain": (b["levels"] > a["levels"] or b["mean_score"] > a["mean_score"] + 0.5)
                                                  and b["levels"] >= a["levels"] - 1}
    if "V5" in S and "V5s" in S:
        a, b = S["V5s"], S["V5"]
        per_act = lambda s: (s["exp_valid_verified"] + s["exp_valid_assumed"]) / max(1, s["decisions"])
        crit["3b_shared_vs_separate"] = {"valid_distinguishing": (a["exp_distinguishing"], b["exp_distinguishing"]),
                                         "levels": (a["levels"], b["levels"]),
                                         "shared_wins": b["exp_distinguishing"] >= a["exp_distinguishing"]
                                         and b["levels"] >= a["levels"]}
    # global regressions vs V1: games V1 completes a level on every seed but the variant does not
    reg = {}
    if "V1" in G:
        sure = {g for g, d in G["V1"].items() if d and min(d.values()) >= 1}
        for v in G:
            if v in ("V1", "B1"):
                continue
            lost = [g for g in sure if min(G[v].get(g, {0: 0}).values()) < min(G["V1"][g].values())]
            reg[v] = sorted(lost)
    crit["6_regressions_vs_V1"] = reg
    crit["6_errors"] = {v: S[v]["errors"] for v in S}
    dup = {v: (S[v]["cache_dup_words"], S[v]["cache_built"]) for v in S if S[v]["cache_built"]}
    crit["4_dedup_profile_dup_words_of_built"] = dup
    for k, v in crit.items():
        print(k, v)
    (DEV / "evaluation.json").write_text(json.dumps({"summary": S, "per_game_levels": G, "criteria": crit},
                                                    indent=1, default=str))


if __name__ == "__main__":
    main()
