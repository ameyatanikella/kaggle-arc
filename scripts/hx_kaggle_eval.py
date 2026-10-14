"""Analyse the counterbalanced Kaggle paired runs B0 vs B0+HX (docs/prereg_hx.md, LLM host).

Inputs: runs/kaggle/pair_hx_ba and runs/kaggle/pair_hx_ab, each with arm_a (B0) and arm_b (B0+HX):
benchmark.json, arm_meta.json, *_requests.jsonl.
Outputs (stdout + runs/hx/kaggle_eval.json):
  * per session and pooled: per-game paired delta of score and levels (game-level bootstrap, 90% CI);
  * order effect (first vs second position);
  * HX telemetry: triggers fired, reports delivered, report characters, requests, errors;
  * tool use in model code: calls of hx_* helpers, `moves`, `.kind`;
  * throughput: completion tokens per arm (from the harness's generated-token counts);
  * the largest per-game gains and regressions (for transcript inspection).
Public games are contaminated; these are descriptive. Differences under ~8 points are not effects.
"""
from __future__ import annotations

import glob
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "pair_hx"          # e.g. pair_hx2q for the quiet-ledger runs
SESSIONS = {"ba": ROOT / f"runs/kaggle/{PREFIX}_ba", "ab": ROOT / f"runs/kaggle/{PREFIX}_ab"}


def load_arm(d: Path) -> dict:
    b = json.loads((d / "benchmark.json").read_text())
    per = {}
    tokens = 0
    for r in b["game_runs"]:
        g = per.setdefault(r["game_id"], {"score": [], "levels": []})
        g["score"].append(float(r.get("final_score") or 0.0))
        g["levels"].append(int(r.get("levels_completed") or 0))
        tokens += sum(int(h.get("generated_tokens") or 0) for h in r.get("history") or [])
        note = str(r.get("solver_note") or "")
    meta = json.loads((d / "arm_meta.json").read_text()) if (d / "arm_meta.json").exists() else {}
    return {"per": per, "gen_tokens_in_actions": tokens, "meta": meta}


def tool_use(d: Path) -> dict:
    codes = {}
    for f in glob.glob(str(d / "*_requests.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            for m in r.get("messages", []):
                if m.get("role") != "assistant":
                    continue
                for tc in m.get("tool_calls") or []:
                    try:
                        c = json.loads(tc["function"]["arguments"]).get("code", "")
                    except Exception:  # noqa: BLE001
                        c = ""
                    if c:
                        codes[hash(c)] = c
    cnt = Counter()
    for c in codes.values():
        for k in ("hx_check(", "hx_probe(", "hx_goals(", "hx_relations(", "hx_wins(", "hx_ledger(", "moves", ".kind"):
            if k in c:
                cnt[k] += 1
    cnt["unique_tool_calls"] = len(codes)
    return dict(cnt)


def paired(A: dict, B: dict, key: str, boot: int = 10000) -> dict:
    games = sorted(set(A) & set(B))
    d = np.array([np.mean(B[g][key]) - np.mean(A[g][key]) for g in games])
    rng = random.Random(0)
    bs = [d[[rng.randrange(len(d)) for _ in d]].mean() for _ in range(boot)]
    lo, hi = np.percentile(bs, [5, 95])
    return {"games": len(games), "mean_a": round(float(np.mean([np.mean(A[g][key]) for g in games])), 2),
            "mean_b": round(float(np.mean([np.mean(B[g][key]) for g in games])), 2),
            "delta": round(float(d.mean()), 2), "ci90": [round(float(lo), 2), round(float(hi), 2)],
            "per_game": {g: round(float(x), 1) for g, x in zip(games, d)}}


def merge(*arms) -> dict:
    out = {}
    for a in arms:
        for g, v in a.items():
            m = out.setdefault(g, {"score": [], "levels": []})
            m["score"] += v["score"]
            m["levels"] += v["levels"]
    return out


def main():
    res = {}
    arms = {}
    for s, d in SESSIONS.items():
        if not (d / "arm_a" / "benchmark.json").exists():
            print("missing", d)
            continue
        A, B = load_arm(d / "arm_a"), load_arm(d / "arm_b")
        arms[s] = (A, B)
        res[s] = {"score": paired(A["per"], B["per"], "score"), "levels": paired(A["per"], B["per"], "levels"),
                  "tokens": [A["gen_tokens_in_actions"], B["gen_tokens_in_actions"]],
                  "hx": (B["meta"].get("b2_summary") or {}).get("hx"), "tool_use_hx_arm": tool_use(d / "arm_b"),
                  "tool_use_b0_arm": tool_use(d / "arm_a")}
        r = res[s]
        print(f"== session {s} (first arm: {'HX' if s == 'ba' else 'B0'})")
        print(f"  score B0 {r['score']['mean_a']} HX {r['score']['mean_b']} delta {r['score']['delta']:+.2f} CI90 {r['score']['ci90']}")
        print(f"  levels/run B0 {r['levels']['mean_a']} HX {r['levels']['mean_b']} delta {r['levels']['delta']:+.2f} CI90 {r['levels']['ci90']}")
        print(f"  generated tokens (attributed to actions) B0 {r['tokens'][0]:,} HX {r['tokens'][1]:,}")
        print(f"  hx telemetry {json.dumps(r['hx'])[:600]}")
        print(f"  tool use HX arm {r['tool_use_hx_arm']}  B0 arm {r['tool_use_b0_arm']}")
    if len(arms) == 2:
        A = merge(arms["ba"][0]["per"], arms["ab"][0]["per"])
        B = merge(arms["ba"][1]["per"], arms["ab"][1]["per"])
        pooled = {"score": paired(A, B, "score"), "levels": paired(A, B, "levels")}
        res["pooled"] = pooled
        first = merge(arms["ba"][1]["per"], arms["ab"][0]["per"])
        second = merge(arms["ba"][0]["per"], arms["ab"][1]["per"])
        res["order_effect_first_minus_second"] = paired(second, first, "score")["delta"]
        p = pooled["score"]
        print(f"== pooled (4 passes per arm): score B0 {p['mean_a']} HX {p['mean_b']} delta {p['delta']:+.2f} CI90 {p['ci90']}")
        print(f"   levels delta {pooled['levels']['delta']:+.2f} CI90 {pooled['levels']['ci90']}; "
              f"order effect (first - second) {res['order_effect_first_minus_second']:+.2f}")
        pg = sorted(p["per_game"].items(), key=lambda kv: kv[1])
        print("   largest regressions:", pg[:4])
        print("   largest gains:", pg[-4:])
        rule = p["delta"] >= -2 and any((res[s].get("hx") or {}).get("reports_delivered", 0) > 0 for s in ("ba", "ab"))
        res["prereg_lb_rule_pass"] = bool(rule)
        print(f"   pre-registered leaderboard rule (pooled delta >= -2 and triggers fired): {'PASS' if rule else 'FAIL'}")
    (ROOT / f"runs/hx/kaggle_eval_{PREFIX}.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
