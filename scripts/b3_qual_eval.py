"""Evaluate runs/b3/qual against docs/qualification_b3.md (paired, game-level bootstrap; exclusive timing)."""
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
Q = ROOT / "runs" / "b3" / "qual"
CATS = ("perception", "inference", "search", "cache", "policy_other")


def load(v):
    return json.loads((Q / v / "results.json").read_text())


def per_game(rs, key):
    d = defaultdict(list)
    for r in rs:
        d[r["game_id"]].append(r.get(key, 0) or 0)
    return {g: float(np.mean(v)) for g, v in d.items()}


def paired(a, b, boot=10000, seed=0):
    gs = sorted(a)
    d = np.array([b[g] - a[g] for g in gs])
    rng = random.Random(seed)
    bs = [float(np.mean([d[rng.randrange(len(d))] for _ in d])) for _ in range(boot)]
    return float(d.mean()), float(np.percentile(bs, 5)), float(np.percentile(bs, 95))


def timing(rs):
    out = {"runs": len(rs), "decisions": 0, "time_cap_runs": [r["game_id"][:4] + f"/s{r['seed']}" for r in rs
                                                               if r.get("stop_reason") == "time_cap"]}
    for k in ("agent_seconds", "engine_seconds", "wall_seconds"):
        out[k] = round(sum(r.get(k, 0) or 0 for r in rs), 2)
    for c in CATS:
        out[f"cpu_{c}"] = round(sum((r.get("agent_stats") or {}).get(f"cpu_{c}", 0) for r in rs), 3)
        out[f"wall_{c}"] = round(sum((r.get("agent_stats") or {}).get(f"wall_{c}", 0) for r in rs), 3)
    out["decisions"] = int(sum((r.get("agent_stats") or {}).get("decisions", 0) for r in rs))
    out["bfs_expansions"] = int(sum((r.get("agent_stats") or {}).get("bfs_expansions", 0) for r in rs))
    out["cpu_search_cache_us_per_decision"] = round(1e6 * (out["cpu_search"] + out["cpu_cache"]) / max(1, out["decisions"]), 2)
    out["cpu_agent_us_per_decision"] = round(1e6 * sum(out[f"cpu_{c}"] for c in CATS) / max(1, out["decisions"]), 2)
    out["agent_wall_ms_per_decision"] = round(1e3 * out["agent_seconds"] / max(1, out["decisions"]), 4)
    out["engine_ms_per_action"] = round(1e3 * out["engine_seconds"] / max(1, sum(r.get("actions", 0) for r in rs)), 4)
    return out


def main():
    R = {v: load(v) for v in ("B1", "B3nocache", "B3c")}
    res = {}
    for v, rs in R.items():
        res[v] = {"levels": sum(r.get("levels_completed", 0) for r in rs),
                  "rhae_mean": float(np.mean([r.get("score", 0) for r in rs])),
                  "wins": sum(1 for r in rs if r.get("stop_reason") == "win"),
                  "errors": sum(1 for r in rs if r.get("error")),
                  "refusals": sum(len(r.get("refused", [])) for r in rs),
                  "stop_reasons": dict(sorted({k: sum(1 for r in rs if r.get("stop_reason") == k)
                                               for k in {r.get("stop_reason") for r in rs}}.items())),
                  "timing": timing(rs) if v != "B1" else {"runs": len(rs), "agent_seconds": round(sum(r.get("agent_seconds", 0) for r in rs), 2),
                                                          "engine_seconds": round(sum(r.get("engine_seconds", 0) for r in rs), 2)}}
    for a, b in (("B1", "B3c"), ("B3nocache", "B3c"), ("B1", "B3nocache")):
        lv = paired(per_game(R[a], "levels_completed"), per_game(R[b], "levels_completed"))
        sc = paired(per_game(R[a], "score"), per_game(R[b], "score"))
        new_err = res[b]["errors"] > res[a]["errors"]
        overrun = len([r for r in R[b] if r.get("stop_reason") == "time_cap"]) > len([r for r in R[a] if r.get("stop_reason") == "time_cap"])
        res[f"{b}_vs_{a}"] = {"levels_delta_per_game": lv, "rhae_delta": sc, "new_error_class": new_err, "runtime_overrun": overrun,
                              "qualifies": lv[0] >= 0 and not new_err and not overrun and sc[0] >= -1.0}
    print(json.dumps(res, indent=1))
    (Q / "evaluation.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
