"""Evaluate runs/b3f/eval/{b3,b3_franzen,franzen} against docs/prereg_b3_franzen.md."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arc3.scoring import game_score  # noqa: E402

E = ROOT / "runs" / "b3f" / "eval"
ARMS = ("b3", "b3_franzen", "franzen")


def load(arm):
    """Per-game results come from the harness's '[finished]' log lines (benchmark.json is not written
    under minimal diagnostics, the competition-rerun setting); attribution from b3_session files."""
    import re
    d = E / arm
    runs = []
    for line in (E / f"{arm}.log").read_text(errors="replace").splitlines():
        m = re.match(r"\[finished\] (\S+) state=(\S+) level=(\d+)/(\d+) score=([\d.]+) actions=(\d+) tokens=(\d+) "
                     r"per-level=(\S+)", line)
        if not m:
            continue
        pl = [x.split("/") for x in m.group(8).strip(",").split(",") if "/" in x]
        runs.append({"game_id": m.group(1), "state": m.group(2), "levels_completed": int(m.group(3)),
                     "number_of_levels": int(m.group(4)), "final_score": float(m.group(5)),
                     "actions_total": int(m.group(6)), "tokens": int(m.group(7)),
                     "actions_per_level": [int(a) for a, b in pl if a.isdigit()],
                     "base_actions_per_level": [int(b) for a, b in pl if b.isdigit()]})
    sess = {}
    if (d / "b3_session").exists():
        for p_ in (d / "b3_session").glob("*.json"):
            s_ = json.loads(p_.read_text())
            sess[s_["game_id"]] = s_
    req = []
    rp = d / "requests.jsonl"
    if rp.exists():
        req = [json.loads(x) for x in rp.read_text().splitlines() if x.strip()]
    chan = json.loads((d / "b3_channel_stats.json").read_text()) if (d / "b3_channel_stats.json").exists() else {}
    return runs, sess, req, chan


def main():
    out = {}
    for arm in ARMS:
        runs, sess, req, chan = load(arm)
        games = {}
        for r in runs:
            g = r["game_id"]
            lc = int(r.get("levels_completed") or 0)
            apl = (r.get("actions_per_level") or [])[:lc]
            base = (r.get("base_actions_per_level") or [])
            sdk = game_score(base, apl) if apl and len(base) >= lc else r.get("final_score", 0.0)
            s = sess.get(g, {})
            note = str(r.get("solver_note") or "")
            games[g[:4]] = {
                "levels": lc, "sdk_score": round(sdk, 3), "taaf_final_score": r.get("final_score"),
                "actions": int(r.get("actions_total") or 0), "actions_per_completed_level": apl,
                "later_levels": max(0, lc - 1), "state": r.get("state"), "tokens": r.get("tokens"),
                "error": "crashed" if r.get("state") == "crashed" else None,
                "actions_by_source": s.get("actions_by_source"), "completions_by_source": s.get("level_completions_by_source"),
                "llm_turns": s.get("llm_turns"), "turn_triggers": s.get("turn_triggers"),
                "llm_wall_s": round((s.get("wall_seconds") or {}).get("llm_turns", 0.0), 1),
                "revisions": len(s.get("revisions") or []),
                "failure": ("no level 1" if lc == 0 else f"stalled at level {lc + 1}"),
            }
        resp = [x for x in req if x.get("event") == "response"]
        tot = {
            "games": len(games), "levels": sum(v["levels"] for v in games.values()),
            "later_levels": sum(v["later_levels"] for v in games.values()),
            "sdk_score_mean": round(sum(v["sdk_score"] for v in games.values()) / max(1, len(games)), 3),
            "actions": sum(v["actions"] for v in games.values()),
            "errors": sum(1 for v in games.values() if v["error"]),
            "llm_responses": len(resp),
            "completion_tokens": sum((x.get("usage") or {}).get("completion_tokens", 0) for x in resp),
            "prompt_tokens": sum((x.get("usage") or {}).get("prompt_tokens", 0) for x in resp),
            "cached_prompt_tokens": sum(((x.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                                        for x in resp),
            "llm_wall_s": round(sum(v["llm_wall_s"] for v in games.values()), 1),
            "b3_channel": chan,
        }
        src = Counter()
        comp = Counter()
        for v in games.values():
            for k, n in (v["actions_by_source"] or {}).items():
                src[k] += n
            for lvl, s_ in (v["completions_by_source"] or []):
                comp[s_] += 1
        tot["actions_by_source"] = dict(src)
        tot["completions_by_source"] = dict(comp)
        out[arm] = {"total": tot, "per_game": games}
    a1, a2 = out["b3"]["total"], out["b3_franzen"]["total"]
    reasoning = sum(n for s_, n in a2["completions_by_source"].items() if s_ in ("llm", "b3_tool"))
    crit = {
        "a_levels_b3f_gt_b3": (a2["levels"], a1["levels"], a2["levels"] > a1["levels"]),
        "b_later_levels_b3f_ge_b3": (a2["later_levels"], a1["later_levels"], a2["later_levels"] >= a1["later_levels"]),
        "c_completions_by_reasoning_sources": (reasoning, reasoning >= 1),
        "d_no_new_error_class": (a2["errors"], a1["errors"], a2["errors"] <= a1["errors"]),
    }
    crit["adopt_integration"] = all(v[-1] for v in crit.values())
    crit["context_franzen_ge_b3f"] = out["franzen"]["total"]["levels"] >= a2["levels"]
    crit["context_b3_ge_b3f"] = a1["levels"] >= a2["levels"]
    out["criteria"] = crit
    for arm in ARMS:
        t = out[arm]["total"]
        print(f"{arm:11s} levels={t['levels']:3d} later={t['later_levels']:2d} sdk_mean={t['sdk_score_mean']:.3f} "
              f"actions={t['actions']:6d} errors={t['errors']} llm_resp={t['llm_responses']} llm_wall={t['llm_wall_s']}s "
              f"by_source={t['actions_by_source']} completions={t['completions_by_source']}")
        for g, v in sorted(out[arm]["per_game"].items()):
            print(f"    {g} lv={v['levels']} sdk={v['sdk_score']:.3f} act={v['actions']} apl={v['actions_per_completed_level']} "
                  f"{v['failure']} turns={v['llm_turns']} triggers={v['turn_triggers']} comp={v['completions_by_source']}")
    print(json.dumps(crit, indent=1))
    (E / "evaluation.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
