"""Conservative T3 reach report from both paired Kaggle commit outputs."""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROMPT = "HX stall check (level "
PROBE = re.compile(r"\bhx_probe\s*\(")
GAME_IDS = set("ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 r11l re86 "
               "s5i5 sb26 sc25 sk48 sp80 su15 tn36 tr87 tu93 vc33 wa30".split())


def text_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(x.get("text", "") for x in content if isinstance(x, dict))
    return ""


def reach_file(path: Path) -> dict:
    by_turn = defaultdict(lambda: {"prompt": False, "probe": False})
    seen_calls = set()
    deliveries = probe_snippets = 0
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("event") not in ("request", "response"):
                continue
            step = int(rec.get("analysis_step") or 0)
            row = by_turn[step]
            messages = rec.get("messages") or []
            if rec.get("event") == "request" and rec.get("request_index_within_turn") == 1:
                user_messages = [m for m in messages if m.get("role") == "user"]
                if user_messages:
                    # A no-action re-prompt reuses the analysis step and resets the request index to 1,
                    # so OR within a step (Amendment 1); each such delivery is also counted.
                    delivered = PROMPT in text_content(user_messages[-1].get("content"))
                    row["prompt"] = row["prompt"] or delivered
                    deliveries += delivered
            for msg in messages:
                if msg.get("role") != "assistant":
                    continue
                for call in msg.get("tool_calls") or []:
                    cid = call.get("id")
                    if not cid or cid in seen_calls:
                        continue
                    seen_calls.add(cid)
                    try:
                        code = json.loads(call["function"]["arguments"]).get("code", "")
                    except (KeyError, TypeError, ValueError):
                        code = ""
                    if PROBE.search(code):
                        row["probe"] = True
                        probe_snippets += 1
    return {"prompted_turns": sum(v["prompt"] for v in by_turn.values()),
            "same_turn_calls": sum(v["prompt"] and v["probe"] for v in by_turn.values()),
            "all_call_turns": sum(v["probe"] for v in by_turn.values()),
            "prompt_deliveries": deliveries, "probe_snippets": probe_snippets}


def load_arm(path: Path, reach: bool) -> tuple[dict, dict]:
    benchmark = json.loads((path / "benchmark.json").read_text())
    runs = benchmark["game_runs"]
    by_game = {}
    for run in runs:
        game = run["game_id"].split("-")[0]
        if game in by_game:
            raise ValueError(f"duplicate game in one-pass arm: {game}")
        row = {"score": float(run.get("final_score") or 0),
               "levels": int(run.get("levels_completed") or 0)}
        if reach:
            files = list(path.glob(f"{run['game_id']}_p0_requests.jsonl"))
            if len(files) != 1:
                raise FileNotFoundError(f"one request log required for {game}: {files}")
            row.update(reach_file(files[0]))
        by_game[game] = row
    if set(by_game) != GAME_IDS:
        raise ValueError(f"wrong game set in {path}: {sorted(set(by_game) ^ GAME_IDS)}")
    meta = json.loads((path / "arm_meta.json").read_text())
    return by_game, meta


def full_batch_plans(meta: dict) -> tuple[int, int]:
    hx = (meta.get("b2_summary") or {}).get("hx") or {}
    interventions = hx.get("interventions") or []
    complete = 0
    for row in interventions:
        plan = row.get("proposed_plan") or []
        actual = row.get("actual_actions") or []
        if plan and len(plan) <= len(actual) and [str(x) for x in plan] == [str(x) for x in actual[:len(plan)]]:
            complete += 1
    return complete, len(interventions)


def interval(values: list[float], seed: int = 20261022, n: int = 10000) -> tuple[float, float]:
    rng = random.Random(seed)
    boots = [sum(values[rng.randrange(len(values))] for _ in values) / len(values)
             for _ in range(n)]
    return tuple(float(x) for x in np.percentile(boots, [5, 95]))


def analyze(ab: Path, ba: Path) -> dict:
    sessions = []
    diffs = defaultdict(list)
    for label, path in (("AB", ab), ("BA", ba)):
        a, ameta = load_arm(path / "arm_a", False)
        b, bmeta = load_arm(path / "arm_b", True)
        if set(a) != set(b):
            raise ValueError("arms have different games")
        prompted = sum(x["prompted_turns"] for x in b.values())
        calls = sum(x["same_turn_calls"] for x in b.values())
        full_plans, logged = full_batch_plans(bmeta)
        hx = (bmeta.get("b2_summary") or {}).get("hx") or {}
        deliveries = sum(x["prompt_deliveries"] for x in b.values())
        snippets = sum(x["probe_snippets"] for x in b.values())
        host_calls = int(hx.get("s_probe_calls") or 0)
        crosscheck = {   # Amendment 1: diagnostic only, never part of the gate
            "parsed_prompt_deliveries": deliveries,
            "host_stall_prompts_delivered": hx.get("stall_prompts_delivered"),
            "deliveries_match": deliveries == int(hx.get("stall_prompts_delivered") or 0),
            "parsed_probe_snippets": snippets,
            "host_s_probe_calls": hx.get("s_probe_calls"),
            "probe_calls_consistent": (snippets == 0) == (host_calls == 0),   # a snippet may call it 0..n times
            "host_s_stall_invocations": hx.get("s_stall_invocations"),
            "host_s_stall_triggers": hx.get("s_stall_triggers"),
            "host_s_probe_actions_followed": hx.get("s_probe_actions_followed"),
        }
        sessions.append({"order": label, "prompted_turns": prompted, "crosscheck": crosscheck,
                         "same_turn_calls": calls,
                         "same_turn_rate": calls / prompted if prompted else None,
                         "full_plans_verified_in_one_batch": full_plans,
                         "interventions_retained": logged,
                         "b2_hx_summary": (bmeta.get("b2_summary") or {}).get("hx"),
                         "arm_a_levels": sum(x["levels"] for x in a.values()),
                         "arm_b_levels": sum(x["levels"] for x in b.values()),
                         "arm_a_score_mean": sum(x["score"] for x in a.values()) / 25,
                         "arm_b_score_mean": sum(x["score"] for x in b.values()) / 25,
                         "per_game_reach": {g: {k: b[g][k] for k in ("prompted_turns", "same_turn_calls")}
                                            for g in sorted(b)}})
        for g in GAME_IDS:
            diffs[g].append({"score": b[g]["score"] - a[g]["score"],
                             "levels": b[g]["levels"] - a[g]["levels"]})
    paired = {}
    for field in ("score", "levels"):
        vals = [sum(x[field] for x in diffs[g]) / 2 for g in sorted(GAME_IDS)]
        paired[field] = {"mean_delta_b_minus_a": sum(vals) / len(vals),
                         "ci90": interval(vals)}
    prompted = sum(s["prompted_turns"] for s in sessions)
    calls = sum(s["same_turn_calls"] for s in sessions)
    full = sum(s["full_plans_verified_in_one_batch"] for s in sessions)
    return {"sessions": sessions, "pooled": {"prompted_turns": prompted,
            "same_turn_calls": calls, "same_turn_rate": calls / prompted if prompted else None,
            "full_plans_verified_in_one_batch": full,
            "reach_gate_pass": bool(prompted and calls / prompted >= .5 and full >= 1)},
            "paired": paired,
            "note": "Full plans across separate action() calls need manual trace verification; the automatic count is a lower bound."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ab", type=Path)
    parser.add_argument("ba", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/grcm/t3/analysis.json")
    args = parser.parse_args()
    result = analyze(args.ab, args.ba)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["pooled"], indent=2))
