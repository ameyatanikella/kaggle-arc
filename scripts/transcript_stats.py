"""Per-game / per-level statistics from TAAF Duck-harness transcripts.

usage: python scripts/transcript_stats.py <run_dir_or_transcripts_dir> [--json out.json]

Parses `--- analysis_step=N | action=M | HH:MM:SS | ...` blocks. For each
analyzer turn: reasoning chars, tool calls, helper usage, level/step from the
user prompt. Aggregates per game and per level: turns, reasoning chars (a
token proxy, ~3.5 chars/token), wall time, actions. Used to find where the
budget goes (stuck levels) and whether a switch/tool actually fired.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

HDR = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \| (\d\d:\d\d:\d\d) \|")
LVL = re.compile(r"Current state: step (\d+), level (\d+)\.")
RCH = re.compile(r"^reasoning_chars: (\d+)")
HELPERS = ("bfs_plan(", "distinguish(", "gf2_solve(", "zm_solve(", "frame_diff(", "UNDO")


def hms(s):
    h, m, x = map(int, s.split(":"))
    return h * 3600 + m * 60 + x


def parse(path: Path) -> list[dict]:
    turns, cur = [], None
    in_code = False
    for line in path.read_text(errors="replace").split("\n"):
        m = HDR.match(line)
        if m:
            if cur:
                turns.append(cur)
            in_code = False
            cur = {"step": int(m.group(1)), "action": int(m.group(2)), "t": hms(m.group(3)),
                   "level": None, "reasoning": 0, "tool_calls": 0, "helpers": defaultdict(int),
                   "solved_mem_ask": False, "game_over": False}
            continue
        if cur is None:
            continue
        m = LVL.search(line)
        if m and cur["level"] is None:
            cur["level"] = int(m.group(2))
        m = RCH.match(line)
        if m:
            cur["reasoning"] += int(m.group(1))
        if line.startswith("["):
            in_code = line.startswith("[TOOL CALL")
            if in_code:
                cur["tool_calls"] += 1
            continue
        if in_code:
            for h in HELPERS:
                if h in line:
                    cur["helpers"][h.rstrip("(")] += line.count(h)
        if "LEVEL" in line and "SOLVED. In your reply" in line:
            cur["solved_mem_ask"] = True
        if "GAME OVER occurred" in line:
            cur["game_over"] = True
    if cur:
        turns.append(cur)
    return turns


def summarize(turns: list[dict]) -> dict:
    if not turns:
        return {}
    per_level = defaultdict(lambda: {"turns": 0, "reasoning_chars": 0, "tool_calls": 0, "t0": None, "t1": None,
                                     "first_action": None, "last_action": None, "game_overs": 0})
    helpers = defaultdict(int)
    last_level = 1
    for t in turns:
        lv = t["level"] or last_level
        last_level = lv
        d = per_level[lv]
        d["turns"] += 1
        d["reasoning_chars"] += t["reasoning"]
        d["tool_calls"] += t["tool_calls"]
        d["game_overs"] += int(t["game_over"])
        d["t0"] = t["t"] if d["t0"] is None else d["t0"]
        d["t1"] = t["t"]
        d["first_action"] = t["action"] if d["first_action"] is None else d["first_action"]
        d["last_action"] = t["action"]
        for k, v in t["helpers"].items():
            helpers[k] += v
    out = {"turns": len(turns), "max_level_seen": max(per_level), "helpers": dict(helpers),
           "solved_mem_asks": sum(t["solved_mem_ask"] for t in turns),
           "reasoning_chars": sum(t["reasoning"] for t in turns), "levels": {}}
    for lv, d in sorted(per_level.items()):
        span = (d["t1"] - d["t0"]) % 86400 if d["t0"] is not None else 0
        out["levels"][lv] = {"turns": d["turns"], "reasoning_ktok~": round(d["reasoning_chars"] / 3500, 1),
                             "tool_calls": d["tool_calls"], "minutes": round(span / 60, 1),
                             "actions~": (d["last_action"] or 0) - (d["first_action"] or 0),
                             "game_overs": d["game_overs"]}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--json")
    a = p.parse_args()
    root = Path(a.path)
    files = sorted((root / "transcripts").glob("*.txt")) if (root / "transcripts").exists() else sorted(root.glob("*.txt"))
    allres = {}
    for f in files:
        s = summarize(parse(f))
        allres[f.stem] = s
        if not s:
            continue
        lv = " ".join(f"L{k}:{v['turns']}t/{v['reasoning_ktok~']}k/{v['minutes']}m" for k, v in s["levels"].items())
        hp = ",".join(f"{k}={v}" for k, v in s["helpers"].items() if v)
        print(f"{f.stem:22s} turns={s['turns']:3d} maxL={s['max_level_seen']} {lv} {hp}")
    if a.json:
        Path(a.json).write_text(json.dumps(allres, indent=1))


if __name__ == "__main__":
    main()
