"""Mechanism checks for B2 switches from TAAF transcripts (does the switch fire, and
does behaviour change in the intended direction?).

usage: python scripts/b2_mechanisms.py <arm_dir> [--window 5]

Per transcript:
  * stuck nudge: turns where the user prompt carried "Progress check:"; actions in
    the `window` turns before vs after; whether the level cleared afterwards and
    how many turns later.
  * solved memory: "LEVEL N SOLVED" asks and whether a "Rule for level N:" line
    appeared in the model's following reasoning/content.
Window comparisons are within-run descriptive statistics, not causal estimates.
"""
import argparse
import re
from pathlib import Path

HDR = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \|")
LVL = re.compile(r"Current state: step (\d+), level (\d+)\.")


def turns_of(path: Path):
    turns, cur, section = [], None, None
    for line in path.read_text(errors="replace").split("\n"):
        m = HDR.match(line)
        if m:
            if cur:
                turns.append(cur)
            cur = {"step": int(m.group(1)), "action": int(m.group(2)), "level": None,
                   "nudge": False, "ask": None, "rule": set(), "user": [], "model": []}
            section = None
            continue
        if cur is None:
            continue
        if line.startswith("[") and line.rstrip().endswith("]") and len(line) < 40:
            section = line.strip("[] \n")
            continue
        if section == "USER PROMPT":
            mm = LVL.search(line)
            if mm and cur["level"] is None:
                cur["level"] = int(mm.group(2))
            if "Progress check:" in line:
                cur["nudge"] = True
            ma = re.search(r"LEVEL (\d+) SOLVED\. In your reply", line)
            if ma:
                cur["ask"] = int(ma.group(1))
        elif section in ("THINKING", "ASSISTANT"):
            for mr in re.finditer(r"Rule for level\s*(\d+)\s*:", line):
                cur["rule"].add(int(mr.group(1)))
    if cur:
        turns.append(cur)
    return turns


def analyse(path: Path, window: int):
    t = turns_of(path)
    out = {"game": path.stem, "turns": len(t), "nudges": [], "asks": 0, "rules": 0}
    last_level = 1
    for i, x in enumerate(t):
        if x["level"] is None:
            x["level"] = last_level
        last_level = x["level"]
    for i, x in enumerate(t):
        if x["nudge"]:
            before = x["action"] - t[max(0, i - window)]["action"]
            j = min(len(t) - 1, i + window)
            after = t[j]["action"] - x["action"]
            cleared = next((k - i for k in range(i + 1, len(t)) if t[k]["level"] > x["level"]), None)
            out["nudges"].append({"turn": i, "level": x["level"], "actions_before": before,
                                  "actions_after": after, "cleared_after_turns": cleared})
        if x["ask"] is not None:
            out["asks"] += 1
            if any(x["ask"] in y["rule"] for y in t[i:i + 3]):
                out["rules"] += 1
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("arm_dir")
    p.add_argument("--window", type=int, default=5)
    a = p.parse_args()
    root = Path(a.arm_dir)
    files = sorted((root / "transcripts").glob("*.txt")) if (root / "transcripts").exists() else sorted(root.glob("*.txt"))
    tot_n = tot_b = tot_a = cleared = asks = rules = 0
    for f in files:
        r = analyse(f, a.window)
        asks += r["asks"]
        rules += r["rules"]
        for n in r["nudges"]:
            tot_n += 1
            tot_b += n["actions_before"]
            tot_a += n["actions_after"]
            cleared += n["cleared_after_turns"] is not None
        if r["nudges"] or r["asks"]:
            print(f"{r['game']:22s} turns={r['turns']:4d} nudges={len(r['nudges'])} asks={r['asks']} rules={r['rules']} "
                  + " ".join(f"[L{n['level']} {n['actions_before']}->{n['actions_after']} clr={n['cleared_after_turns']}]"
                             for n in r["nudges"][:4]))
    print(f"\nTOTAL nudges={tot_n} actions/window before={tot_b / max(1, tot_n):.1f} after={tot_a / max(1, tot_n):.1f} "
          f"level cleared later={cleared}/{tot_n} | solved-memory asks={asks} rules captured={rules}")


if __name__ == "__main__":
    main()
