"""Count wasted no-op actions per level in Duck-harness transcripts (action echo lines).

usage: python scripts/noop_loops.py <transcripts_dir>

With ARC3_ACTION_ECHO=1 every executed action prints
"[action] NAME -> effect" or "[action] NAME -> NO-OP (executed, no change inside the board area)".
Per tool result: actions, no-ops, longest run of consecutive no-ops. Aggregated by level.
B0 arms its stale-state guard only from level 2 (ARC3_GUARDS_FROM_LEVEL=2). This
measures what that costs on level 1.
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

HDR = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \|")
LVL = re.compile(r"Current state: step (\d+), level (\d+)\.")
ECHO = re.compile(r"\[action\] (.+?) -> (NO-OP|effect|[^\n\\]*)")


def main(d):
    agg = defaultdict(lambda: {"actions": 0, "noops": 0, "snippets": 0, "runs_ge5": 0, "runs_ge10": 0, "max_run": 0,
                               "wasted_in_runs_ge5": 0})
    files = sorted(Path(d).glob("*.txt"))
    for f in files:
        level = 1
        for block in re.split(r"\n(?=--- analysis_step=)", f.read_text(errors="replace")):
            m = LVL.search(block)
            if m:
                level = int(m.group(2))
            for res in re.findall(r"\[TOOL RESULT: python\]\n(.*?)(?=\n\[[A-Z ]+\]|\Z)", block, re.S):
                seq = [x[1] for x in ECHO.findall(res)]
                if not seq:
                    continue
                a = agg[level]
                a["snippets"] += 1
                a["actions"] += len(seq)
                n_noop = sum(1 for s in seq if s == "NO-OP")
                a["noops"] += n_noop
                run = best = 0
                runs = []
                for s in seq:
                    if s == "NO-OP":
                        run += 1
                    else:
                        if run:
                            runs.append(run)
                        run = 0
                if run:
                    runs.append(run)
                best = max(runs) if runs else 0
                a["max_run"] = max(a["max_run"], best)
                a["runs_ge5"] += sum(1 for r in runs if r >= 5)
                a["runs_ge10"] += sum(1 for r in runs if r >= 10)
                a["wasted_in_runs_ge5"] += sum(r - 1 for r in runs if r >= 5)
    print(f"{len(files)} transcripts")
    for lv in sorted(agg):
        a = agg[lv]
        print(f"L{lv}: snippets={a['snippets']:5d} actions={a['actions']:6d} noops={a['noops']:5d} "
              f"({100 * a['noops'] / max(1, a['actions']):.0f}%) runs>=5={a['runs_ge5']} runs>=10={a['runs_ge10']} "
              f"max_run={a['max_run']} wasted(in runs>=5, beyond 1st)={a['wasted_in_runs_ge5']}")


if __name__ == "__main__":
    main(sys.argv[1])
