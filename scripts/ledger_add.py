"""Append a run summary to runs/ledger.jsonl (keeps losing runs too)."""
import json, sys, time
from pathlib import Path
run = Path(sys.argv[1]); note = sys.argv[2] if len(sys.argv) > 2 else ""
s = json.loads((run / "summary.json").read_text())
m = s["meta"]
row = {"run": str(run), "utc": m["started_utc"], "agent": m["agent"], "config": m["config"],
       "commit": m["commit"], "dirty": m["dirty"], "games": len(m["games"]), "seeds": m["seeds"],
       "max_actions": m["max_actions"], "game_seconds": m["game_seconds"],
       "mean_score": round(s["mean_score"], 4), "total_levels": s["total_levels"],
       "games_with_progress": s["games_with_progress"], "wins": s["wins"], "errors": s["errors"],
       "total_actions": s["total_actions"], "wall_s": s["wall_seconds_total"],
       "split": "public25-dev (all 25 public games; contaminated by inspection)", "note": note}
with open("runs/ledger.jsonl", "a") as f:
    f.write(json.dumps(row) + "\n")
print(row)
