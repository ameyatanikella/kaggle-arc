"""Download the public ARC-AGI-3 games through the official arc_agi client.

Uses Arcade in NORMAL mode, which fetches an anonymous API key and downloads
each game's metadata + source into ENVIRONMENTS_DIR (the same thing the
official starter's `make play-local` does). The agent never reads these files;
they are only used by the local evaluator to run games offline.
"""
import json
import sys
from pathlib import Path

import arc_agi

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/environment_files")
out_dir.mkdir(parents=True, exist_ok=True)

arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.NORMAL,
                     environments_dir=str(out_dir))
envs = arc.get_environments()
print(f"{len(envs)} environments listed")
for e in envs:
    print(e.game_id, e.title, e.tags, e.baseline_actions)
ok = 0
for e in envs:
    w = arc.make(e.game_id)
    if w is None:
        print("FAILED", e.game_id)
    else:
        ok += 1
print(f"downloaded/loaded {ok}/{len(envs)}")
