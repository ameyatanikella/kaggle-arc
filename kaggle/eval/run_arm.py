"""Run ONE evaluation arm of the harness on the public games (Kaggle commit run).

Executed as a subprocess by the paired-eval notebook so arms cannot share
in-process scheduler state. Inherits the notebook's environment (all ARC3_*
switches, server URL) and the bundle paths (.pth written by the notebook).

usage: python run_arm.py <arm_label> <seconds> <passes> <out_dir> [switches_json]
switches_json: {"B2_BAR_FIX": "1", ...}; empty/absent = plain B0.
"""
import asyncio
import json
import os
import pickle
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

arm, T, passes, out = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
switches = json.loads(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else {}
out.mkdir(parents=True, exist_ok=True)
for k in [k for k in os.environ if k.startswith("B2_")]:
    del os.environ[k]                       # arms never inherit each other's switches
os.environ.update({k: str(v) for k, v in switches.items()})
b2 = None
if switches:
    sys.path.insert(0, "/tmp")
    import b2_patches as b2
    b2.install()

import arc_agi  # noqa: E402
import taaf.game_api  # noqa: E402

BUNDLE_DIR = Path("/kaggle/taaf-kaggle-source-share")
with open(BUNDLE_DIR / "deploy_target.pkl", "rb") as f:
    target = pickle.load(f)
target.actual_run_as_submission = False
target.is_competition_rerun = False
with open(BUNDLE_DIR / "benchmark_initial.pkl", "rb") as f:
    bm = pickle.load(f)
bm.job_dir = out
# Same solver settings as the B0 customization hook (non-submission branch).
bm.solver.analyzer_timeout = 900.0
bm.solver.max_actions_per_game = None
bm.solver.save_request_logs = True
bm.game_weights = None
bm.n_passes = passes
bm.solver.concurrency = 25 * passes
bm.solver.max_runtime_s_per_game = T

env_dir = str(Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels").parent / "environment_files")
if not Path(env_dir).is_dir():                     # mount layout varies between kernels (RUNBOOK)
    for _root, _dirs, _files in os.walk("/kaggle/input"):
        if _root.count("/") > 6:
            _dirs[:] = []
            continue
        if "environment_files" in _dirs:
            env_dir = os.path.join(_root, "environment_files")
            break
print(f"[arm {arm}] environment_files: {env_dir}", flush=True)
spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
ids = [e.game_id for e in arcade.available_environments]
bm.games = [taaf.game_api.GameAPI(env_name=g, arcade_spec=spec) for g in ids]
print(f"[arm {arm}] switches={switches} {len(ids)} games x {passes} passes, {T:.0f}s", flush=True)
t0 = time.time()
asyncio.run(bm.run(soft_end_time=datetime.now() + timedelta(seconds=T + 120),
                   runtime_environment=target, minimal_diagnostics=False))
res = {"arm": arm, "switches": switches, "seconds": time.time() - t0, "passes": passes, "games": ids}
if b2 is not None:
    res["b2_summary"] = b2.b2_summary()
(out / "arm_meta.json").write_text(json.dumps(res, indent=1, default=str))
print(f"[arm {arm}] done in {time.time() - t0:.0f}s", flush=True)
