"""Run the (patched) Duck harness locally on the public games, mirroring the B0
notebook's configuration (env flags copied from Franzen's cell 4), against any
OpenAI-compatible endpoint (mock server, local model, or a rented GPU server).

Run with the harness venv:
  build/b0src/ARC3-Inference/.venv/bin/python kaggle/local/run_harness_local.py \
     --src build/b0src --games ls20,vc33 --minutes 3 --base-url http://127.0.0.1:8765/v1 \
     --model mock --out runs/harness_mock_smoke
Extra env overrides: --env KEY=VALUE (repeatable), e.g. B2 switches.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Verbatim from the B0 notebook (Franzen v3, cell 4), except the endpoint keys.
B0_SETUP_ENV = {
    'USE_TF': '0', 'TRANSFORMERS_NO_TF': '1', 'TRANSFORMERS_NO_TORCHVISION': '1',
    'LOCAL_ANALYZER_PROVIDER': 'vllm', 'OPENAI_PROVIDER': 'vllm',
    'LOCAL_ANALYZER_APP_NAME': 'ARC3 Agent Harness',
    'ARC3_REASONING_HISTORY_KEY': 'reasoning_content',
    'LOCAL_ANALYZER_ENABLE_THINKING': 'true', 'LOCAL_ANALYZER_TOOL_STEPS': '0',
    'LOCAL_ANALYZER_TOOL_TIMEOUT': '30', 'LOCAL_ANALYZER_YIELD_SECONDS': '0',
    'LOCAL_ANALYZER_YIELD_TOKENS': '2048', 'LOCAL_ANALYZER_TEMPERATURE': '0.7',
    'LOCAL_ANALYZER_TOP_P': '0.95', 'LOCAL_ANALYZER_TOP_K': '20',
    'LOCAL_ANALYZER_CONTEXT_WINDOW': (116 + 12) * 1024, 'LOCAL_ANALYZER_MAX_OUTPUT': 12 * 1024,
    'ARC3_CONTEXT_DRAIN_TOKENS': 58 * 1024, 'ARC3_HISTORY_ASSISTANT_TURNS': 150,
    'ARC3_HISTORY_TURN_DRAIN': 30, 'ARC3_HISTORY_DRAIN_COALESCE': '1',
    'LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS': '3072', 'ARC3_MIDDLE_TRUNCATION': '1', 'ARC3_ACTION_ECHO': '1',
    'ARC3_EXPLAIN_GAMEPLAY_CHANGED': '1', 'ARC3_NEW_CHANGED_PROMPTS': '1',
    'ARC3_LEVEL_TRANSFER_GUIDANCE': '1', 'ARC3_ACTION_INFO': '1', 'ARC3_DEDUPE_MULTICALL_LINE': '1',
    'ARC3_FRAME_DIFF_HINT': '1',
    'MULTIMODAL_CONTEXT': 'current_grid', 'MULTIMODAL_UPSCALE': '10', 'ARC3_DIFF_IMAGE': '1',
    'ARC3_GAMEOVER_DIFF_IMAGE': '1', 'ARC3_ANIMATION': '1',
    'ARC3_GUARDS_FROM_LEVEL': 2, 'ARC3_BATCH_NOOP_BLOCK': '1', 'ARC3_STALE_STATE_BLOCK': '1',
    'ARC3_MEMORY_SECTIONS': 'off',
    'ARC3_PERSISTENT_FUNCTIONS': '1', 'ARC3_PERSISTENT_FUNCTIONS_SCOPE': 'game',
    'ARC3_PERSISTENT_FUNCTIONS_IMPORTS': '1',
    'EXPOSE_UNDO': 'on',
}
B0_PRIORITY_ENV = {
    'ARC3_MAX_ACTIVE_STREAMS': 10, 'ARC3_PRIORITY_REFRESH_QUEUE': '1', 'ARC3_PRIORITY_PACE': '0',
    'ARC3_PRIORITY_TAIL_FADE': '1', 'ARC3_PRIORITY_TAIL_FADE_FRACTION': '0.2',
    'ARC3_PRIORITY_TAIL_LOOKUP': 'remaining', 'ARC3_PRIORITY_SCORE_NORMALIZATION': '1',
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(ROOT / "build" / "b0src"))
    p.add_argument("--games", default="ls20")
    p.add_argument("--minutes", type=float, default=3.0)
    p.add_argument("--passes", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--streams", type=int, default=10)
    p.add_argument("--base-url", default="http://127.0.0.1:8765/v1")
    p.add_argument("--model", default="mock")
    p.add_argument("--out", required=True)
    p.add_argument("--env", action="append", default=[])
    p.add_argument("--no-priority", action="store_true")
    p.add_argument("--max-tokens-per-game", type=int, default=0)
    a = p.parse_args()

    env = dict(B0_SETUP_ENV)
    if not a.no_priority:
        env.update(B0_PRIORITY_ENV)
        env['ARC3_MAX_ACTIVE_STREAMS'] = a.streams
    env.update({'LOCAL_ANALYZER_BASE_URL': a.base_url, 'OPENAI_BASE_URL': a.base_url,
                'LOCAL_ANALYZER_MODEL_ID': a.model, 'INFERENCE_ANALYZER_MODEL': a.model,
                'MPLBACKEND': 'Agg', 'TAAF_RUN_AS_SUBMISSION': '0', 'TAAF_MINIMAL_DIAGNOSTICS': '0',
                'OPERATION_MODE': 'offline'})
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    os.environ.update({k: str(v) for k, v in env.items()})
    os.environ["ONLY_RESET_LEVELS"] = "true"      # as in the notebook (offline arcade)

    src = Path(a.src)
    sys.path.insert(0, str(src / "ARC3-Inference"))
    sys.path.insert(0, str(src / "tufa-arc-agi-framework" / "src"))
    import arc_agi
    import taaf.benchmark
    import taaf.game_api
    from inference.framework.solver import HarnessSolver

    env_dir = str(ROOT / "data" / "environment_files")
    spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    all_ids = [e.game_id for e in arc.available_environments]
    want = [g.strip() for g in a.games.split(",")] if a.games != "all" else None
    ids = [g for g in all_ids if want is None or any(g.startswith(w) for w in want)]
    games = [taaf.game_api.GameAPI(env_name=g, arcade_spec=spec) for g in ids]

    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    solver = HarnessSolver(label="duck-harness-local", model="local", analyzer_timeout=900.0,
                           max_actions_per_game=None, max_runtime_s_per_game=a.minutes * 60,
                           max_generated_tokens_per_game=(a.max_tokens_per_game or None),
                           concurrency=a.concurrency * a.passes if not a.no_priority else a.concurrency,
                           save_request_logs=True)
    bm = taaf.benchmark.Benchmark(label="local", games=games, solver=solver, n_passes=a.passes, job_dir=out)
    (out / "local_run_env.json").write_text(json.dumps({"env": env, "games": ids, "args": vars(a)}, indent=1, default=str))
    t0 = time.time()
    soft_end = datetime.now() + timedelta(minutes=a.minutes + 1.5)
    asyncio.run(bm.run(soft_end_time=soft_end, runtime_environment=None, minimal_diagnostics=False))
    print(f"done in {time.time() - t0:.1f}s; outputs in {out}")


if __name__ == "__main__":
    main()
