"""Local evaluation runner: one process per game, official scoring, full logs.

Usage:
  python -m arc3.runner --agent explorer --games all --workers 10 \
      --max-actions 1500 --game-seconds 300 --out runs/b1_explorer_v1

Each game runs in its own process (arc_agi's ONLY_RESET_LEVELS switch is
process-global and must be flipped after the opening `make`). The per-game
score is the official arc_agi scorecard result, computed in-process by
`close_scorecard()`; nothing is hand-computed.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import multiprocessing as mp
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENV_DIR = ROOT / "data" / "environment_files"

AGENTS = {
    "random": "arc3.agents.random_agent:RandomAgent",
    "explorer": "arc3.agents.explorer:ExplorerAgent",
    "b3_ablation": "arc3.research.b3_ablation.agent:B3Agent",
    "b3": "arc3.agents.b3.agent:B3Agent",
}


def load_agent_cls(spec: str):
    spec = AGENTS.get(spec, spec)
    mod, cls = spec.split(":")
    return getattr(importlib.import_module(mod), cls)


def list_games(env_dir: Path = ENV_DIR) -> list[str]:
    out = []
    for meta in sorted(env_dir.rglob("metadata.json")):
        out.append(json.loads(meta.read_text())["game_id"])
    return out


def _quiet_logger() -> logging.Logger:
    lg = logging.getLogger("arc3.arcade")
    lg.handlers.clear()
    lg.addHandler(logging.NullHandler())
    lg.setLevel(logging.ERROR)
    lg.propagate = False
    logging.getLogger("arc_agi.scorecard").setLevel(logging.ERROR)
    return lg


def play_one(args: dict) -> dict:
    """Runs in a worker process."""
    game_id = args["game_id"]
    out_dir = Path(args["out"])
    os.environ.pop("ONLY_RESET_LEVELS", None)
    os.environ["OPERATION_MODE"] = "offline"
    import arc_agi  # noqa: E402  (after env setup)
    from arc3.env import CompetitionEnv, ActionRefused

    t_start = time.time()
    res = {"game_id": game_id, "agent": args["agent"], "seed": args["seed"], "error": None}
    log: list = []
    frames: list = []
    agent = None
    env = None
    try:
        arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE,
                                environments_dir=str(args["env_dir"]),
                                recordings_dir=str(out_dir / "recordings"),
                                logger=_quiet_logger())
        env = CompetitionEnv(arcade, game_id, log=log)
        cls = load_agent_cls(args["agent"])
        agent = cls(game_id=game_id, config=args.get("config") or {}, seed=args["seed"])
        obs = env.obs
        frames.append(obs.grid.copy())
        deadline = t_start + args["game_seconds"]
        stop_reason = "agent_stop"
        while True:
            if obs.state == "WIN":
                stop_reason = "win"
                break
            if env.actions_used >= args["max_actions"]:
                stop_reason = "action_cap"
                break
            if time.time() >= deadline:
                stop_reason = "time_cap"
                break
            ta = time.perf_counter()
            act = agent.act(obs, deadline)
            res["agent_seconds"] = res.get("agent_seconds", 0.0) + time.perf_counter() - ta
            if act is None:
                stop_reason = "agent_stop"
                break
            try:
                obs = env.step(act.action, act.x, act.y)
            except ActionRefused as e:
                res.setdefault("refused", []).append(str(e))
                if len(res["refused"]) > 20:
                    stop_reason = "too_many_refusals"
                    break
                continue
            frames.append(obs.grid.copy())
        res["stop_reason"] = stop_reason
    except Exception as e:  # recorded, never hidden
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()
    res["wall_seconds"] = round(time.time() - t_start, 2)
    res["actions"] = env.actions_used if env else 0
    res["engine_seconds"] = round(sum(r.get("dt", 0.0) for r in log), 3)
    res["resets"] = env.resets if env else 0
    res["level_actions"] = env.level_action_counts if env else []
    if agent is not None and hasattr(agent, "finalize"):
        try:
            agent.finalize()
        except Exception as e:  # recorded, never hidden
            res["finalize_error"] = f"{type(e).__name__}: {e}"
    res["agent_stats"] = getattr(agent, "stats", {}) if agent else {}
    res["max_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)
    # Official score for this game (in-process scorecard).
    try:
        sc = arcade.close_scorecard()
        envs = sc.environments if sc else []
        if envs:
            e = envs[0]
            run = max(e.runs, key=lambda r: r.score)
            res["score"] = float(e.score)
            res["levels_completed"] = int(e.levels_completed)
            res["n_levels"] = len(run.level_scores or [])
            res["level_scores"] = [round(s, 3) for s in (run.level_scores or [])]
            res["official_level_actions"] = run.level_actions
            res["baselines"] = run.level_baseline_actions
        else:
            res["score"] = 0.0
            res["levels_completed"] = 0
    except Exception as e:
        res["score_error"] = f"{type(e).__name__}: {e}"
        res.setdefault("score", 0.0)
    gdir = out_dir / "games"
    gdir.mkdir(parents=True, exist_ok=True)
    with open(gdir / f"{game_id}.log.jsonl", "w") as f:
        for row in log:
            f.write(json.dumps(row) + "\n")
    if frames:
        np.savez_compressed(gdir / f"{game_id}.frames.npz", frames=np.stack(frames).astype(np.uint8))
    with open(gdir / f"{game_id}.result.json", "w") as f:
        json.dump(res, f, indent=1, default=str)
    return res


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--", "arc3"],
                                      stderr=subprocess.DEVNULL).decode().strip()
        return bool(out)
    except Exception:
        return True


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="explorer")
    p.add_argument("--games", default="all", help="comma list of game-id prefixes, or 'all'")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-actions", type=int, default=1500)
    p.add_argument("--game-seconds", type=float, default=300)
    p.add_argument("--seeds", default="0")
    p.add_argument("--config", default="{}", help="JSON dict passed to the agent")
    p.add_argument("--out", required=True)
    p.add_argument("--env-dir", default=str(ENV_DIR))
    a = p.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    all_games = list_games(Path(a.env_dir))
    if a.games == "all":
        games = all_games
    else:
        pref = [g.strip() for g in a.games.split(",") if g.strip()]
        games = [g for g in all_games if any(g.startswith(p_) for p_ in pref)]
    seeds = [int(s) for s in a.seeds.split(",")]
    config = json.loads(a.config)
    meta = {
        "agent": a.agent, "games": games, "seeds": seeds, "config": config,
        "max_actions": a.max_actions, "game_seconds": a.game_seconds, "workers": a.workers,
        "commit": git_commit(), "dirty": git_dirty(), "python": platform.python_version(),
        "machine": platform.machine(), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        import arc_agi, arcengine  # noqa
        from importlib.metadata import version
        meta["arc_agi"] = version("arc-agi")
        meta["arcengine"] = version("arcengine")
    except Exception:
        pass
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1))
    jobs = [{"game_id": g, "agent": a.agent, "seed": s, "config": config, "out": str(out / f"seed{s}"),
             "env_dir": a.env_dir, "max_actions": a.max_actions, "game_seconds": a.game_seconds}
            for s in seeds for g in games]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    results = []
    with ctx.Pool(processes=a.workers, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(play_one, jobs):
            results.append(r)
            print(f"[{len(results):3d}/{len(jobs)}] {r['game_id']:16s} seed={r['seed']} score={r.get('score', 0):6.2f} "
                  f"lv={r.get('levels_completed', 0)}/{r.get('n_levels', '?')} act={r['actions']:5d} "
                  f"stop={r.get('stop_reason')} t={r['wall_seconds']:.0f}s"
                  + (f" ERR={r['error']}" if r.get("error") else ""), flush=True)
    summary = summarize(results)
    summary["wall_seconds_total"] = round(time.time() - t0, 1)
    summary["meta"] = meta
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (out / "results.json").write_text(json.dumps(results, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_game", "meta")}, indent=1))
    return summary


def summarize(results: list[dict]) -> dict:
    by_seed: dict = {}
    for r in results:
        by_seed.setdefault(r["seed"], []).append(r)
    seed_means = {s: float(np.mean([r.get("score", 0.0) for r in rs])) for s, rs in by_seed.items()}
    per_game: dict = {}
    for r in results:
        g = per_game.setdefault(r["game_id"], {"scores": [], "levels": [], "actions": [], "errors": 0})
        g["scores"].append(round(r.get("score", 0.0), 3))
        g["levels"].append(r.get("levels_completed", 0))
        g["actions"].append(r.get("actions", 0))
        g["errors"] += 1 if r.get("error") else 0
    return {
        "mean_score": float(np.mean(list(seed_means.values()))) if seed_means else 0.0,
        "seed_means": seed_means,
        "total_levels": int(sum(r.get("levels_completed", 0) for r in results)),
        "games_with_progress": int(sum(1 for r in results if r.get("levels_completed", 0) > 0)),
        "wins": int(sum(1 for r in results if r.get("stop_reason") == "win")),
        "errors": int(sum(1 for r in results if r.get("error"))),
        "total_actions": int(sum(r.get("actions", 0) for r in results)),
        "per_game": per_game,
    }


if __name__ == "__main__":
    main()
