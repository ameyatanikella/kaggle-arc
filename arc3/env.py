"""Competition-semantics environment over the official `arc_agi` package.

The agent only ever sees `Observation` objects (frames + public metadata).
It never sees game source, sprites, hidden state or evaluator internals.

Accounting mirrors arc_agi 0.9.9 scorecard semantics (see competition_contract.md):
  * every ACTION1..7 that returns a non-empty frame costs 1 action;
  * every agent RESET is a *level* reset and costs 1 action;
  * actions after GAME_OVER (other than RESET) are refused locally (the real
    engine would return an empty frame and not count them, but they are
    always a bug in the agent);
  * one environment creation per game.
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

SIMPLE_ACTIONS = (1, 2, 3, 4, 5, 7)
CLICK_ACTION = 6
RESET_ACTION = 0


@dataclass(frozen=True)
class Observation:
    frames: tuple            # tuple[np.ndarray(64,64) uint8], animation frames of the last call
    state: str               # NOT_FINISHED | WIN | GAME_OVER | NOT_PLAYED
    levels_completed: int
    win_levels: int
    available_actions: tuple  # tuple[int]
    actions_used: int        # counted actions so far (this game)
    level_actions_used: int  # counted actions so far on the current level

    @property
    def grid(self) -> np.ndarray:
        return self.frames[-1]


def frame_hash(grid: np.ndarray) -> str:
    return hashlib.blake2b(np.ascontiguousarray(grid, dtype=np.uint8).tobytes(), digest_size=8).hexdigest()


class ActionRefused(Exception):
    pass


class CompetitionEnv:
    """One game, one `make`, level resets only."""

    def __init__(self, arcade, game_id: str, log: Optional[list] = None):
        self.game_id = game_id
        self.arcade = arcade
        t0 = time.time()
        self.wrapper = arcade.make(game_id)   # opening RESET (full reset -> starts the play)
        if self.wrapper is None:
            raise RuntimeError(f"make({game_id}) failed")
        # From now on every RESET is a level reset (competition semantics). Must
        # be set *after* make(), otherwise the opening RESET never starts the play.
        os.environ["ONLY_RESET_LEVELS"] = "true"
        self.actions_used = 0
        self.level_actions_used = 0
        self.level_action_counts: list[int] = []
        self.resets = 0
        self.log = log if log is not None else []
        self.make_seconds = time.time() - t0
        raw = self.wrapper.observation_space
        self._last = self._to_obs(raw)

    # ------------------------------------------------------------------
    def _to_obs(self, raw) -> Observation:
        frames = tuple(np.asarray(f, dtype=np.uint8) for f in raw.frame) if raw.frame else ()
        if not frames:
            frames = self._last.frames if getattr(self, "_last", None) else (np.zeros((64, 64), np.uint8),)
        state = raw.state.name if hasattr(raw.state, "name") else str(raw.state)
        return Observation(
            frames=frames,
            state=state,
            levels_completed=int(raw.levels_completed),
            win_levels=int(raw.win_levels),
            available_actions=tuple(int(a) for a in (raw.available_actions or ())),
            actions_used=self.actions_used,
            level_actions_used=self.level_actions_used,
        )

    @property
    def obs(self) -> Observation:
        return self._last

    def step(self, action: int, x: Optional[int] = None, y: Optional[int] = None, reasoning=None) -> Observation:
        from arcengine import GameAction

        prev = self._last
        if prev.state == "WIN":
            raise ActionRefused("game already won")
        if action != RESET_ACTION:
            if prev.state == "GAME_OVER":
                raise ActionRefused("non-RESET action after GAME_OVER")
            if prev.available_actions and action not in prev.available_actions:
                raise ActionRefused(f"action {action} not in available_actions {prev.available_actions}")
        data = {}
        if action == CLICK_ACTION:
            if x is None or y is None or not (0 <= x <= 63 and 0 <= y <= 63):
                raise ActionRefused(f"ACTION6 needs x,y in [0,63], got {x},{y}")
            data = {"x": int(x), "y": int(y)}
        t0 = time.time()
        raw = self.wrapper.step(GameAction.from_id(action), data=data, reasoning=reasoning)
        dt = time.time() - t0
        if raw is None:
            raise RuntimeError(f"step({action}) returned None")
        counted = bool(raw.frame) or action == RESET_ACTION
        if counted:
            self.actions_used += 1
            self.level_actions_used += 1
        if action == RESET_ACTION:
            self.resets += 1
        obs = self._to_obs(raw)
        if obs.levels_completed > prev.levels_completed:
            self.level_action_counts.append(self.level_actions_used)
            self.level_actions_used = 0
            obs = Observation(**{**obs.__dict__, "level_actions_used": 0, "actions_used": self.actions_used})
        else:
            obs = Observation(**{**obs.__dict__, "actions_used": self.actions_used,
                                 "level_actions_used": self.level_actions_used})
        self._last = obs
        self.log.append({
            "t": len(self.log), "a": action, "x": x, "y": y, "state": obs.state,
            "lvl": obs.levels_completed, "nf": len(obs.frames), "h": frame_hash(obs.grid),
            "dt": round(dt, 5), "counted": counted,
        })
        return obs
