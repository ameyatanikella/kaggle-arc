"""Faithful typed event log.

The complete history is h_t = (o_0, a_0, e_1, ..., a_{t-1}, e_t). One engine call (one action)
returns 1..N frames. They are classified, never compared blindly:

  kind 'move'      ordinary step: frames[:-1] are animation, frames[-1] is the next DECISION state
                   on the same level. Only these give dynamics transitions (pre, a, post).
  kind 'win'       the action completed a level: frames[-2] is the final state of the old level (the
                   WIN frame, absent if the engine returned one frame: then we abstain), frames[-1]
                   is the INIT frame of the next level. (pre, a, win) is a terminal transition;
                   (win -> init) is NOT a transition and is never diffed.
  kind 'game_win'  the action completed the last level (state WIN).
  kind 'death'     state GAME_OVER after the action: frames[-1] is the DEATH frame; terminal.
  kind 'reset'     a RESET (agent or automatic). It costs an action and returns the level start; it
                   is not a game-dynamics transition and is excluded from dynamics.
  kind 'refused'   the engine returned no frame (e.g. a non-RESET after GAME_OVER): uncounted.

Grids are deduplicated by content. Every step stores the legal-action mask that held *before* the
action and the click coordinates. Verified against the engine: at a level-up step frames[-2] is
the old level's last frame and frames[-1] the new level's first (vc33, lp85, ft09, r11l, sp80, tu93).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

MOVE, WIN, GAME_WIN, DEATH, RESET, REFUSED = "move", "win", "game_win", "death", "reset", "refused"


def grid_digest(g: np.ndarray) -> bytes:
    return hashlib.blake2b(np.ascontiguousarray(g, dtype=np.uint8).tobytes(), digest_size=12).digest()


@dataclass
class Step:
    t: int                       # step index
    level: int                   # 0-based index of the level the action was taken on
    action: tuple                # (a,) or (6, x, y); RESET is (0,)
    legal: tuple                 # available actions before the action
    kind: str                    # MOVE / WIN / GAME_WIN / DEATH / RESET / REFUSED
    pre: int                     # grid id of the decision state the action was taken from
    post: Optional[int]          # grid id of the next decision state on the SAME level (MOVE, RESET)
    terminal: Optional[int] = None   # grid id of the win / death frame (WIN, GAME_WIN, DEATH)
    init: Optional[int] = None   # grid id of the next level's first frame (WIN)
    n_frames: int = 1
    anim: tuple = ()             # grid ids of intermediate frames (kept only if keep_anim)
    transient: int = 0           # pixels that changed more than once during the animation
    automatic: bool = False
    source: str = ""             # who chose the action (agent / model / tool / auto)


@dataclass
class EventLog:
    keep_anim: bool = False
    grids: list = field(default_factory=list)
    _gid: dict = field(default_factory=dict)
    steps: list = field(default_factory=list)
    level_start: dict = field(default_factory=dict)     # level -> grid id of its first decision state
    current: Optional[int] = None                       # grid id of the current decision state
    level: int = 0

    # ------------------------------------------------------------------ storage
    def gid(self, g: np.ndarray) -> int:
        g = np.asarray(g, dtype=np.uint8)
        k = grid_digest(g)
        i = self._gid.get(k)
        if i is None:
            i = len(self.grids)
            self.grids.append(g.copy())
            self._gid[k] = i
        return i

    def grid(self, i: Optional[int]) -> Optional[np.ndarray]:
        return None if i is None else self.grids[i]

    def start(self, grid: np.ndarray, level: int = 0) -> None:
        self.current = self.gid(grid)
        self.level = level
        self.level_start.setdefault(level, self.current)

    # ------------------------------------------------------------------ recording
    def record(self, action: tuple, frames: Iterable[np.ndarray], state: str, levels_before: int,
               levels_after: int, legal: tuple = (), pre_grid: Optional[np.ndarray] = None,
               automatic: bool = False, source: str = "") -> Step:
        """Record one engine call. `pre_grid` overrides the remembered current state (the LLM host
        passes the engine's own previous grid). Returns the Step."""
        frames = [np.asarray(f, dtype=np.uint8) for f in frames]
        pre = self.gid(pre_grid) if pre_grid is not None else self.current
        if pre is None:
            raise ValueError("EventLog.start() was not called")
        t = len(self.steps)
        a = tuple(int(v) for v in action)
        if not frames:
            st = Step(t, levels_before, a, tuple(legal), REFUSED, pre, None, n_frames=0, automatic=automatic,
                      source=source)
            self.steps.append(st)
            return st
        last = self.gid(frames[-1])
        transient = _transient_pixels(self.grids[pre], frames) if len(frames) > 2 else 0
        anim = tuple(self.gid(f) for f in frames[:-1]) if self.keep_anim else ()
        if a[0] == 0:
            kind, post, term, init = RESET, last, None, None
        elif state == "WIN":
            kind, post, term, init = GAME_WIN, None, last, None
        elif levels_after > levels_before:
            kind, post, init = WIN, None, last
            term = self.gid(frames[-2]) if len(frames) >= 2 else None    # abstain without a win frame
        elif state == "GAME_OVER":
            kind, post, term, init = DEATH, None, last, None
        else:
            kind, post, term, init = MOVE, last, None, None
        st = Step(t, levels_before, a, tuple(legal), kind, pre, post, term, init, len(frames), anim, transient,
                  automatic, source)
        self.steps.append(st)
        self.current = last
        self.level = levels_after
        if kind == WIN:
            self.level_start.setdefault(levels_after, last)
        return st

    # ------------------------------------------------------------------ queries
    def transitions(self, level: Optional[int] = None, include_terminal: bool = False):
        """Dynamics transitions between consecutive decision states: (pre_id, action, post_id, kind).
        RESET, REFUSED and level switches are never transitions."""
        for s in self.steps:
            if level is not None and s.level != level:
                continue
            if s.kind == MOVE:
                yield s.pre, s.action, s.post, MOVE
            elif include_terminal and s.kind in (WIN, GAME_WIN, DEATH) and s.terminal is not None:
                yield s.pre, s.action, s.terminal, s.kind

    def level_steps(self, level: int) -> list:
        return [s for s in self.steps if s.level == level]

    def wins(self) -> list:
        """One record per completed level: level, start id, the winning step, win and init ids, and
        the ids of every decision state visited on that level."""
        out = []
        for s in self.steps:
            if s.kind in (WIN, GAME_WIN):
                visited = {s.pre}
                for q in self.level_steps(s.level):
                    visited.add(q.pre)
                    if q.post is not None:
                        visited.add(q.post)
                out.append({"level": s.level, "start": self.level_start.get(s.level), "step": s,
                            "win": s.terminal, "init": s.init, "visited": sorted(visited)})
        return out

    def deaths(self, level: Optional[int] = None) -> list:
        return [s for s in self.steps if s.kind == DEATH and (level is None or s.level == level)]

    def actions_on_level(self, level: int) -> int:
        """Counted actions on a level (RESET included, refused excluded): the scored quantity."""
        return sum(1 for s in self.steps if s.level == level and s.kind != REFUSED)


def _transient_pixels(pre: np.ndarray, frames: list) -> int:
    """Pixels whose value changes more than once along pre -> frames (states not visible in the
    decision frames; Franzen's animation-hint criterion)."""
    seq = [pre] + list(frames)
    changes = np.zeros(pre.shape, dtype=np.int32)
    for a, b in zip(seq, seq[1:]):
        changes += (a != b)
    return int((changes > 1).sum())
