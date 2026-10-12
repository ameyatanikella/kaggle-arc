"""Agent interface shared by all local agents.

An agent receives only `Observation`s (see env.py). It returns an `Act`, or
None to stop playing this game. The trusted runner (not the agent) owns action
legality, deadlines and accounting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .env import Observation


@dataclass(frozen=True)
class Act:
    action: int
    x: Optional[int] = None
    y: Optional[int] = None
    why: str = ""


class Agent:
    name = "base"

    def __init__(self, game_id: str, config: Optional[dict] = None, seed: int = 0):
        self.game_id = game_id
        self.config = dict(config or {})
        self.seed = seed
        self.stats: dict = {}

    def act(self, obs: Observation, deadline: float) -> Optional[Act]:
        raise NotImplementedError
