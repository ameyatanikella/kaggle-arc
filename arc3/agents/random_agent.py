"""Uniform random legal actions (pipeline sanity baseline)."""
from __future__ import annotations

import random

from ..agent_api import Act, Agent


class RandomAgent(Agent):
    name = "random"

    def __init__(self, game_id, config=None, seed=0):
        super().__init__(game_id, config, seed)
        self.rng = random.Random(hash((game_id, seed)) & 0xFFFFFFFF)

    def act(self, obs, deadline):
        if obs.state == "GAME_OVER":
            return Act(0, why="reset after game over")
        acts = [a for a in obs.available_actions if a != 0] or [1, 2, 3, 4]
        a = self.rng.choice(acts)
        if a == 6:
            return Act(6, self.rng.randrange(64), self.rng.randrange(64))
        return Act(a)
