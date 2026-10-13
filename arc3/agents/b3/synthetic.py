"""Synthetic frame-based games for B3's targeted controls (no ARC data involved).

CorridorGame: a 2x2 mover (colour 3) on row 30 walks right (ACTION4) / left (ACTION3) in
4-px steps from x=8; reaching x>=56 completes the level. Options per level:
  switches   initial bits of three irrelevant switches (row 10, x=10/14/18), toggled by
             ACTION1 / ACTION2 / ACTION5 (pure nuisance unless `coupled`)
  blocker    a wall at corridor cell 6 (the mover cannot pass) present from the start
  coupled    switch A ON makes cell 6 bounce the mover back to the start (A is outside the
             plan's dependency region: the "independent" assumption is wrong)
  delay      the wall at cell 6 opens only after `delay` CONSECUTIVE ACTION5 presses
             (no visible change before the last one; any other action resets the count)
  jam        two consecutive ACTION1 presses jam the door mechanism until the level restarts
             (a destructive hidden effect)
The runner loop mirrors arc3.runner: the agent sees Observations only.
"""
from __future__ import annotations

import numpy as np

from ...env import Observation


class CorridorGame:
    def __init__(self, levels: list[dict]):
        self.levels = levels
        self.level = 0
        self.actions_used = 0
        self.level_actions = 0
        self.state = "NOT_FINISHED"
        self._start_level()

    def _start_level(self):
        L = self.levels[self.level]
        self.cfg = L
        self.x = 8
        self.sw = list(L.get("switches", (0, 0, 0)))
        self.count5 = 0
        self.prev1 = False
        self.jammed = False
        self.door_open = not (L.get("blocker") or L.get("delay"))

    def grid(self):
        g = np.zeros((64, 64), np.uint8)
        g[0, :] = 5                                   # HUD row (unchanging here)
        g[30:32, self.x:self.x + 2] = 3
        if not self.cfg.get("hide_switches"):
            for i, b in enumerate(self.sw):
                g[10, 10 + 4 * i] = 7 if b else 1
        if not self.door_open:
            g[28:34, 33] = 8                          # wall just past corridor cell 6 (x=32)
        return g

    def obs(self):
        return Observation(frames=(self.grid(),), state=self.state, levels_completed=self.level,
                           win_levels=len(self.levels), available_actions=(1, 2, 3, 4, 5),
                           actions_used=self.actions_used, level_actions_used=self.level_actions)

    def step(self, a: int):
        self.actions_used += 1
        self.level_actions += 1
        if a == 0:
            self._start_level()
            self.state = "NOT_FINISHED"
            return self.obs()
        L = self.cfg
        if a == 5:
            self.count5 += 1
            if L.get("delay") and not self.jammed and self.count5 >= L["delay"]:
                self.door_open = True
            self.sw[2] ^= 1 if not L.get("delay") else 0
        else:
            self.count5 = 0
        if a == 1:
            if L.get("jam") and self.prev1:
                self.jammed = True
            self.sw[0] ^= 1
        self.prev1 = a == 1
        if a == 2:
            self.sw[1] ^= 1
        if a == 4:
            nx = self.x + 4
            if nx >= 33 and self.x < 33 and not self.door_open:
                nx = self.x
            if L.get("coupled") and self.sw[0] and self.x == 28:
                nx = 8
            self.x = nx
        if a == 3:
            self.x = max(8, self.x - 4)
        if self.x >= 56:
            self.level += 1
            self.level_actions = 0
            if self.level >= len(self.levels):
                self.state = "WIN"
                self.level -= 1
                o = self.obs()
                return Observation(**{**o.__dict__, "levels_completed": len(self.levels)})
            self._start_level()
        return self.obs()


def play(agent, game, max_actions=3000):
    obs = game.obs()
    while obs.state != "WIN" and game.actions_used < max_actions:
        act = agent.act(obs, deadline=float("inf"))
        if act is None:
            break
        obs = game.step(act.action)
    return {"levels": obs.levels_completed, "actions": game.actions_used, "won": obs.state == "WIN"}
