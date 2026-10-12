"""B1: deterministic state-graph explorer (CPU only, no learned model).

State identity is the frame with learned HUD/volatile pixels masked. Each state
has a candidate action set: the available simple actions, plus one click per
non-background connected component when ACTION6 is available (object-centred
clicks, never a 64x64 sweep). Policy:

  1. try an untried candidate in the current state;
  2. else follow the shortest path of known, non-fatal edges to the nearest
     state that still has untried candidates (replanned every step);
  3. else, if the level start can reach such a state, level-RESET (costs 1);
  4. else stop (no remaining evidence-gathering moves are known).

GAME_OVER marks the incoming edge fatal and is followed by RESET. Every real
action is charged by the trusted runner; nothing here assumes rollback.
"""
from __future__ import annotations

import random
from collections import deque
from typing import Optional

import numpy as np

from ..agent_api import Act, Agent
from ..perception import VolatileMask, background_color, components

FATAL = "FATAL"


class ExplorerAgent(Agent):
    name = "explorer"

    def __init__(self, game_id, config=None, seed=0):
        super().__init__(game_id, config, seed)
        c = self.config
        self.max_clicks = int(c.get("max_clicks", 48))
        self.level_action_cap = int(c.get("level_action_cap", 600))
        self.shuffle = bool(c.get("shuffle", False))
        self.rng = random.Random(seed)
        self.vmask = VolatileMask()
        self.level = -1
        self.prev_grid: Optional[np.ndarray] = None
        self.prev_node: Optional[int] = None
        self.last: Optional[tuple] = None
        self.stats = {"levels_seen": 0, "nodes": 0, "edges": 0, "fatal": 0, "resets": 0,
                      "plans": 0, "stop": None}
        self._new_level()

    # ---------------- graph bookkeeping ----------------
    def _new_level(self):
        self.grids: list[np.ndarray] = []      # node id -> raw grid
        self.raw_index: dict[bytes, int] = {}  # raw bytes -> node id
        self.edges: dict[tuple, object] = {}   # (node, act) -> node | FATAL
        self.cands: dict[int, list] = {}       # node -> candidate acts
        self.tried: dict[int, set] = {}
        self.mask_version = -1
        self.key_of: list[bytes] = []
        self.start_node: Optional[int] = None

    def _node(self, grid: np.ndarray) -> int:
        b = grid.tobytes()
        n = self.raw_index.get(b)
        if n is None:
            n = len(self.grids)
            self.raw_index[b] = n
            self.grids.append(grid.copy())
            self.key_of.append(self.vmask.key(grid))
            self.stats["nodes"] += 1
        return n

    def _refresh_keys(self):
        self.key_of = [self.vmask.key(g) for g in self.grids]

    def _abstract(self):
        """Merge raw nodes that are equal under the volatile mask."""
        rep: dict[bytes, int] = {}
        canon = []
        for i, k in enumerate(self.key_of):
            canon.append(rep.setdefault(k, i))
        return canon

    def _candidates(self, node: int, avail: tuple) -> list:
        if node in self.cands:
            return self.cands[node]
        acts = [(a,) for a in avail if a not in (0, 6)]
        if 6 in avail:
            g = self.grids[node]
            bg = background_color(g)
            comps = [c for c in components(g, skip_colors=(bg,)) if not self.vmask.mask[c.cy, c.cx]]
            comps.sort(key=lambda c: (c.size, c.y0, c.x0))
            seen = set()
            for c in comps:
                k = (c.cy, c.cx)
                if k in seen:
                    continue
                seen.add(k)
                acts.append((6, c.cx, c.cy))
                if len(acts) >= self.max_clicks + 7:
                    break
        if self.shuffle:
            self.rng.shuffle(acts)
        self.cands[node] = acts
        return acts

    def _untried(self, canon_node: int, canon, avail) -> list:
        # Union of tried sets over raw nodes mapping to this abstract node.
        tried = set()
        for raw, cn in enumerate(canon):
            if cn == canon_node:
                tried |= self.tried.get(raw, set())
        return [a for a in self._candidates(canon_node, avail) if a not in tried]

    def _bfs_to_frontier(self, src: int, canon, avail) -> Optional[list]:
        adj: dict[int, list] = {}
        for (n, a), m in self.edges.items():
            if m == FATAL:
                continue
            adj.setdefault(canon[n], []).append((a, canon[m]))
        frontier_cache: dict[int, bool] = {}

        def has_frontier(cn):
            if cn not in frontier_cache:
                frontier_cache[cn] = bool(self._untried(cn, canon, avail))
            return frontier_cache[cn]

        q = deque([src])
        prev = {src: None}
        while q:
            u = q.popleft()
            if u != src and has_frontier(u):
                path = []
                while prev[u] is not None:
                    pu, a = prev[u]
                    path.append(a)
                    u = pu
                return path[::-1]
            for a, v in adj.get(u, []):
                if v not in prev:
                    prev[v] = (u, a)
                    q.append(v)
        return None

    # ---------------- policy ----------------
    def act(self, obs, deadline):
        grid = obs.grid
        if obs.levels_completed != self.level:
            if self.level >= 0:
                self.stats["levels_seen"] += 1
            self.level = obs.levels_completed
            self._new_level()
            self.prev_node, self.last = None, None
        node = self._node(grid)
        if self.start_node is None:
            self.start_node = node
        # record the transition that led here
        if self.prev_node is not None and self.last is not None:
            self.tried.setdefault(self.prev_node, set()).add(self.last)
            if obs.state == "GAME_OVER":
                self.edges[(self.prev_node, self.last)] = FATAL
                self.stats["fatal"] += 1
            else:
                self.edges[(self.prev_node, self.last)] = node
                self.stats["edges"] = len(self.edges)
                old = self.vmask.mask.copy()
                self.vmask.update(self.prev_grid, grid)
                if not np.array_equal(old, self.vmask.mask):
                    self._refresh_keys()
        if obs.state == "GAME_OVER":
            self.prev_node, self.last, self.prev_grid = None, None, None
            self.stats["resets"] += 1
            return Act(0, why="reset after game over")
        if obs.level_actions_used >= self.level_action_cap:
            self.stats["stop"] = "level_action_cap"
            return None
        avail = obs.available_actions
        canon = self._abstract()
        here = canon[node]
        choice = None
        untried = self._untried(here, canon, avail)
        if untried:
            choice = untried[0]
        else:
            path = self._bfs_to_frontier(here, canon, avail)
            if path:
                self.stats["plans"] += 1
                choice = path[0]
            elif self.start_node is not None and here != canon[self.start_node] and obs.level_actions_used > 0:
                if self._bfs_to_frontier(canon[self.start_node], canon, avail) is not None or \
                        self._untried(canon[self.start_node], canon, avail):
                    self.prev_node, self.last, self.prev_grid = None, None, None
                    self.stats["resets"] += 1
                    return Act(0, why="reset to reach unexplored frontier")
        if choice is None:
            self.stats["stop"] = "exhausted"
            return None
        self.prev_node, self.last, self.prev_grid = node, choice, grid
        if choice[0] == 6:
            return Act(6, choice[1], choice[2])
        return Act(choice[0])
