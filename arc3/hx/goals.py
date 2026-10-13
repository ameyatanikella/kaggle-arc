"""Goal hypotheses from level transitions and object changes (no dynamics model).

A completed level is evidence about the goal, never proof. For each level the module compares the
WIN frame (last frame of the completed level, from the event log) with every decision state
visited on that level:
  * a fact is a predicate of one frame over the object parse (colours are categorical):
      present(c) / absent(c)          some / no pixel of colour c
      touch(c1, c2)                   an object of c1 and one of c2 have touching or overlapping boxes
      inside(c1, c2)                  an object of c1 lies within the box of an object of c2
      row(c1, c2) / col(c1, c2)       an object of c1 and one of c2 share a centre row / column
      shape(c1, c2)                   an object of c1 has the same cell pattern as one of c2
      count_eq(c1, c2)                as many objects of c1 as of c2
  * a goal candidate is a fact TRUE in the win frame and FALSE in the level's start state;
    its counter-evidence is the number of earlier visited states where it was already true
    without a win (0 = consistent with "sufficient"); its support grows with every level whose
    win it explains;
  * hazard candidates come the same way from death frames (true at death, false just before).

Transfer: at a new level the candidates are re-evaluated on its first frame. A candidate already
true there cannot be a sufficient goal of the new level (or the goal changed) and is flagged;
the rest are reported as unsatisfied goal hypotheses with a heuristic distance.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .objects import Obj, background, parse

MAX_OBJS = 120


def _boxes(objs: list) -> dict:
    by: dict = {}
    for o in objs:
        by.setdefault(o.color, []).append(o)
    return by


def _gap(a: Obj, b: Obj) -> int:
    dy = max(0, max(a.y0, b.y0) - min(a.y1, b.y1) - 1)
    dx = max(0, max(a.x0, b.x0) - min(a.x1, b.x1) - 1)
    return dy + dx


def _inside(a: Obj, b: Obj) -> bool:
    return b.y0 <= a.y0 and a.y1 <= b.y1 and b.x0 <= a.x0 and a.x1 <= b.x1 and a.size < b.size


def facts(grid: np.ndarray, mask: Optional[np.ndarray] = None) -> dict:
    """fact -> heuristic distance to being true (0 when true). Abstains (returns {}) when the
    parse abstains (too many components)."""
    g = np.asarray(grid)
    bg = background(g)
    objs = parse(g, mask=mask, bg=bg, max_objects=MAX_OBJS)
    if not objs:
        return {}
    by = _boxes(objs)
    colors = sorted(by)
    out: dict = {}
    present = set(np.unique(g[~mask] if mask is not None else g).tolist()) - {bg}
    for c in colors:
        out[("present", c)] = 0
    for c in range(16):
        if c != bg and c not in present:
            out[("absent", c)] = 0
    for i, c1 in enumerate(colors):
        for c2 in colors:
            if c1 == c2:
                continue
            A, B = by[c1], by[c2]
            if len(A) * len(B) > 900:
                continue
            gap = min(_gap(a, b) for a in A for b in B)
            if gap == 0:
                out[("touch", min(c1, c2), max(c1, c2))] = 0
            if any(_inside(a, b) for a in A for b in B):
                out[("inside", c1, c2)] = 0
            if c1 < c2:
                if any(a.center[0] == b.center[0] for a in A for b in B):
                    out[("row", c1, c2)] = 0
                if any(a.center[1] == b.center[1] for a in A for b in B):
                    out[("col", c1, c2)] = 0
                sa, sb = {a.shape for a in A}, {b.shape for b in B}
                if sa & sb:
                    out[("shape", c1, c2)] = 0
                if len(A) == len(B):
                    out[("count_eq", c1, c2)] = 0
    return out


def distance(fact: tuple, grid: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Heuristic distance of a frame from making `fact` true (0 = true). Used only to order
    exploration; never as evidence."""
    g = np.asarray(grid)
    kind = fact[0]
    if kind == "absent":
        m = g == fact[1]
        if mask is not None:
            m &= ~mask
        return float(m.sum())
    if kind == "present":
        return 0.0 if (g == fact[1]).any() else 1.0
    objs = parse(g, mask=mask, max_objects=MAX_OBJS)
    by = _boxes(objs)
    A, B = by.get(fact[1], []), by.get(fact[2], [])
    if not A or not B:
        return 64.0
    if kind in ("touch", "inside"):
        return float(min(_gap(a, b) for a in A for b in B))
    if kind == "row":
        return float(min(abs(a.center[0] - b.center[0]) for a in A for b in B))
    if kind == "col":
        return float(min(abs(a.center[1] - b.center[1]) for a in A for b in B))
    if kind == "count_eq":
        return float(abs(len(A) - len(B)))
    return 0.0 if fact in facts(g, mask) else 1.0


def describe(fact: tuple) -> str:
    k = fact[0]
    if k in ("present", "absent"):
        return f"{k}(colour {fact[1]})"
    return f"{k}(colour {fact[1]}, colour {fact[2]})"


class GoalModel:
    """Goal and hazard candidates accumulated over a game (mechanics live elsewhere)."""

    def __init__(self, mask_fn=None, max_visited: int = 400):
        self.mask_fn = mask_fn or (lambda: None)
        self.max_visited = max_visited
        self.cand: dict = {}        # fact -> {"support": [levels], "counter": int, "refuted_at": [levels]}
        self.hazard: dict = {}      # fact -> count
        self.levels_seen: list = []
        self.last_counter: dict = {}
        self.n = {"levels": 0, "transfers_checked": 0, "transfer_hits": 0, "abstain": 0}

    def observe_win(self, level: int, start: np.ndarray, visited: list, win: Optional[np.ndarray]) -> list:
        """Record a completed level. Returns the facts that explain this win (true at win, false at
        start), sorted by counter-evidence."""
        if win is None:
            self.n["abstain"] += 1
            return []
        mask = self.mask_fn()
        fw = facts(win, mask)
        fs = facts(start, mask)
        if not fw or not fs:
            self.n["abstain"] += 1
            return []
        new = [f for f in fw if f not in fs]
        if not new:
            self.n["abstain"] += 1
            return []
        vis = visited[-self.max_visited:]
        counter = {f: 0 for f in new}
        for g in vis:
            fg = facts(g, mask)
            for f in new:
                if f in fg:
                    counter[f] += 1
        # transfer check: did a candidate from earlier levels predict this win?
        for f, rec in self.cand.items():
            if rec["support"] and level not in rec["support"]:
                self.n["transfers_checked"] += 1
                if f in new:
                    self.n["transfer_hits"] += 1
        for f in new:
            rec = self.cand.setdefault(f, {"support": [], "counter": 0, "refuted_at": []})
            rec["support"].append(level)
            rec["counter"] += counter[f]
        self.levels_seen.append(level)
        self.n["levels"] += 1
        self.last_counter = counter
        return sorted(new, key=lambda f: (counter[f], f))

    def observe_death(self, pre: np.ndarray, death: np.ndarray) -> list:
        mask = self.mask_fn()
        fp, fd = facts(pre, mask), facts(death, mask)
        new = [f for f in fd if f not in fp]
        for f in new:
            self.hazard[f] = self.hazard.get(f, 0) + 1
        return new

    def ranked(self, limit: int = 8) -> list:
        """Candidates by (levels explained desc, counter-evidence asc)."""
        items = [(f, r) for f, r in self.cand.items() if not r["refuted_at"]]
        items.sort(key=lambda fr: (-len(fr[1]["support"]), fr[1]["counter"], fr[0]))
        return items[:limit]

    def at_level_start(self, level: int, init: np.ndarray, limit: int = 6) -> dict:
        """Re-test candidates on the first frame of a new level: already-true ones are flagged
        (cannot be a sufficient goal here, or the goal changed)."""
        mask = self.mask_fn()
        fi = facts(init, mask)
        open_, already = [], []
        for f, r in self.ranked(limit * 3):
            if f in fi:
                already.append(f)
            else:
                open_.append((f, distance(f, init, mask)))
        return {"open": open_[:limit], "already_true": already[:limit]}

    def satisfied_without_win(self, grid: np.ndarray, level: int) -> list:
        """Candidates true in a non-winning state of `level` (counter-evidence; call on visited
        states). Marks a candidate refuted when it has explained no level and is contradicted."""
        fg = facts(grid, self.mask_fn())
        hit = [f for f in self.cand if f in fg]
        for f in hit:
            self.cand[f]["counter"] += 1
        return hit

    def score(self, grid: np.ndarray, top: int = 3) -> float:
        """Exploration heuristic: sum of distances to the top candidates (lower is closer)."""
        mask = self.mask_fn()
        r = self.ranked(top)
        if not r:
            return 0.0
        return float(sum(distance(f, grid, mask) for f, _ in r))
