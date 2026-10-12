"""Distinguishing experiments between executable world models.

`shortest_distinguishing_sequence`: BFS over the product state space of k
deterministic models from *known* current states. If it returns a sequence
before the bound, that sequence is a shortest separating one. Returning None
after exhausting a finite product space proves no separating sequence exists
*in the modelled space*; returning None because of the bound proves nothing.
Under partial observability this is not exact (belief states are not modelled).
"""
from __future__ import annotations

from collections import deque
from typing import Callable, Hashable, Optional, Sequence

Step = Callable[[Hashable, Hashable], tuple]   # (state, action) -> (next_state, observation)


def shortest_distinguishing_sequence(models: Sequence[Step], states: Sequence[Hashable],
                                     actions: Sequence[Hashable], max_depth: int = 8,
                                     max_nodes: int = 200_000,
                                     action_cost: Optional[Callable[[Hashable], float]] = None) -> dict:
    """Returns {"sequence": [...]|None, "exhausted": bool, "nodes": n, "split": {obs: [model idx]}}."""
    start = tuple(states)
    q = deque([(start, [])])
    seen = {start}
    nodes = 0
    bounded = False
    while q:
        tup, path = q.popleft()
        if len(path) >= max_depth:
            bounded = True
            continue
        for a in actions:
            nexts, obs = [], []
            ok = True
            for m, s in zip(models, tup):
                try:
                    ns, o = m(s, a)
                except Exception:
                    ns, o = ("<error>", "<error>")
                nexts.append(ns)
                obs.append(o)
            nodes += 1
            if len(set(map(_h, obs))) > 1:
                split: dict = {}
                for i, o in enumerate(obs):
                    split.setdefault(_h(o), []).append(i)
                return {"sequence": path + [a], "exhausted": False, "nodes": nodes, "split": split}
            nt = tuple(nexts)
            if nt not in seen:
                if len(seen) >= max_nodes:
                    bounded = True
                    continue
                seen.add(nt)
                q.append((nt, path + [a]))
    return {"sequence": None, "exhausted": not bounded, "nodes": nodes, "split": None}


def _h(o):
    try:
        hash(o)
        return o
    except TypeError:
        return repr(o)


def trace(model: Step, state, seq):
    out = []
    for a in seq:
        state, o = model(state, a)
        out.append(o)
    return out


def markov_conflicts(transitions: Sequence[tuple]) -> list:
    """transitions: (obs_key, action, next_obs_key). Same (obs, action) with different
    next observations refutes any model that is Markov in the visible frame: look for
    hidden state (timer, mode, selection, inventory) before raising model degree."""
    seen: dict = {}
    conflicts = []
    for o, a, o2 in transitions:
        k = (o, a)
        if k in seen and seen[k] != o2:
            conflicts.append((o, a, seen[k], o2))
        seen.setdefault(k, o2)
    return conflicts


def commutes_on(states: Sequence, f: Callable, g: Callable) -> tuple[bool, list]:
    """Empirical commutation check of two (non-invertible) transformations on given
    states. True only says 'no counterexample among these states'."""
    bad = [s for s in states if f(g(s)) != g(f(s))]
    return (len(bad) == 0), bad


def fit_guarded_affine(xs: Sequence[float], ys: Sequence[float]):
    """1-D guarded affine fit y = a*x+b on x<t, y = c on x>=t (saturation/collision),
    exact on the data. Returns the simplest exact model or None (MDL: plain affine first)."""
    pts = list(zip(xs, ys))
    def affine(points):
        if len(points) < 2:
            return (0.0, points[0][1]) if points else (0.0, 0.0)
        (x0, y0), (x1, y1) = points[0], points[-1]
        if x1 == x0:
            return None
        a = (y1 - y0) / (x1 - x0)
        b = y0 - a * x0
        return (a, b) if all(abs(a * x + b - y) < 1e-9 for x, y in points) else None
    pts.sort()
    lin = affine(pts)
    if lin is not None:
        return {"kind": "affine", "a": lin[0], "b": lin[1]}
    for k in range(1, len(pts)):
        left, right = pts[:k], pts[k:]
        al = affine(left)
        if al is None:
            continue
        c = right[0][1]
        if all(abs(y - c) < 1e-9 for _, y in right):
            return {"kind": "guarded", "a": al[0], "b": al[1], "threshold": right[0][0], "sat": c}
    return None
