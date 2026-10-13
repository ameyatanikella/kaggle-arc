"""State abstraction checks.

Complete diagnostic model (synthetic controls):
  bisimulation_quotient() is ordinary partition refinement. Two states are interchangeable
  only if their retained observation (label: goal status, visible features, enabled actions)
  agrees and every permitted action leads to equivalent states. check_quotient() re-verifies
  every state-action edge against the quotient. A visible-frame key is NOT sufficient: a hidden
  counter keeps states apart even when their frames are equal.
Learned model (the agent):
  abstractions are provisional. The temporal graph refines on transition conflicts (graph.py).
  Dependency-projected plan reuse (plans.py) treats pixels outside a plan's dependency region
  as nuisance for that plan only, and validates every step on the region. A counterexample
  (e.g. a coupled object outside the region changing the outcome) invalidates the entry.
  No certificate derived from a learned model is claimed to be unconditionally correct.
"""
from __future__ import annotations

import numpy as np


def bisimulation_quotient(states, actions, step, label, enabled=None):
    """step(s, a) -> s' (or None if a is not enabled in s). Returns {state: block id}."""
    def lab0(s):
        en = tuple(a for a in actions if (enabled(s, a) if enabled else step(s, a) is not None))
        return (label(s), en)
    blocks = {s: lab0(s) for s in states}
    ids = {k: i for i, k in enumerate(dict.fromkeys(blocks.values()))}
    cur = {s: ids[blocks[s]] for s in states}
    while True:
        sig = {s: (cur[s], tuple(cur.get(step(s, a)) if step(s, a) is not None else None for a in actions))
               for s in states}
        ids = {k: i for i, k in enumerate(dict.fromkeys(sig.values()))}
        new = {s: ids[sig[s]] for s in states}
        if len(set(new.values())) == len(set(cur.values())):
            return new
        cur = new


def check_quotient(states, actions, step, blocks) -> int:
    """Number of state-action edges inconsistent with the quotient (0 = valid)."""
    seen: dict = {}
    bad = 0
    for s in states:
        for a in actions:
            t = step(s, a)
            k = (blocks[s], a)
            v = blocks.get(t) if t is not None else None
            if k in seen and seen[k] != v:
                bad += 1
            seen.setdefault(k, v)
    return bad


def dependency_region(frames, margin: int = 1) -> np.ndarray:
    """Pixels a plan touches: every pixel that changes between consecutive expected frames,
    dilated by `margin` (an approximation of its preconditions, refined on counterexamples)."""
    m = np.zeros(frames[0].shape, bool)
    for a, b in zip(frames, frames[1:]):
        m |= a != b
    if margin and m.any():
        d = m.copy()
        for dy in range(-margin, margin + 1):
            for dx in range(-margin, margin + 1):
                d |= np.roll(np.roll(m, dy, 0), dx, 1)
        m = d
    return m
