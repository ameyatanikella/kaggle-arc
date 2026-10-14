"""Relational scene matching (HX2): camera translation and neighbourhood-refined object identity.

camera_shift(g0, g1)  the global translation (dy, dx) that best aligns two consecutive decision
                      frames, estimated from the non-background, non-HUD pixels. Returns None
                      (abstain) unless the best shift is non-zero, explains >= 85% of the pixels and
                      beats the zero shift by >= 0.25 and the runner-up by >= 0.05.
refined_match(a, b)   object correspondence where identical objects (same colour and shape) are
                      told apart by a 1-step Weisfeiler-Leman refinement: each object's label is
                      extended with the sorted multiset of (neighbour label, direction) for objects
                      whose boxes are within 1 cell. Matching stays abstaining when refined labels
                      still collide. A refinement cannot separate every pair of scenes (regular
                      configurations defeat it), so ambiguity remains a reported outcome.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .objects import Match, Obj, background, match, parse


def camera_shift(g0: np.ndarray, g1: np.ndarray, mask: Optional[np.ndarray] = None, max_shift: int = 12):
    a, b = np.asarray(g0), np.asarray(g1)
    bg = background(a)
    valid = a != bg
    if mask is not None:
        valid &= ~mask
    if valid.sum() < 20:
        return None
    H, W = a.shape
    best = []
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            ys0, ys1 = max(0, -dy), min(H, H - dy)
            xs0, xs1 = max(0, -dx), min(W, W - dx)
            src = a[ys0:ys1, xs0:xs1]
            dst = b[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
            v = valid[ys0:ys1, xs0:xs1]
            n = int(v.sum())
            if n < 20:
                continue
            best.append((float((src[v] == dst[v]).sum()) / n, dy, dx))
    if not best:
        return None
    best.sort(reverse=True)
    zero = next((s for s, dy, dx in best if dy == 0 and dx == 0), 0.0)
    s0, dy, dx = best[0]
    s1 = best[1][0] if len(best) > 1 else 0.0
    if (dy, dx) == (0, 0) or s0 < 0.85 or s0 - zero < 0.25 or s0 - s1 < 0.05:
        return None
    return dy, dx, round(s0, 3)


def _gap(a: Obj, b: Obj) -> int:
    dy = max(0, max(a.y0, b.y0) - min(a.y1, b.y1) - 1)
    dx = max(0, max(a.x0, b.x0) - min(a.x1, b.x1) - 1)
    return dy + dx


def refined_labels(objs: list) -> list:
    out = []
    for o in objs:
        nb = []
        for p in objs:
            if p is o or _gap(o, p) > 1:
                continue
            d = (int(np.sign(p.center[0] - o.center[0])), int(np.sign(p.center[1] - o.center[1])))
            nb.append((p.color, p.shape, d))
        out.append((o.sig, tuple(sorted(nb))))
    return out


def refined_match(before: list, after: list) -> Match:
    """Like objects.match, but ambiguous identities are retried with refined labels."""
    m = match(before, after)
    if not m.ambiguous:
        return m
    lb = dict(zip(map(id, before), refined_labels(before)))
    la = dict(zip(map(id, after), refined_labels(after)))
    still = []
    moved = list(m.moved)
    used = set()
    for ob, cands in m.ambiguous:
        same = [oa for oa in cands if la[id(oa)] == lb[id(ob)] and id(oa) not in used]
        others = [x for x, _ in m.ambiguous if x is not ob and lb[id(x)] == lb[id(ob)]]
        if len(same) == 1 and not others:
            oa = same[0]
            used.add(id(oa))
            moved.append((ob, oa, oa.y0 - ob.y0, oa.x0 - ob.x0))
        else:
            still.append((ob, cands))
    return Match(moved, still, m.appeared, m.vanished, m.recolored, m.unchanged)
