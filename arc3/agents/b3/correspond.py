"""Object correspondence with retained alternatives (component O in docs/prereg_b3.md).

Identical-looking objects (the same exact shape) make frame-to-frame identity ambiguous.
  commit     assign by nearest previous position (ordinary greedy tracking)
  retain     keep every feasible assignment (bijections between the k tracked identities and the
             k instances; k is small, so plain enumeration, no matching-counting machinery).
             Each hypothesis carries the per-(action, identity) displacements it implies. A
             hypothesis is pruned when its implied displacement contradicts what it learned
             earlier for that action and identity (zero displacement = blocked, always allowed).
             Survivors are weighted by a motion prior (smaller total displacement is likelier):
             a weight, not a posterior over the full world model.
An observation of identity j is usable when every surviving hypothesis agrees on its position.
"""
from __future__ import annotations

import itertools
import math


class Retained:
    def __init__(self, k: int, max_hyps: int = 32):
        self.k = k
        self.hyps = None          # list of (assignment tuple, learned {(a, j): d}, logw)
        self.max_hyps = max_hyps
        self.contradictions = 0

    def start(self, positions):
        self.prev = list(positions)
        self.hyps = [(tuple(range(self.k)), {}, 0.0)]     # identities = initial instance order

    def update(self, action, positions, gate: float = 12.0, max_assign: int = 64):
        """positions: k instance positions (y, x) in the new frame. Returns resolved positions per
        identity (None where hypotheses disagree). Assignments are enumerated by backtracking
        over motion-gated candidates (instances within `gate` px), at most `max_assign` per
        hypothesis: ordinary gated data association, bounded."""
        self.enumerated = 0
        new = []
        for assign, learned, logw in self.hyps:
            prev_pos = [self.prev[assign[j]] for j in range(self.k)]
            cands = [[i for i in range(self.k) if math.dist(prev_pos[j], positions[i]) <= gate]
                     for j in range(self.k)]
            out_perms = []

            def bt(j, used, perm):
                if len(out_perms) >= max_assign:
                    return
                if j == self.k:
                    out_perms.append(tuple(perm))
                    return
                for i in sorted(cands[j], key=lambda i: math.dist(prev_pos[j], positions[i])):
                    if i not in used:
                        used.add(i)
                        perm.append(i)
                        bt(j + 1, used, perm)
                        perm.pop()
                        used.discard(i)
            bt(0, set(), [])
            self.enumerated += len(out_perms)
            for perm in out_perms:
                ok, L2, cost = True, dict(learned), 0.0
                for j in range(self.k):
                    d = (positions[perm[j]][0] - prev_pos[j][0], positions[perm[j]][1] - prev_pos[j][1])
                    cost += abs(d[0]) + abs(d[1])
                    key = (action, j)
                    if d != (0, 0):
                        if key in L2 and L2[key] != d:
                            ok = False
                            break
                        L2[key] = d
                if ok:
                    new.append((perm, L2, logw - cost / 4.0))
        if not new:
            self.contradictions += 1
            new = [(tuple(range(self.k)), {}, 0.0)]           # model inadequate here: restart
        new.sort(key=lambda h: -h[2])
        # merge duplicates (same assignment and learned model)
        seen, keep = set(), []
        for h in new:
            sig = (h[0], tuple(sorted(h[1].items())))
            if sig not in seen:
                seen.add(sig)
                keep.append(h)
        self.hyps = keep[:self.max_hyps]
        self.prev = list(positions)
        out = []
        for j in range(self.k):
            ps = {positions[h[0][j]] for h in self.hyps}
            out.append(next(iter(ps)) if len(ps) == 1 else None)
        return out


def commit_nearest(prev_positions, positions):
    """Greedy nearest assignment (ordinary tracking); returns positions ordered by identity."""
    used, out = set(), []
    for p in prev_positions:
        best = min((i for i in range(len(positions)) if i not in used),
                   key=lambda i: math.dist(p, positions[i]))
        used.add(best)
        out.append(positions[best])
    return out
