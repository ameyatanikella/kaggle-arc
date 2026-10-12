"""Structure of a component's parameter-to-observation map: rank, circuits, relations.

Rows are the derivatives of each observation with respect to the parameters:
  * linear components (over Q or GF(2)): the exact design row; no randomness.
  * polynomial components: the Jacobian row evaluated at a random integer point. This gives
    the rank in the algebraic matroid with P(underestimate) <= D/|S| per point (Schwartz-Zippel),
    where D bounds the degree of the relevant minor and |S| is the size of the sampling set.
    Both are recorded with the result; two independent points are used and the max rank kept.

The matroid only proposes candidate dependent sets (circuits). Consistency is decided by the
actual relation: a left-null vector (linear) or an implicit polynomial (polynomial components).
"""
from __future__ import annotations

import math
import pickle
import random
import time
from fractions import Fraction
from itertools import combinations
from typing import Optional, Sequence

from arc3.algebra.linalg import nullspace, rank as q_rank
from arc3.diag.poly import Poly, implicit_relations

SAMPLE_SET = 2 ** 31 - 1


# ---------------------------------------------------------------- field-generic rank
def gf2_rank(rows: Sequence[Sequence[int]]) -> int:
    basis: dict = {}
    r = 0
    for row in rows:
        v = sum((int(x) & 1) << j for j, x in enumerate(row))
        while v:
            h = v.bit_length() - 1
            if h in basis:
                v ^= basis[h]
            else:
                basis[h] = v
                r += 1
                break
    return r


def rank(rows, field: str) -> int:
    if not rows:
        return 0
    return gf2_rank(rows) if field == "gf2" else q_rank(rows)


# ---------------------------------------------------------------- circuits
def is_circuit(rows, idx, field) -> bool:
    sub = [rows[i] for i in idx]
    if rank(sub, field) != len(idx) - 1:
        return False
    return all(rank([rows[i] for i in idx if i != j], field) == len(idx) - 1 for j in idx)


def fundamental_circuits(rows, field, order: Sequence[int]) -> list[tuple]:
    """Greedy basis in `order`; for every non-basis row its unique circuit with the basis."""
    out, basis = [], []
    for i in order:
        if rank([rows[j] for j in basis + [i]], field) > len(basis):
            basis.append(i)
            continue
        # unique representation of row i in the basis -> support
        if field == "gf2":
            sup = _gf2_support(rows, basis, i)
        else:
            sup = _q_support(rows, basis, i)
        out.append(tuple(sorted(sup + [i])))
    return out


def _q_support(rows, basis, i):
    if not basis:
        return []
    A = [list(col) for col in zip(*[rows[j] for j in basis])]       # columns = basis rows
    aug = [a + [b] for a, b in zip(A, rows[i])]
    ns = nullspace([list(r) for r in aug])
    v = next(x for x in ns if x[-1] != 0)
    return [basis[k] for k in range(len(basis)) if v[k] != 0]


def _gf2_support(rows, basis, i):
    if not basis:
        return []
    # eliminate with combination tracking
    vecs = [(sum((int(x) & 1) << j for j, x in enumerate(rows[b])), 1 << k) for k, b in enumerate(basis)]
    piv: dict = {}
    for v, comb in vecs:
        while v:
            h = v.bit_length() - 1
            if h in piv:
                v ^= piv[h][0]
                comb ^= piv[h][1]
            else:
                piv[h] = (v, comb)
                break
    t = sum((int(x) & 1) << j for j, x in enumerate(rows[i]))
    comb = 0
    while t:
        h = t.bit_length() - 1
        t ^= piv[h][0]
        comb ^= piv[h][1]
    return [basis[k] for k in range(len(basis)) if comb >> k & 1]


def all_circuits(rows, field, max_subsets: int = 60000, n_bases: int = 12, seed: int = 0):
    """All circuits by enumeration when C(n, <= r+1) is small, else the union of fundamental
    circuits for several basis orders (a cycle-space generating set, not every circuit).
    Returns (circuits, complete: bool)."""
    n = len(rows)
    r = rank(rows, field)
    total = sum(math.comb(n, s) for s in range(2, r + 2))
    if total <= max_subsets:
        found = []
        for s in range(1, r + 2):
            for idx in combinations(range(n), s):
                if is_circuit(rows, idx, field):
                    found.append(idx)
        return found, True
    rng = random.Random(seed)
    found = set(fundamental_circuits(rows, field, list(range(n))))
    for _ in range(n_bases - 1):
        order = list(range(n))
        rng.shuffle(order)
        found |= set(fundamental_circuits(rows, field, order))
    return sorted(found), False


# ---------------------------------------------------------------- relations
def linear_relation(rows, idx, field):
    """Coefficients w (over the circuit) with sum_i w_i * row_i = 0; unique up to scale."""
    if field == "gf2":
        return {i: 1 for i in idx}
    MT = [list(col) for col in zip(*[rows[i] for i in idx])]
    ns = nullspace(MT)
    assert len(ns) == 1, "a circuit has a one-dimensional relation space"
    return {i: c for i, c in zip(idx, ns[0])}


def jacobian_rows(fs: Sequence[Poly], n_params: int, seed: int = 0):
    """Exact Jacobian rows at two random integer points; returns (rows_at_best_point, meta)."""
    rng = random.Random(seed)
    best, best_r = None, -1
    for _ in range(2):
        pt = [rng.randrange(1, SAMPLE_SET) for _ in range(n_params)]
        rows = [[f.diff(j).eval(pt) for j in range(n_params)] for f in fs]
        r = q_rank(rows) if rows else 0
        if r > best_r:
            best, best_r = rows, r
    d = max((f.degree() for f in fs), default=1)
    D = min(n_params, len(fs)) * max(d - 1, 0)
    return best, {"points": 2, "minor_degree_bound": D, "sample_set": SAMPLE_SET,
                  "p_rank_underestimate_per_point": D / SAMPLE_SET}


def polynomial_relation(fs: Sequence[Poly], idx, max_degree=4):
    res = implicit_relations([fs[i] for i in idx], max_degree=max_degree)
    if res is None:
        return None
    D, rels = res
    return rels[0]


# ---------------------------------------------------------------- cache
class StructureCache:
    """Structural results keyed by (component, version, context signature, design signature).
    Each entry records the assumptions it relied on; invalidate(a) drops every entry that
    depends on assumption a. Costs are measured, not assumed."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.store: dict = {}
        self.deps: dict = {}
        self.stats = {"hits": 0, "misses": 0, "invalidated": 0, "compute_s": 0.0, "setup_s": 0.0}

    def get_or_compute(self, key, assumptions, fn):
        if self.enabled and key in self.store:
            self.stats["hits"] += 1
            return self.store[key]
        self.stats["misses"] += 1
        t0 = time.perf_counter()
        val = fn()
        self.stats["compute_s"] += time.perf_counter() - t0
        if self.enabled:
            self.store[key] = val
            for a in assumptions:
                self.deps.setdefault(a, set()).add(key)
        return val

    def invalidate(self, assumption) -> int:
        keys = self.deps.pop(assumption, set())
        n = 0
        for k in keys:
            if self.store.pop(k, None) is not None:
                n += 1
        self.stats["invalidated"] += n
        return n

    def memory_bytes(self) -> int:
        try:
            return len(pickle.dumps(self.store))
        except Exception:
            return -1
