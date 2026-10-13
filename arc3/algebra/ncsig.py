"""Exact zero test / equality for division-free NONCOMMUTATIVE polynomial expressions.

Scope (the only language this module makes claims about):
  formulas built from noncommuting variables x_1..x_n, rational constants, + and *, compiled
  to acyclic algebraic branching programs (ABPs). Guards, modulo, saturation, stochastic
  effects, hidden counters and executable code are OUTSIDE this equivalence claim.

Exact signature (motivated by the rational hitting-matrix construction cited in
docs/prereg_b3.md):
  T_i[r, q] = (-1)^(r-q-1) / (r * i^(r-q)) for q < r, otherwise 0. The signature of f is
  f(T_1..T_n) e_0 in Q^d, with d = (n-1) m(m-1)/2 + m for an ABP with m vertices. It is zero
  iff f is the zero polynomial, within that size promise.
  T_i v is applied in O(d) rational operations with no matrix stored (division by z+i then
  integration):
      q_{-1} = 0; for r = 0..d-2: q_r = (v_r - q_{r-1}) / i; out[r+1] = q_r / (r+1).
  An ABP is evaluated backwards from the sink, one vector per vertex.
  Comparing two programs uses ONE dimension sized for their difference (m1 + m2 vertices).
  Independently sized fingerprints are never compared.
Free-algebra identity is not environment identity. For example, in a toggle system A^2 = I,
but A^2 - I is a nonzero free polynomial.
"""
from __future__ import annotations

import random
from collections import Counter
from fractions import Fraction as F


# ------------------------------------------------------------------ expressions
# ('var', i) i>=1 | ('const', c) | ('add', e1, e2) | ('mul', e1, e2) | ('neg', e)
def var(i):
    return ("var", i)


def const(c):
    return ("const", F(c))


def add(a, b):
    return ("add", a, b)


def mul(a, b):
    return ("mul", a, b)


def sub(a, b):
    return ("add", a, ("neg", b))


def expand(e) -> dict:
    """Independent check / ordinary canonicalization: {word (tuple of var ids): coefficient}."""
    t = e[0]
    if t == "var":
        return {(e[1],): F(1)}
    if t == "const":
        return {(): e[1]} if e[1] else {}
    if t == "neg":
        return {w: -c for w, c in expand(e[1]).items()}
    a, b = expand(e[1]), expand(e[2])
    out: Counter = Counter()
    if t == "add":
        for w, c in a.items():
            out[w] += c
        for w, c in b.items():
            out[w] += c
    else:
        for w1, c1 in a.items():
            for w2, c2 in b.items():
                out[w1 + w2] += c1 * c2
    return {w: c for w, c in out.items() if c}


# ------------------------------------------------------------------ formula -> ABP
def to_abp(e):
    """Series-parallel compilation with endpoint identification (no connector vertices):
    product = identify sink(P) with source(Q); sum = identify both sources and both sinks;
    negation = scale the edges leaving the source (every path uses exactly one of them).
    Returns (edges, m); vertices are topologically ordered with 0 = source and m-1 = sink;
    an edge is (src, dst, var, coeff) and var 0 marks a scalar edge."""
    parent: list = []
    edges: list = []

    def new():
        parent.append(len(parent))
        return len(parent) - 1

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    def merge(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
        return ra

    def build(x):
        t = x[0]
        if t in ("var", "const"):
            s, d = new(), new()
            edges.append([s, d, x[1] if t == "var" else 0, F(1) if t == "var" else x[1]])
            return s, d
        if t == "neg":
            s, d = build(x[1])
            for ed in edges:
                if find(ed[0]) == find(s):
                    ed[3] = -ed[3]
            return s, d
        s1, d1 = build(x[1])
        s2, d2 = build(x[2])
        if t == "add":
            return merge(s1, s2), merge(d1, d2)
        merge(d1, s2)                                       # mul: series composition
        return find(s1), find(d2)

    s, d = build(e)
    s, d = find(s), find(d)
    E = [(find(a), find(b), i, c) for a, b, i, c in edges if c != 0]
    verts = sorted({v for a, b, _, _ in E for v in (a, b)} | {s, d})
    order = _topo(verts, E, s)
    pos = {v: k for k, v in enumerate(order)}
    assert pos[s] == 0 and pos[d] == len(order) - 1, "compiled program must have a unique source/sink"
    return [(pos[a], pos[b], i, c) for a, b, i, c in E], len(order)


def _topo(verts, edges, s):
    indeg = {v: 0 for v in verts}
    out = {v: [] for v in verts}
    for a, b, _, _ in edges:
        out[a].append(b)
        indeg[b] += 1
    ready = [v for v in verts if indeg[v] == 0]
    ready.sort(key=lambda v: v != s)
    res = []
    while ready:
        v = ready.pop(0)
        res.append(v)
        for w in out[v]:
            indeg[w] -= 1
            if indeg[w] == 0:
                ready.append(w)
    return res


def abp_expand(edges, m) -> dict:
    vals = [Counter() for _ in range(m)]
    vals[m - 1][()] = F(1)
    by_src: dict = {}
    for a, b, i, c in edges:
        by_src.setdefault(a, []).append((b, i, c))
    for a in range(m - 2, -1, -1):
        for b, i, c in by_src.get(a, ()):
            for w, k in vals[b].items():
                vals[a][((i,) + w) if i else w] += c * k
    return {w: c for w, c in vals[0].items() if c}


# ------------------------------------------------------------------ exact signature
def bound(m: int, n: int) -> int:
    return (n - 1) * m * (m - 1) // 2 + m


def t_apply(i: int, v: list) -> list:
    q = F(0)
    out = [F(0)] * len(v)
    for r in range(len(v) - 1):
        q = (v[r] - q) / i
        out[r + 1] = q / (r + 1)
    return out


def signature(edges, m: int, n: int, d: int) -> list:
    if d < bound(m, n):
        raise ValueError(f"dimension {d} below the size promise {bound(m, n)} for m={m}, n={n}")
    vals = [None] * m
    vals[m - 1] = [F(1)] + [F(0)] * (d - 1)
    by_src: dict = {}
    for a, b, i, c in edges:
        by_src.setdefault(a, []).append((b, i, c))
    for a in range(m - 2, -1, -1):
        acc = [F(0)] * d
        for b, i, c in by_src.get(a, ()):
            vb = vals[b] if vals[b] is not None else [F(0)] * d
            v = t_apply(i, vb) if i else vb
            acc = [x + c * y for x, y in zip(acc, v)]
        vals[a] = acc
    return vals[0]


def is_zero(e, n: int) -> bool:
    edges, m = to_abp(e)
    return not any(signature(edges, m, n, bound(m, n)))


def equal(e1, e2, n: int) -> bool:
    """Exact equality as free noncommutative polynomials: one common dimension sized for the
    difference (m1 + m2 vertices)."""
    g1, m1 = to_abp(e1)
    g2, m2 = to_abp(e2)
    d = bound(m1 + m2, n)
    return signature(g1, m1, n, d) == signature(g2, m2, n, d)


# ------------------------------------------------------------------ established baseline: random matrices
def random_matrix_test(e, n: int, degree: int, p: int = 1000003, trials: int = 2, seed: int = 0) -> bool:
    """Randomized zero test (Amitsur-Levitzki size k = degree//2 + 1, Schwartz-Zippel over F_p):
    returns True = 'zero' (false only with probability <= degree/p per trial)."""
    import numpy as np
    k = degree // 2 + 1
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        mats = {i: rng.integers(0, p, size=(k, k), dtype=np.int64) for i in range(1, n + 1)}

        def ev(x):
            t = x[0]
            if t == "var":
                return mats[x[1]]
            if t == "const":
                c = x[1]
                val = (c.numerator * pow(c.denominator, -1, p)) % p
                return (np.eye(k, dtype=np.int64) * val) % p
            if t == "neg":
                return (-ev(x[1])) % p
            a, b = ev(x[1]), ev(x[2])
            return (a + b) % p if t == "add" else (a @ b) % p
        if ev(e).any():
            return False
    return True


def degree_of(e) -> int:
    t = e[0]
    if t == "var":
        return 1
    if t == "const":
        return 0
    if t == "neg":
        return degree_of(e[1])
    a, b = degree_of(e[1]), degree_of(e[2])
    return max(a, b) if t == "add" else a + b


def random_formula(rng: random.Random, n: int, size: int):
    if size <= 1:
        return var(rng.randint(1, n)) if rng.random() < 0.8 else const(rng.choice([-2, -1, 1, 2]))
    left = rng.randint(1, size - 1)
    op = rng.choice(["add", "mul", "add"])
    a, b = random_formula(rng, n, left), random_formula(rng, n, size - left)
    return ("add", a, b) if op == "add" else ("mul", a, b)
