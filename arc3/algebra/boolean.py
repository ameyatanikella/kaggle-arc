"""Multilinear (Boolean-cube) models  f(b) = sum_S theta_S prod_{i in S} b_i.

Over the reals every f:{0,1}^n -> R has a unique multilinear representation
(b_i^2 = b_i). Exact identities on {0,1}: AND = xy, OR = x+y-xy, XOR = x+y-2xy,
toggle(b,u) = b+u-2bu. A complete truth table determines f; the Mobius
coefficients are theta_S = sum_{T subseteq S} (-1)^{|S|-|T|} f(1_T).
Low-degree sparsity is an *assumption* to be tested, not a guarantee.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import Iterable, Optional, Sequence

from .linalg import IncrementalRank, rank, solve


def monomials(n: int, max_degree: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    for d in range(0, max_degree + 1):
        out.extend(combinations(range(n), d))
    return out


def feature_row(b: Sequence[int], monos: Sequence[tuple[int, ...]]) -> list[int]:
    return [int(all(b[i] for i in S)) for S in monos]


def mobius_coefficients(table: dict[tuple[int, ...], float], n: int) -> dict[tuple[int, ...], Fraction]:
    """Exact multilinear coefficients from a COMPLETE truth table {b: f(b)}."""
    if len(table) != 2 ** n:
        raise ValueError("Mobius inversion needs the complete truth table")
    coef: dict[tuple[int, ...], Fraction] = {}
    for S in monomials(n, n):
        tot = Fraction(0)
        for k in range(len(S) + 1):
            for T in combinations(S, k):
                b = tuple(1 if i in T else 0 for i in range(n))
                tot += (-1) ** (len(S) - k) * Fraction(table[b])
        if tot != 0:
            coef[S] = tot
    return coef


def interaction_estimable(observed: Iterable[Sequence[int]], monos: Sequence[tuple[int, ...]]) -> dict:
    """Which coefficients are identifiable from the observed configurations?
    Returns rank and, per monomial, whether its unit vector lies in the row space."""
    rows = [feature_row(b, monos) for b in observed]
    r = rank(rows) if rows else 0
    ident = {}
    for j, S in enumerate(monos):
        e = [0] * len(monos)
        e[j] = 1
        ident[S] = rank(rows + [e]) == r if rows else False
    return {"rank": r, "n_params": len(monos), "identifiable": ident}


def best_interaction_probe(observed: Sequence[Sequence[int]], candidates: Sequence[Sequence[int]],
                           monos: Sequence[tuple[int, ...]], cost: Optional[dict] = None) -> Optional[tuple]:
    """Cheapest reachable configuration whose feature row increases the rank.
    `cost[b]` must include travel/setup cost; unreachable configs are omitted by the caller."""
    inc = IncrementalRank(len(monos))
    for b in observed:
        inc.add(feature_row(b, monos))
    best = None
    for b in candidates:
        if inc.gain(feature_row(b, monos)):
            c = (cost or {}).get(tuple(b), 1)
            if best is None or c < best[1]:
                best = (tuple(b), c)
    return best


def fit_exact(observations: Sequence[tuple[Sequence[int], float]], monos: Sequence[tuple[int, ...]]):
    """Exact rational least-norm-free solution (free vars 0) if consistent, else None."""
    A = [feature_row(b, monos) for b, _ in observations]
    y = [Fraction(v) for _, v in observations]
    return solve(A, y)


def sparse_fit(observations: Sequence[tuple[Sequence[int], float]], monos: Sequence[tuple[int, ...]],
               max_support: int = 3) -> list[dict]:
    """Enumerate supports of size <= max_support (smallest first); return every
    support that fits all observations exactly. Several survivors = unidentified."""
    A = [feature_row(b, monos) for b, _ in observations]
    y = [Fraction(v) for _, v in observations]
    out = []
    for k in range(0, max_support + 1):
        for supp in combinations(range(len(monos)), k):
            sub = [[row[j] for j in supp] for row in A]
            if k == 0:
                if all(v == 0 for v in y):
                    out.append({"support": (), "theta": {}})
                continue
            x = solve(sub, y)
            if x is None:
                continue
            if all(sum(r[i] * x[i] for i in range(k)) == yy for r, yy in zip(sub, y)):
                out.append({"support": tuple(monos[j] for j in supp),
                            "theta": {monos[j]: x[i] for i, j in enumerate(supp)}})
        if out:
            return out     # minimal description length first
    return out


def additivity_check(d00: float, d10: float, d01: float, d11: float) -> Fraction:
    """Second difference d11-d10-d01+d00; nonzero refutes an additive 2-switch model."""
    return Fraction(d11) - Fraction(d10) - Fraction(d01) + Fraction(d00)
