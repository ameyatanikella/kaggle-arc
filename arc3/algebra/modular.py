"""Linear systems over GF(2) and Z_m (m not necessarily prime).

Gaussian elimination gives feasibility and *a* solution, not the action-minimal one.
Minimum-cost solutions are found by enumerating the (small) solution set exactly;
when it is too large we return the best found within a bound and say so.
"""
from __future__ import annotations

from itertools import product
from typing import Optional, Sequence


# ---------------------------------------------------------------- GF(2)
def gf2_rref(A: Sequence[Sequence[int]], b: Optional[Sequence[int]] = None):
    rows = [[x & 1 for x in r] + ([b[i] & 1] if b is not None else []) for i, r in enumerate(A)]
    ncol = len(A[0]) if A else 0
    piv: list[int] = []
    r = 0
    for c in range(ncol):
        p = next((i for i in range(r, len(rows)) if rows[i][c]), None)
        if p is None:
            continue
        rows[r], rows[p] = rows[p], rows[r]
        for i in range(len(rows)):
            if i != r and rows[i][c]:
                rows[i] = [a ^ x for a, x in zip(rows[i], rows[r])]
        piv.append(c)
        r += 1
    return rows, piv, r


def gf2_solve_all(A: Sequence[Sequence[int]], b: Sequence[int]):
    """Return (particular_solution, nullspace_basis) or None if infeasible.
    A is (equations x unknowns): A u = b (mod 2)."""
    rows, piv, r = gf2_rref(A, b)
    n = len(A[0]) if A else 0
    for i in range(r, len(rows)):
        if rows[i][n]:
            return None
    x = [0] * n
    for i, c in enumerate(piv):
        x[c] = rows[i][n]
    free = [j for j in range(n) if j not in piv]
    basis = []
    for f in free:
        v = [0] * n
        v[f] = 1
        for i, c in enumerate(piv):
            v[c] = rows[i][f]
        basis.append(v)
    return x, basis


def gf2_min_cost(A, b, cost: Optional[Sequence[float]] = None, max_enum_bits: int = 20):
    """Exact minimum-cost solution when nullity <= max_enum_bits (enumerates 2^nullity).
    Returns dict(solution, cost, exact, n_solutions) or None if infeasible."""
    res = gf2_solve_all(A, b)
    if res is None:
        return None
    x0, basis = res
    n = len(x0)
    w = list(cost) if cost is not None else [1.0] * n
    k = len(basis)
    if k > max_enum_bits:
        # bounded: greedy improvement from the particular solution (not exact)
        best = x0[:]
        improved = True
        while improved:
            improved = False
            for v in basis:
                cand = [a ^ c for a, c in zip(best, v)]
                if sum(wi for wi, ci in zip(w, cand) if ci) < sum(wi for wi, ci in zip(w, best) if ci):
                    best, improved = cand, True
        return {"solution": best, "cost": sum(wi for wi, ci in zip(w, best) if ci), "exact": False,
                "n_solutions": 2 ** k}
    best, best_c = None, None
    for mask in range(2 ** k):
        x = x0[:]
        for j in range(k):
            if mask >> j & 1:
                x = [a ^ c for a, c in zip(x, basis[j])]
        c = sum(wi for wi, xi in zip(w, x) if xi)
        if best_c is None or c < best_c:
            best, best_c = x, c
    return {"solution": best, "cost": best_c, "exact": True, "n_solutions": 2 ** k}


# ---------------------------------------------------------------- Z_m
def zm_solve_min(A: Sequence[Sequence[int]], b: Sequence[int], m: int,
                 cost: Optional[Sequence[float]] = None, max_states: int = 2_000_000):
    """Minimum-cost u in {0..m-1}^n with A u = b (mod m) by exact enumeration with
    pruning (works for any m, prime or not). Cost of u_j presses is cost_j * u_j.
    Returns dict(solution, cost, exact) or None if infeasible within the bound."""
    n = len(A[0]) if A else 0
    if m ** n > max_states:
        raise ValueError(f"search space m^n={m ** n} exceeds max_states; reduce variables first")
    w = list(cost) if cost is not None else [1.0] * n
    best, best_c = None, None
    for u in product(range(m), repeat=n):
        if all(sum(a * x for a, x in zip(row, u)) % m == bi % m for row, bi in zip(A, b)):
            c = sum(wi * ui for wi, ui in zip(w, u))
            if best_c is None or c < best_c:
                best, best_c = list(u), c
    if best is None:
        return None
    return {"solution": best, "cost": best_c, "exact": True}
