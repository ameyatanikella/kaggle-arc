"""Exact rational linear algebra (fractions.Fraction) for small systems.

Floating-point thresholds are never used to *prove* a rank here; when an input is
float we convert with Fraction(x).limit_denominator(...) only if the caller asks.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Iterable, Optional, Sequence


def _frac_rows(rows: Iterable[Sequence]) -> list[list[Fraction]]:
    return [[Fraction(x) for x in r] for r in rows]


def rref(rows: Iterable[Sequence]) -> tuple[list[list[Fraction]], list[int]]:
    """Reduced row-echelon form over Q. Returns (R, pivot_columns)."""
    m = _frac_rows(rows)
    if not m:
        return [], []
    ncol = len(m[0])
    piv: list[int] = []
    r = 0
    for c in range(ncol):
        p = next((i for i in range(r, len(m)) if m[i][c] != 0), None)
        if p is None:
            continue
        m[r], m[p] = m[p], m[r]
        inv = 1 / m[r][c]
        m[r] = [x * inv for x in m[r]]
        for i in range(len(m)):
            if i != r and m[i][c] != 0:
                f = m[i][c]
                m[i] = [a - f * b for a, b in zip(m[i], m[r])]
        piv.append(c)
        r += 1
        if r == len(m):
            break
    return m[:r], piv


def rank(rows: Iterable[Sequence]) -> int:
    return len(rref(rows)[1])


class IncrementalRank:
    """Maintains an RREF basis; `gain(row)` is 1 iff row is independent of the basis."""

    def __init__(self, ncol: int):
        self.ncol = ncol
        self.basis: list[list[Fraction]] = []   # rows in RREF
        self.piv: list[int] = []

    def _reduce(self, row: Sequence) -> list[Fraction]:
        v = [Fraction(x) for x in row]
        for b, c in zip(self.basis, self.piv):
            if v[c] != 0:
                f = v[c]
                v = [a - f * x for a, x in zip(v, b)]
        return v

    def gain(self, row: Sequence) -> int:
        return int(any(x != 0 for x in self._reduce(row)))

    def add(self, row: Sequence) -> bool:
        v = self._reduce(row)
        c = next((i for i, x in enumerate(v) if x != 0), None)
        if c is None:
            return False
        inv = 1 / v[c]
        v = [x * inv for x in v]
        for i, b in enumerate(self.basis):
            if b[c] != 0:
                f = b[c]
                self.basis[i] = [a - f * x for a, x in zip(b, v)]
        self.basis.append(v)
        self.piv.append(c)
        return True

    @property
    def rank(self) -> int:
        return len(self.basis)


def solve(A: Sequence[Sequence], b: Sequence) -> Optional[list[Fraction]]:
    """One exact solution of A x = b over Q (free variables = 0), or None if inconsistent."""
    aug = [list(r) + [bi] for r, bi in zip(A, b)]
    R, piv = rref(aug)
    n = len(A[0]) if A else 0
    if n in piv:            # pivot in the augmented column -> inconsistent
        return None
    x = [Fraction(0)] * n
    for row, c in zip(R, piv):
        x[c] = row[n]
    return x


def nullspace(A: Sequence[Sequence]) -> list[list[Fraction]]:
    R, piv = rref(A)
    n = len(A[0]) if A else 0
    free = [j for j in range(n) if j not in piv]
    basis = []
    for f in free:
        v = [Fraction(0)] * n
        v[f] = Fraction(1)
        for row, c in zip(R, piv):
            v[c] = -row[f]
        basis.append(v)
    return basis


def left_dependencies(rows: Sequence[Sequence]) -> list[list[Fraction]]:
    """Vectors w with w^T M = 0: linear relations among the *rows* (measurements).
    In a linear-in-parameters model y = M theta, each w gives the consistency
    check w^T y = 0 on predicted outputs (an algebraic circuit of the row matroid)."""
    if not rows:
        return []
    MT = [list(col) for col in zip(*rows)]
    return nullspace(MT)
