"""Exact multivariate polynomials over Q and implicit relations among them.

A component's observation function y_i = f_i(theta) (context constants already substituted)
is a Poly in the parameters. Polynomial relations R(y_1..y_k) = 0 that hold for EVERY theta
are found by undetermined coefficients (implicitization by linear algebra): for total degree
D = 1, 2, ... write R = sum_a c_a y^a over all y-monomials of degree <= D, substitute
y_i = f_i(theta), and require every theta-coefficient to vanish. This gives a linear system in
the c_a over Q; its nullspace is the space of relations of degree <= D. For a circuit of the
algebraic matroid the elimination ideal is principal, so at the minimal degree the space is
one-dimensional. The search is bounded (max_degree, max_unknowns): no relation found within
the bound means "not derived", never "independent".
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations_with_replacement
from typing import Optional, Sequence

from arc3.algebra.linalg import nullspace

Mono = tuple  # exponent vector


class Poly:
    __slots__ = ("n", "terms")

    def __init__(self, n: int, terms: Optional[dict] = None):
        self.n = n
        self.terms = {}
        for m, c in (terms or {}).items():
            c = Fraction(c)
            if c != 0:
                self.terms[tuple(m)] = c

    @staticmethod
    def const(n: int, c) -> "Poly":
        return Poly(n, {(0,) * n: c})

    @staticmethod
    def var(n: int, i: int) -> "Poly":
        e = [0] * n
        e[i] = 1
        return Poly(n, {tuple(e): 1})

    def _coerce(self, o) -> "Poly":
        return o if isinstance(o, Poly) else Poly.const(self.n, o)

    def __add__(self, o):
        o = self._coerce(o)
        t = dict(self.terms)
        for m, c in o.terms.items():
            t[m] = t.get(m, 0) + c
        return Poly(self.n, t)

    __radd__ = __add__

    def __neg__(self):
        return Poly(self.n, {m: -c for m, c in self.terms.items()})

    def __sub__(self, o):
        return self + (-self._coerce(o))

    def __rsub__(self, o):
        return self._coerce(o) - self

    def __mul__(self, o):
        o = self._coerce(o)
        t: dict = {}
        for m1, c1 in self.terms.items():
            for m2, c2 in o.terms.items():
                m = tuple(a + b for a, b in zip(m1, m2))
                t[m] = t.get(m, 0) + c1 * c2
        return Poly(self.n, t)

    __rmul__ = __mul__

    def __pow__(self, k: int):
        out = Poly.const(self.n, 1)
        base = self
        while k:
            if k & 1:
                out = out * base
            base = base * base
            k >>= 1
        return out

    def is_zero(self) -> bool:
        return not self.terms

    def degree(self) -> int:
        return max((sum(m) for m in self.terms), default=0)

    def eval(self, point: Sequence) -> Fraction:
        s = Fraction(0)
        for m, c in self.terms.items():
            v = c
            for x, e in zip(point, m):
                if e:
                    v *= Fraction(x) ** e
            s += v
        return s

    def diff(self, i: int) -> "Poly":
        t = {}
        for m, c in self.terms.items():
            if m[i]:
                mm = list(m)
                mm[i] -= 1
                t[tuple(mm)] = c * m[i]
        return Poly(self.n, t)

    def substitute(self, images: Sequence["Poly"]) -> "Poly":
        """Replace variable j by images[j] (all images share one ring)."""
        n2 = images[0].n
        out = Poly(n2)
        for m, c in self.terms.items():
            term = Poly.const(n2, c)
            for j, e in enumerate(m):
                if e:
                    term = term * images[j] ** e
            out = out + term
        return out

    def __repr__(self):
        if not self.terms:
            return "0"
        parts = []
        for m, c in sorted(self.terms.items(), reverse=True):
            mon = "*".join(f"x{i}" + (f"^{e}" if e > 1 else "") for i, e in enumerate(m) if e)
            parts.append(f"{c}" + (f"*{mon}" if mon else ""))
        return " + ".join(parts)


def _y_monomials(k: int, D: int) -> list[tuple]:
    out = []
    for d in range(D + 1):
        for combo in combinations_with_replacement(range(k), d):
            e = [0] * k
            for i in combo:
                e[i] += 1
            out.append(tuple(e))
    return out


def implicit_relations(fs: Sequence[Poly], max_degree: int = 4, max_unknowns: int = 500):
    """Relations among y_i = fs[i](theta) of the smallest total degree D <= max_degree.
    Returns (D, [Poly in len(fs) variables]) or None if none exists within the bounds."""
    k = len(fs)
    if k == 0:
        return None
    powers: dict = {}

    def pw(i, e):
        if (i, e) not in powers:
            powers[(i, e)] = fs[i] ** e
        return powers[(i, e)]

    for D in range(1, max_degree + 1):
        ym = _y_monomials(k, D)
        if len(ym) > max_unknowns:
            return None
        cols = []
        for a in ym:
            p = Poly.const(fs[0].n, 1)
            for i, e in enumerate(a):
                if e:
                    p = p * pw(i, e)
            cols.append(p)
        tm = sorted({m for p in cols for m in p.terms})
        M = [[p.terms.get(m, Fraction(0)) for p in cols] for m in tm]
        ns = nullspace(M) if tm else [[Fraction(int(j == 0)) for j in range(len(ym))]]
        if ns:
            return D, [Poly(k, {ym[j]: v[j] for j in range(len(ym)) if v[j] != 0}) for v in ns]
    return None
