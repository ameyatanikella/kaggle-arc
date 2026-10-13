"""Verification of the restricted exact noncommutative signature (arc3/algebra/ncsig.py)."""
import importlib.util
import random
from fractions import Fraction as F
from pathlib import Path

import pytest

from arc3.algebra import ncsig as N

A, B, C = N.var(1), N.var(2), N.var(3)


def word_sig(word, d):
    v = [F(1)] + [F(0)] * (d - 1)
    for i in reversed(word):
        v = N.t_apply(i, v)
    return v


def test_recurrence_equals_direct_matrix():
    for i in (1, 2, 3):
        v = [F(j - 2, 3) for j in range(9)]
        direct = [sum(F((-1) ** (r - q - 1), r * i ** (r - q)) * v[q] for q in range(r)) if r else F(0)
                  for r in range(9)]
        assert direct == N.t_apply(i, v)


def test_distributive_identity_is_exactly_zero():
    assert N.is_zero(N.sub(N.mul(N.add(A, B), C), N.add(N.mul(A, C), N.mul(B, C))), 3)
    assert N.equal(N.mul(N.add(A, B), C), N.add(N.mul(A, C), N.mul(B, C)), 3)   # different shapes


def test_commutator_nonzero_though_scalar_substitution_misses_it():
    comm = N.sub(N.mul(A, B), N.mul(B, A))
    assert not N.is_zero(comm, 2)
    rng = random.Random(0)
    for _ in range(20):                                   # commuting scalars always give 0
        a, b = F(rng.randint(-50, 50)), F(rng.randint(-50, 50))
        assert a * b - b * a == 0


def test_truncation_false_zero():
    assert not any(word_sig((1,) * 8, 8))                 # too short: false zero
    assert any(word_sig((1,) * 8, 9))                     # adequate length detects A^8
    with pytest.raises(ValueError):
        g, m = N.to_abp(N.mul(A, A))
        N.signature(g, m, 1, N.bound(m, 1) - 1)          # the size promise is enforced


def test_environment_identity_is_not_free_identity():
    expr = N.sub(N.mul(A, A), N.const(1))                 # toggle semantics: A^2 = I
    assert not N.is_zero(expr, 1)                          # nonzero in the free algebra
    import numpy as np
    T = np.array([[0, 1], [1, 0]])
    assert (T @ T == np.eye(2)).all()                      # ... but zero in this environment


def test_random_formulas_agree_with_independent_expansion():
    rng = random.Random(721116)
    for trial in range(120):
        e1 = N.random_formula(rng, 3, rng.randint(2, 9))
        e2 = e1 if trial % 3 == 0 else N.random_formula(rng, 3, rng.randint(2, 9))
        diff = N.sub(e1, e2) if trial % 3 else N.add(N.sub(e1, e2), N.sub(e2, e1))
        assert N.is_zero(diff, 3) == (N.expand(diff) == {})
        g, m = N.to_abp(diff)
        assert N.abp_expand(g, m) == N.expand(diff)


def test_random_matrix_baseline_agrees():
    rng = random.Random(5)
    for _ in range(40):
        e = N.sub(N.random_formula(rng, 2, 6), N.random_formula(rng, 2, 6))
        assert N.random_matrix_test(e, 2, max(1, N.degree_of(e))) == (N.expand(e) == {})


PROBE = Path("/Users/ameya/Documents/Codex/2026-10-11/do/outputs/math_transfer_probes.py")


@pytest.mark.skipif(not PROBE.exists(), reason="external probe file not present")
def test_cross_check_with_report_probe_implementation():
    spec = importlib.util.spec_from_file_location("mtp", PROBE)
    mtp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mtp)                          # defines functions only; main() not run
    rng = random.Random(9)
    for _ in range(30):
        e = N.random_formula(rng, 3, rng.randint(2, 7))
        g, m = N.to_abp(e)
        d = N.bound(m, 3)
        assert [F(x) for x in mtp.abp_jet(g, m, 3, d)] == N.signature(g, m, 3, d)
