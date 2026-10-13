"""Correctness boundaries of the contextual action algebra (arc3/hx/algebra.py).

Synthetic systems with known dynamics. These tests pin what the module may and may not predict;
they are not evidence of game performance.
"""
import itertools
import random

from arc3.hx.algebra import (ABSORB, COMMUTE, CYCLE, IDEM, INVERSE, INVOL, NOOP, OVERRIDE,
                             Algebra)


def explore(alg, step, start, actions, n, seed=0):
    """Random walk observing every transition; returns visited states."""
    rng = random.Random(seed)
    s = start
    seen = {s}
    for _ in range(n):
        a = rng.choice(actions)
        u = step(s, a)
        alg.observe(s, a, u)
        s = u
        seen.add(s)
    return seen


def check_predictions(alg, step, states, actions):
    """Every prediction the algebra makes must be correct for these systems."""
    made = wrong = 0
    for s in states:
        for a in actions:
            p = alg.predict(s, a)
            if p is not None:
                made += 1
                wrong += int(p[0] != step(s, a))
    return made, wrong


def test_xor_toggles_commute_and_involution_predict_unseen():
    # 4 lights, click i toggles light i and i+1 (mod 4). All clicks commute, each is an involution.
    acts = [(6, i, 0) for i in range(4)]

    def step(s, a):
        i = a[1]
        s = list(s)
        s[i] ^= 1
        s[(i + 1) % 4] ^= 1
        return tuple(s)

    alg = Algebra(ctx=lambda k, a: ((6, "button"), a[1]))
    start = (0, 0, 0, 0)
    seen = explore(alg, step, start, acts, 40, seed=1)
    made, wrong = check_predictions(alg, step, seen, acts)
    assert wrong == 0
    assert made > 0
    rel = {(r["rel"], r["level"]) for r in alg.relations() if r["counter"] == 0}
    assert (INVOL, 0) in rel and (COMMUTE, 0) in rel


def test_wall_noop_does_not_generalise_and_inverse_has_precondition():
    # 1-D corridor 0..5, LEFT/RIGHT; walls at the ends make the move a no-op there.
    L, R = (3,), (4,)

    def step(s, a):
        return max(0, s - 1) if a == L else min(5, s + 1)

    alg = Algebra()
    seen = explore(alg, step, 2, [L, R], 200, seed=2)
    # NOOP of LEFT is refuted at L0 (it moves in the interior), so no global no-op prediction
    assert alg.holds(NOOP, 3, (L,)) is None
    # LEFT then RIGHT = 1 holds whenever LEFT changed the state (precondition), so it predicts
    made, wrong = check_predictions(alg, step, seen, [L, R])
    assert wrong == 0
    assert alg.holds(INVERSE, 3, (L, R)) is not None


def test_last_click_wins_overrides_and_refutes_commutation():
    # state = last clicked button; click b sets state to b
    acts = [(6, i, 0) for i in range(4)]

    def step(s, a):
        return a[1]

    alg = Algebra(ctx=lambda k, a: ((6, "button"), a[1]))
    seen = explore(alg, step, -1, acts, 60, seed=3)
    rels = {(r["rel"], r["level"]): r for r in alg.relations(limit=200)}
    assert rels[(COMMUTE, 0)]["counter"] > 0          # order matters
    assert rels[(OVERRIDE, 0)]["counter"] == 0         # ab = b
    made, wrong = check_predictions(alg, step, seen, acts)
    assert wrong == 0


def test_cycle_period_five():
    # one button advances a counter mod 5; another resets to 0. Orbit index 0, period 5.
    A, Z = (1,), (2,)

    def step(s, a):
        return (s + 1) % 5 if a == A else 0

    alg = Algebra(max_cycle=8)
    # observe two full orbits from different states (support 5 states for a^5 = 1)
    s = 0
    for _ in range(12):
        u = step(s, A)
        alg.observe(s, A, u)
        s = u
    assert alg.holds("%s|5" % CYCLE, 0, (A,)) is not None
    assert alg.holds("%s|3" % CYCLE, 0, (A,)) is None       # a^3 != 1 (counterexamples)
    idx, per = alg.note_cycles(2, A)
    assert (idx, per) == (0, 5)


def test_context_refinement_by_mode():
    # mode 0: action X moves right; mode 1: X is a no-op. Inventory token = mode.
    X, M = (1,), (2,)

    def step(s, a):
        pos, mode = s
        if a == M:
            return (pos, 1 - mode)
        return (min(6, pos + 1), mode) if mode == 0 else s

    alg = Algebra(inventory=lambda k: k[1])
    rng = random.Random(4)
    s = (0, 0)
    for _ in range(300):
        a = rng.choice([X, M, X])
        u = step(s, a)
        alg.observe(s, a, u)
        s = u
        if rng.random() < 0.05:
            s = (0, rng.choice([0, 1]))
    # coarse NOOP(X) is refuted, but at L1 (mode 1) it holds
    assert alg.holds(NOOP, (3, 0), (X,)) is None
    h = alg.holds(NOOP, (3, 1), (X,))
    assert h is not None and h[0] == 1


def test_hidden_state_conflict_is_reported_not_modelled():
    # visible state never changes; a hidden flag set by A makes B change nothing visible either,
    # but the outcome of C depends on the flag -> same visible state + action, different result.
    A, C = (1,), (3,)
    alg = Algebra()
    alg.observe("v", C, "v")
    alg.observe("v", A, "v")
    alg.observe("v", C, "w")          # C after A behaves differently
    assert alg.conflicts and alg.conflicts[0][:2] == ("v", C)


def test_absorb_relation():
    # x toggles a flag, y reads... construct aba = b: state is a pair (f, g); a flips f, b sets g := 1 and f := 0.
    A, B = (1,), (2,)

    def step(s, a):
        f, g = s
        return (1 - f, g) if a == A else (0, 1)

    alg = Algebra()
    states = list(itertools.product([0, 1], [0, 1]))
    for s in states:
        for a in (A, B):
            alg.observe(s, a, step(s, a))
    rels = {(r["rel"], r["level"], r["word"]): r for r in alg.relations(limit=200)}
    # a b a = b?  from (0,0): a->(1,0) b->(0,1) a->(1,1) vs b->(0,1): fails; relation must be refuted
    key = (ABSORB, 0, ((1,), (2,)))
    assert key in rels and rels[key]["counter"] > 0
    # b a b = a?  (word (b,a)): from (0,0): b->(0,1) a->(1,1) b->(0,1) vs a->(1,0): fails as well
    # idempotence of b holds everywhere b is effective
    assert alg.holds(IDEM, (0, 0), (B,)) is not None
