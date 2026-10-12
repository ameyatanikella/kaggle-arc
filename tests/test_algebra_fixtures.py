"""Mathematical correctness fixtures (math_contract.md). These check correctness
boundaries of the tools; they are NOT evidence of ARC performance."""
import itertools
import math
import random
from fractions import Fraction

import pytest

from arc3.algebra.boolean import (additivity_check, best_interaction_probe, feature_row,
                                  interaction_estimable, mobius_coefficients, monomials, sparse_fit)
from arc3.algebra.design import (Candidate, Hypothesis, greedy_min_cost_basis, minimax_regret,
                                 pair_separation, partition_entropy, select_combined,
                                 select_rank_only, select_separation_only, should_act)
from arc3.algebra.distinguish import (commutes_on, fit_guarded_affine, markov_conflicts,
                                      shortest_distinguishing_sequence)
from arc3.algebra.linalg import IncrementalRank, left_dependencies, rank
from arc3.algebra.modular import gf2_min_cost, gf2_solve_all, zm_solve_min

MONOS2 = [(0,), (1,), (0, 1)]  # d = t1*b1 + t2*b2 + t12*b1*b2


# 1. OR vs XOR and the interaction-design rank test -------------------------
def test_or_xor_rank_and_probe_choice():
    rows = [feature_row(b, MONOS2) for b in [(1, 0), (0, 1)]]
    assert rank(rows) == 2
    assert rank(rows + [feature_row((1, 1), MONOS2)]) == 3
    OR = lambda b: b[0] | b[1]
    XOR = lambda b: b[0] ^ b[1]
    assert all(OR(b) == XOR(b) for b in [(1, 0), (0, 1)])
    assert OR((1, 1)) != XOR((1, 1))
    info = interaction_estimable([(1, 0), (0, 1)], MONOS2)
    assert info["identifiable"][(0,)] and info["identifiable"][(1,)]
    assert not info["identifiable"][(0, 1)]
    # designer picks the combined configuration, accounting for travel/setup cost
    pick = best_interaction_probe([(1, 0), (0, 1)], [(1, 0), (0, 1), (1, 1)], MONOS2,
                                  cost={(1, 0): 1, (0, 1): 1, (1, 1): 7})
    assert pick == ((1, 1), 7)
    # if (1,1) is unreachable the caller omits it -> no rank-raising probe exists
    assert best_interaction_probe([(1, 0), (0, 1)], [(1, 0), (0, 1)], MONOS2) is None
    # exact identities on the cube
    for x, y in itertools.product((0, 1), repeat=2):
        assert (x + y - x * y) == (x | y) and (x + y - 2 * x * y) == (x ^ y)
    # sparse fit: after observing all three, OR and XOR get distinct coefficients
    obs_or = [((1, 0), 1), ((0, 1), 1), ((1, 1), 1)]
    obs_xor = [((1, 0), 1), ((0, 1), 1), ((1, 1), 0)]
    assert sparse_fit(obs_or, MONOS2)[0]["theta"][(0, 1)] == -1
    assert sparse_fit(obs_xor, MONOS2)[0]["theta"][(0, 1)] == -2
    # with only two observations, both models survive (unidentified, not "OR by default")
    two = [((1, 0), 1), ((0, 1), 1)]
    fits = sparse_fit(two, MONOS2, max_support=3)
    assert all((0, 1) not in f["theta"] for f in fits)  # MDL picks additive, but...
    assert not interaction_estimable([(1, 0), (0, 1)], MONOS2)["identifiable"][(0, 1)]


def test_mobius_exact_and_additivity_circuit():
    n = 3
    f = lambda b: (b[0] ^ b[1]) + 2 * b[2]
    table = {b: f(b) for b in itertools.product((0, 1), repeat=n)}
    coef = mobius_coefficients(table, n)
    assert coef == {(0,): 1, (1,): 1, (0, 1): -2, (2,): 2}
    for b in table:
        assert sum(c * all(b[i] for i in S) for S, c in coef.items()) == table[b]
    assert additivity_check(0, 1, 1, 2) == 0          # additive
    assert additivity_check(0, 1, 1, 0) == -2         # XOR: violates additivity
    # the circuit as a left dependency of the design rows
    rows = [feature_row(b, [(), (0,), (1,)]) for b in [(0, 0), (1, 0), (0, 1), (1, 1)]]
    deps = left_dependencies(rows)
    assert len(deps) == 1
    w = deps[0]
    assert [x / w[0] for x in w] == [1, -1, -1, 1]     # d00 - d10 - d01 + d11 = 0


# 2. Full rank with a sign ambiguity; useful zero-rank-gain measurement -------
def test_sign_ambiguity_rank_only_fails_combined_succeeds():
    y0 = 4
    branches = [t for t in (2, -2) if t * t == y0]
    # J row of y0 at either branch is nonzero -> rank 1 = number of parameters
    chosen_rows = [[Fraction(2 * branches[0])]]
    hyps = [Hypothesis(f"theta={t}", 0.5,
                       predict=(lambda e, t=t: (t * t) if e == "y0" else (t + 1) ** 2),
                       best_action=("right" if t > 0 else "left")) for t in branches]
    cands = [Candidate("repeat_y0", cost=1, experiment="y0", jac_row=[Fraction(4)]),
             Candidate("measure_y1", cost=3, experiment="y1", jac_row=[Fraction(6)])]
    # rank-only selector: no candidate raises rank -> it stops (and would act on a wrong branch half the time)
    assert select_rank_only(cands, chosen_rows) is None
    # combined selector picks y1 because the branches recommend different actions
    pick = select_combined(cands, hyps, chosen_rows)
    assert pick is not None and pick.name == "measure_y1"
    # and y1 actually resolves the branch: theta = (y1 - y0 - 1)/2
    for t in branches:
        assert Fraction((t + 1) ** 2 - y0 - 1, 2) == t
    # if both branches recommended the same action, the combined selector would not pay
    same = [Hypothesis(h.name, h.weight, h.predict, best_action="right") for h in hyps]
    assert select_combined(cands, same, chosen_rows) is None


# 3. Cheap redundant probe vs costly reachable informative probe --------------
def test_cheap_redundant_vs_costly_informative():
    hyps = [Hypothesis("A", 0.5, predict=lambda e: 0 if e == "cheap" else 1, best_action="go_left"),
            Hypothesis("B", 0.5, predict=lambda e: 0 if e == "cheap" else 2, best_action="go_right")]
    cands = [Candidate("cheap", 1, "cheap"), Candidate("costly", 5, "costly")]
    assert partition_entropy(hyps, "cheap") == 0.0
    assert partition_entropy(hyps, "costly") == 1.0
    assert select_separation_only(cands, hyps).name == "costly"
    assert select_combined(cands, hyps).name == "costly"
    cheapest = min(cands, key=lambda c: c.cost)
    assert cheapest.name == "cheap"               # a cost-only rule would waste an action


# 4. Apparently informative experiment that cannot be reached -----------------
def test_unreachable_experiment_is_never_selected():
    hyps = [Hypothesis("A", .5, lambda e: 0 if e != "far" else 1, "x"),
            Hypothesis("B", .5, lambda e: 0 if e != "far" else 2, "y")]
    cands = [Candidate("far", math.inf, "far"), Candidate("near", 2, "near")]
    assert select_combined(cands, hyps) is None   # 'near' separates nothing; 'far' unreachable
    assert select_separation_only(cands, hyps) is None
    rows = [[1, 0], [0, 1]]
    assert greedy_min_cost_basis(rows, [math.inf, 1]) == [1]


# 5. Identical visible states with different hidden timers --------------------
def test_hidden_timer_detected_and_explained():
    # visible state 'S', action 'a': first time -> 'S', second time -> 'X' (timer)
    trans = [("S", "a", "S"), ("S", "a", "X")]
    assert markov_conflicts(trans)
    def model(state, a):  # state = (visible, timer)
        vis, t = state
        nt = t + 1
        return ((vis if nt < 2 else "X"), nt), (vis if nt < 2 else "X")
    s = ("S", 0)
    obs = []
    for _ in range(2):
        s, o = model(s, "a")
        obs.append(o)
    assert obs == ["S", "X"]


# 6. Collision/threshold guard that a smooth fit misses -----------------------
def test_threshold_guard_vs_smooth_fit():
    xs = list(range(0, 9))
    ys = [min(x, 5) for x in xs]        # wall at 5
    m = fit_guarded_affine(xs, ys)
    # The guard location is itself only identified up to {5, 6}: both splits are
    # exact on this data (x=5 lies on both pieces). The fitter returns the first.
    assert m["kind"] == "guarded" and m["threshold"] in (5, 6) and m["sat"] == 5
    # a single polynomial of low degree does not fit exactly (least squares residual > 0)
    import numpy as np
    for deg in (1, 2, 3):
        coef = np.polyfit(xs, ys, deg)
        assert np.max(np.abs(np.polyval(coef, xs) - ys)) > 1e-6


# 7. Equivalent latent descriptions yielding the same policy -------------------
def test_value_equivalent_models_need_no_experiment():
    # Two models disagree only about an irrelevant region; same optimal plan.
    plans = ["p1", "p2"]
    cost = {("p1", "M1"): 10, ("p1", "M2"): 10, ("p2", "M1"): 14, ("p2", "M2"): 14}
    act, pi, reg = should_act(plans, ["M1", "M2"], lambda p, m: cost[(p, m)], regret_tolerance=0)
    assert act and pi == "p1" and reg == 0
    # when the models recommend different plans, regret is positive -> keep testing
    cost2 = {("p1", "M1"): 10, ("p1", "M2"): math.inf, ("p2", "M1"): 30, ("p2", "M2"): 12}
    act2, pi2, reg2 = should_act(plans, ["M1", "M2"], lambda p, m: cost2[(p, m)], regret_tolerance=5,
                                 fail_cost=1000)
    assert not act2 and pi2 == "p2" and reg2 == 20


# 8. Multiple GF(2) solutions with different action costs ---------------------
def test_gf2_multiple_solutions_min_cost():
    # 3 buttons, 2 lights: b1 toggles L1, b2 toggles L2, b3 toggles both. target: both on.
    A = [[1, 0, 1],
         [0, 1, 1]]
    b = [1, 1]
    x0, basis = gf2_solve_all(A, b)
    assert len(basis) == 1                       # two solutions: {b1,b2} and {b3}
    unit = gf2_min_cost(A, b)
    assert unit["solution"] == [0, 0, 1] and unit["cost"] == 1 and unit["exact"]
    # if b3 is far away (travel cost 5), the two-press solution is cheaper
    far = gf2_min_cost(A, b, cost=[1, 1, 5])
    assert far["solution"] == [1, 1, 0] and far["cost"] == 2
    assert gf2_solve_all([[1, 1]], [1]) is not None
    assert gf2_solve_all([[1, 1], [1, 1]], [1, 0]) is None   # infeasible


def test_zm_nonprime_modulus():
    # 4-state dial: press advances by 2 (mod 4). From 0, reach 1? impossible; reach 2: one press.
    assert zm_solve_min([[2]], [1], 4) is None
    assert zm_solve_min([[2]], [2], 4)["solution"] == [1]
    # treating Z_4 as a field (inverting 2) would be invalid: 2 has no inverse mod 4
    assert math.gcd(2, 4) != 1


# 9. Locally commuting pair fails to commute in another mode --------------------
def test_commutation_is_local():
    # state = (x, mode); f: move right by 1; g: in mode 0 no-op, in mode 1 doubles x
    f = lambda s: (s[0] + 1, s[1])
    g = lambda s: (s[0] if s[1] == 0 else 2 * s[0], s[1])
    ok0, _ = commutes_on([(x, 0) for x in range(5)], f, g)
    ok1, bad = commutes_on([(x, 1) for x in range(5)], f, g)
    assert ok0 and not ok1 and bad


# 10. Shortest distinguishing sequence via product BFS ------------------------
def test_shortest_distinguishing_sequence():
    # 1-D corridor of length 5; model A: wall at 3; model B: wall at 4. Start at 0.
    def make(wall):
        def step(x, a):
            nx = x + (1 if a == "R" else -1)
            nx = max(0, min(nx, wall))
            return nx, nx
        return step
    res = shortest_distinguishing_sequence([make(3), make(4)], [0, 0], ["L", "R"], max_depth=10)
    assert res["sequence"] == ["R", "R", "R", "R"]
    same = shortest_distinguishing_sequence([make(3), make(3)], [0, 0], ["L", "R"], max_depth=10)
    assert same["sequence"] is None and same["exhausted"]     # proven: no separation in model space
    bounded = shortest_distinguishing_sequence([make(3), make(4)], [0, 0], ["L", "R"], max_depth=2)
    assert bounded["sequence"] is None and not bounded["exhausted"]   # bound hit: proves nothing


# 11. Rado-Edmonds greedy == brute-force min-cost basis (linear matroid) --------
def test_greedy_basis_matches_bruteforce():
    rng = random.Random(0)
    for _ in range(60):
        n, d = rng.randint(3, 7), rng.randint(2, 4)
        rows = [[rng.randint(-2, 2) for _ in range(d)] for _ in range(n)]
        costs = [rng.randint(1, 9) for _ in range(n)]
        r = rank(rows)
        g = greedy_min_cost_basis(rows, costs)
        assert rank([rows[i] for i in g]) == r
        best = min((sum(costs[i] for i in S) for S in itertools.combinations(range(n), r)
                    if rank([rows[i] for i in S]) == r), default=0)
        assert sum(costs[i] for i in g) == best


def test_partition_entropy_is_expected_info_gain():
    hyps = [Hypothesis(str(i), w, predict=lambda e, i=i: i % 2) for i, w in enumerate([.25, .25, .25, .25])]
    assert abs(partition_entropy(hyps, None) - 1.0) < 1e-12
    hyps3 = [Hypothesis(str(i), 1 / 3, predict=lambda e, i=i: i) for i in range(3)]
    assert abs(partition_entropy(hyps3, None) - math.log2(3)) < 1e-12
