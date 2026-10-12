"""Verification of the B2_TOOLS sandbox helpers.

Each helper is checked three ways:
  * against an independent brute-force oracle on random instances;
  * against its interface contract: every key named in TOOL_DOC (the exact prompt
    text the model sees) is actually returned, so prompt and runtime cannot drift;
  * against the sandbox time budget: adversarial inputs return within time_limit.
"""
import itertools
import random
import sys
import time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle" / "b2"))
sys.path.insert(0, str(ROOT))
import sandbox_tools as T  # noqa: E402
from arc3.algebra.linalg import nullspace, rank  # noqa: E402  (independent implementation)

RNG = random.Random(12345)

CONTRACT = {
    "bfs_plan": {"plan", "exhausted", "nodes"},
    "distinguish": {"sequence", "groups", "errors", "exhausted"},
    "gf2_solve": {"presses", "cost", "exact", "n_solutions"},
    "zm_solve": {"presses", "cost"},
    "fit_boolean": {"terms", "alternatives", "unidentified", "complete"},
    "fit_guarded": {"kind", "a", "b", "cap", "guard_between"},
}


# ------------------------------------------------------------------ interface contract
def test_doc_names_every_returned_key_and_every_tool():
    doc = T.TOOL_DOC
    for name, keys in CONTRACT.items():
        assert f"`{name}(" in doc, name
        for k in keys:
            assert f"'{k}'" in doc, (name, k)
    assert set(CONTRACT) == set(T.B2_TOOL_NAMES)


def test_returned_keys_match_contract():
    r = T.bfs_plan(0, lambda s, a: s + 1 if s < 3 else None, lambda s: s == 3, ["+"])
    assert set(r) == CONTRACT["bfs_plan"] and r["plan"] == ["+", "+", "+"]
    r = T.distinguish([lambda s, a: (s, 0), lambda s, a: (s, 1)], [0, 0], ["x"])
    assert set(r) == CONTRACT["distinguish"] and r["sequence"] == ["x"]
    assert set(T.gf2_solve([[1]], [1])) == CONTRACT["gf2_solve"]
    assert set(T.zm_solve([[1]], [1], 3)) == CONTRACT["zm_solve"]
    assert set(T.fit_boolean([[0], [1]], [0, 1])) == CONTRACT["fit_boolean"]
    assert set(T.fit_guarded([0, 1, 2], [0, 1, 2])) == CONTRACT["fit_guarded"]


# ------------------------------------------------------------------ gf2 / zm vs brute force
def _brute_mod(A, b, m, w):
    n = len(A[0])
    best = None
    for u in itertools.product(range(m), repeat=n):
        if all(sum(a * x for a, x in zip(row, u)) % m == bi % m for row, bi in zip(A, b)):
            c = sum(wi * ui for wi, ui in zip(w, u))
            best = c if best is None else min(best, c)
    return best


def test_gf2_matches_bruteforce():
    for _ in range(400):
        n, m = RNG.randint(1, 8), RNG.randint(1, 7)
        A = [[RNG.randint(0, 1) for _ in range(n)] for _ in range(m)]
        b = [RNG.randint(0, 1) for _ in range(m)]
        w = [RNG.randint(1, 6) for _ in range(n)]
        best = _brute_mod(A, b, 2, w)
        got = T.gf2_solve(A, b, cost=w)
        assert (got is None) == (best is None)
        if got:
            assert got["exact"] and got["cost"] == best
            u = [1 if j in got["presses"] else 0 for j in range(n)]
            assert all(sum(a * x for a, x in zip(r, u)) % 2 == bi for r, bi in zip(A, b))
            assert got["cost"] == sum(w[j] for j in got["presses"])


def test_zm_matches_bruteforce_including_nonprime():
    for m in (2, 3, 4, 6):
        for _ in range(60):
            n, k = RNG.randint(1, 4), RNG.randint(1, 3)
            A = [[RNG.randint(0, m - 1) for _ in range(n)] for _ in range(k)]
            b = [RNG.randint(0, m - 1) for _ in range(k)]
            w = [RNG.randint(1, 4) for _ in range(n)]
            best = _brute_mod(A, b, m, w)
            got = T.zm_solve(A, b, m, cost=w)
            assert (got is None) == (best is None)
            if got:
                assert got["cost"] == best


def test_gf2_respects_time_budget():
    A = [[RNG.randint(0, 1) for _ in range(60)] for _ in range(20)]   # nullity ~40
    b = [RNG.randint(0, 1) for _ in range(20)]
    t = time.time()
    r = T.gf2_solve(A, b, time_limit=2.0)
    assert time.time() - t < 3.0
    assert r is None or r["exact"] is False


# ------------------------------------------------------------------ fit_boolean
def _features(rows, monos):
    return [[int(all(r[i] for i in mono)) for mono in monos] for r in rows]


def test_fit_boolean_minimal_support_and_identifiability_vs_oracle():
    for _ in range(80):
        n = RNG.randint(2, 4)
        monos = [()] + [(i,) for i in range(n)] + list(itertools.combinations(range(n), 2))
        true = {mono: RNG.choice([-2, -1, 1, 2]) for mono in RNG.sample(monos, RNG.randint(1, 3))}
        configs = list(itertools.product((0, 1), repeat=n))
        rows = [list(c) for c in RNG.sample(configs, RNG.randint(2, len(configs)))]
        vals = [sum(c * all(r[i] for i in mono) for mono, c in true.items()) for r in rows]
        got = T.fit_boolean(rows, vals)
        feat = _features(rows, monos)
        k = len(got["terms"])
        for kk in range(0, k):                              # oracle 1: nothing sparser fits exactly
            for supp in itertools.combinations(range(len(monos)), kk):
                if kk == 0:
                    assert not all(v == 0 for v in vals)
                    continue
                sub = [[f[j] for j in supp] for f in feat]
                aug = [r + [v] for r, v in zip(sub, vals)]
                assert rank(aug) > rank(sub), (rows, vals, supp)
        for r, v in zip(rows, vals):                        # the reported rule reproduces the data
            pred = sum(Fraction(c) * all(r[i] for i in mono) for mono, c in got["terms"].items())
            assert pred == v
        ns = nullspace(feat)                                # oracle 2: unidentified <=> in nullspace support
        oracle_unid = {monos[j] for j in range(len(monos)) if any(v[j] != 0 for v in ns)}
        assert set(got["unidentified"]) == oracle_unid
        assert got["complete"] is True


def test_fit_boolean_or_xor_fixture_and_validation():
    two = T.fit_boolean([[1, 0], [0, 1]], [1, 1])
    assert (0, 1) in two["unidentified"]                    # interaction not pinned yet
    amb = T.fit_boolean([[1, 0], [0, 1], [1, 1]], [1, 1, 0])
    assert amb["terms"] == {(): 1, (0, 1): -1}              # without (0,0): sparser rule wins (MDL)
    assert amb["unidentified"]                              # ...and the tool says the data is not enough
    xor = T.fit_boolean([[0, 0], [1, 0], [0, 1], [1, 1]], [0, 1, 1, 0])
    assert xor["terms"] == {(0,): 1, (1,): 1, (0, 1): -2} and xor["unidentified"] == []
    for bad in ([[2, 0], [0, 1]], [[0, 1], [1]]):
        try:
            T.fit_boolean(bad, [1, 1])
            assert False, "non-binary / ragged rows must be rejected"
        except ValueError:
            pass


def test_fit_boolean_respects_time_budget():
    rows = [[RNG.randint(0, 1) for _ in range(12)] for _ in range(30)]
    vals = [RNG.randint(0, 5) for _ in rows]
    t = time.time()
    r = T.fit_boolean(rows, vals, time_limit=2.0)
    assert time.time() - t < 4.0
    assert r is None or r["complete"] is False or r["terms"] is not None


# ------------------------------------------------------------------ fit_guarded
def test_fit_guarded_recovers_walls_and_reports_interval():
    for _ in range(200):
        a = RNG.choice([1, 2, -1, 3])
        b = RNG.randint(-3, 3)
        xs = sorted(RNG.sample(range(-10, 20), RNG.randint(6, 12)))
        wall = RNG.choice(xs[2:-2])
        kind = RNG.choice(["cap_high", "cap_low"])
        cap = a * wall + b
        if kind == "cap_high":
            ys = [a * x + b if x < wall else cap for x in xs]
        else:
            ys = [a * x + b if x > wall else cap for x in xs]
        got = T.fit_guarded(xs, ys)
        assert got is not None and got["kind"] in ("cap_high", "cap_low")
        assert got["cap"] == cap
        lo, hi = got["guard_between"]
        assert lo <= wall <= hi
        for x, y in zip(xs, ys):                            # the reported rule reproduces the data
            on_line = abs(got["a"] * x + got["b"] - y) < 1e-9
            if lo < x < hi:
                assert on_line or y == got["cap"]
            elif (got["kind"] == "cap_high" and x >= hi) or (got["kind"] == "cap_low" and x <= lo):
                assert y == got["cap"]
            else:
                assert on_line


def test_fit_guarded_examples():
    assert T.fit_guarded(range(9), [min(x, 5) for x in range(9)])["guard_between"] == (4, 6)
    low = T.fit_guarded(range(9), [max(x, 3) for x in range(9)])
    assert low["kind"] == "cap_low" and low["guard_between"] == (2, 4)
    assert T.fit_guarded(range(9), [min(max(x, 2), 6) for x in range(9)]) is None   # two guards: out of class
    step = T.fit_guarded([0, 1, 2], [0, 5, 5])               # a step: slope from the 2 points that must be on the line
    assert step["kind"] == "cap_low" and step["a"] == 0 and step["cap"] == 0 and step["guard_between"] == (0, 1)
    # slope never estimated from points that could be capped: [0,5] alone is ambiguous -> affine only
    assert T.fit_guarded([0, 1], [0, 5])["kind"] == "affine"


# ------------------------------------------------------------------ bfs_plan vs brute force
def test_bfs_plan_is_shortest_on_random_graphs():
    for _ in range(150):
        nstates = RNG.randint(3, 12)
        acts = ["a", "b", "c"][: RNG.randint(1, 3)]
        trans = {(s, a): (RNG.randrange(nstates) if RNG.random() > 0.2 else None)
                 for s in range(nstates) for a in acts}
        goal = RNG.randrange(nstates)
        step = lambda s, a: trans[(s, a)]
        best = None
        for L in range(0, nstates + 1):
            for seq in itertools.product(acts, repeat=L):
                s, ok = 0, True
                for a in seq:
                    s = step(s, a)
                    if s is None:
                        ok = False
                        break
                if ok and s == goal:
                    best = L
                    break
            if best is not None:
                break
        r = T.bfs_plan(0, step, lambda s: s == goal, acts)
        if best is None:
            assert r["plan"] is None and r["exhausted"]
        else:
            assert len(r["plan"]) == best
            s = 0
            for a in r["plan"]:
                s = step(s, a)
            assert s == goal


# ------------------------------------------------------------------ distinguish vs brute force
def test_distinguish_is_shortest_and_ignores_crashes():
    for _ in range(150):
        nstates = RNG.randint(2, 6)
        acts = ["x", "y"]

        def mk():
            tr = {(s, a): RNG.randrange(nstates) for s in range(nstates) for a in acts}
            ob = {s: RNG.randrange(2) for s in range(nstates)}
            return lambda s, a: (tr[(s, a)], ob[tr[(s, a)]])
        sims = [mk(), mk()]
        best = None
        for L in range(1, 7):
            for seq in itertools.product(acts, repeat=L):
                s0 = s1 = 0
                for a in seq:
                    s0, o0 = sims[0](s0, a)
                    s1, o1 = sims[1](s1, a)
                    if o0 != o1:
                        best = L
                        break
                if best:
                    break
            if best:
                break
        r = T.distinguish(sims, [0, 0], acts, max_depth=6)
        if best is None:
            assert r["sequence"] is None
        else:
            assert len(r["sequence"]) == best

    def crash(s, a):
        raise KeyError("unmodelled")
    r = T.distinguish([lambda s, a: (s, "same"), crash], [0, 0], ["a"], max_depth=3)
    assert r["sequence"] is None and r["errors"][1]["count"] >= 1   # a crash is not a difference
    r = T.distinguish([lambda s, a: (s, "same")] * 2, [0, 0], ["a"], max_depth=3)
    assert r["sequence"] is None and r["exhausted"]


def test_search_tools_respect_time_budget():
    t = time.time()
    r = T.bfs_plan(0, lambda s, a: s + 1, lambda s: False, ["a"], max_nodes=10 ** 9, time_limit=1.0)
    assert time.time() - t < 2.0 and r["plan"] is None and not r["exhausted"]
    t = time.time()
    r = T.distinguish([lambda s, a: (s + 1, 0)] * 2, [0, 0], ["a", "b"], max_depth=60,
                      max_nodes=10 ** 9, time_limit=1.0)
    assert time.time() - t < 2.0 and r["sequence"] is None and not r["exhausted"]
