"""Stage-1 fixtures for the DIAG candidate (docs/prereg_diag.md). C must pass all; B is recorded."""
import json
from fractions import Fraction
from pathlib import Path

from arc3.diag.checkers import AlgebraicChecker, Fitter, OrdinaryChecker
from arc3.diag.ledger import Component, Observation
from arc3.diag.poly import Poly
from arc3.diag.structure import StructureCache

RECORD = Path(__file__).resolve().parents[1] / "runs" / "diag" / "fixtures_b_vs_c.json"
_record = {}


def _log(name, b, c):
    _record[name] = {"B": {"verdict": b.verdict, "diagnoses": b.diagnoses, "unverifiable": b.unverifiable},
                     "C": {"verdict": c.verdict, "diagnoses": c.diagnoses, "unverifiable": c.unverifiable,
                           "suspects": c.suspects, "ambiguity": c.ambiguity, "suggestions": c.suggestions,
                           "total_s": c.costs.get("total_s")}}
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(_record, indent=1, default=str))


def _both(comp, obs, name, k_max=3):
    b = OrdinaryChecker(k_max=k_max, poly_draws=80).check(comp, obs)     # reduced budget: recorded only
    c = AlgebraicChecker(k_max=k_max).check(comp, obs)
    _log(name, b, c)
    assert not c.fallback, c.costs
    return b, c


# ---------- bilinear (nonlinear in parameters): y_ij = a_i * b_j, gauge fixed by b_0 = 1
A, B = [2, 3, 5], [1, 4, 7, 2]
NP = 3 + 3                                     # a0..a2, b1..b3


def _bilinear():
    def poly(ctx):
        i, j = ctx["i"], ctx["j"]
        a = Poly.var(NP, i)
        return a if j == 0 else a * Poly.var(NP, 3 + j - 1)
    return Component(name="bilinear", version=1, kind="polynomial",
                     params=["a0", "a1", "a2", "b1", "b2", "b3"],
                     explains=lambda o: o.quantity == "y", applies=lambda o: (True, ""),
                     poly=poly, design_key=lambda ctx: (ctx["i"], ctx["j"]),
                     applicability_text="all cells")


def _grid(corrupt=()):
    obs = []
    for i in range(3):
        for j in range(4):
            v = A[i] * B[j] + (7 if (i, j) in corrupt else 0)
            obs.append(Observation(f"y{i}{j}", "y", v, {"i": i, "j": j}))
    return obs


def test_f1_correct_nonlinear_model():
    b, c = _both(_bilinear(), _grid(), "F1")
    assert c.verdict == "consistent" and not c.diagnoses
    assert c.relations and all(r["satisfied"] for r in c.relations if r["satisfied"] is not None)
    assert any(r["relation"] and "^" not in r["relation"] for r in c.relations)   # 2x2 minors found
    assert c.identifiability["jacobian_rank"] == 6
    assert "not a uniqueness claim" in c.identifiability["claim"]


def test_f2_one_incorrect_observation():
    comp = _bilinear()
    b, c = _both(comp, _grid(corrupt={(1, 2)}), "F2")
    assert c.verdict == "contradiction-localized" and c.diagnoses == [["y12"]]
    # repair: drop the diagnosed observation, refit, predict the true value of y12 and all others
    obs = [o for o in _grid(corrupt={(1, 2)}) if o.id != "y12"]
    F = Fitter(comp, obs, seed=1)
    th = F.fit(list(range(len(obs))))
    assert th is not None
    from arc3.diag.checkers import _compile, _peval
    pred = _peval(_compile(comp.poly({"i": 1, "j": 2})), th)[0]
    assert abs(pred - A[1] * B[2]) < 1e-6


def test_f3_multiple_incorrect_observations():
    for faults in ({(0, 1), (2, 3)}, {(0, 1), (1, 3), (2, 2)}):
        b, c = _both(_bilinear(), _grid(corrupt=faults), f"F3_{len(faults)}")
        truth = sorted(f"y{i}{j}" for i, j in faults)
        assert c.verdict in ("contradiction-localized", "contradiction-ambiguous")
        assert truth in [sorted(d) for d in c.diagnoses]
        assert not (c.verdict == "contradiction-localized" and len(c.diagnoses[0]) == 1)
        if c.verdict == "contradiction-ambiguous":
            assert any(s["type"] == "reobserve" for s in c.suggestions)


def test_f4_full_rank_with_unresolved_sign():
    th = lambda: Poly.var(1, 0)

    def poly(ctx):
        return th() ** 2 * ctx["u"] if ctx["type"] == "sq" else th() * ctx["w"]
    comp = Component(name="signed", version=1, kind="polynomial", params=["theta"],
                     explains=lambda o: True, applies=lambda o: (True, ""), poly=poly,
                     design_key=lambda ctx: tuple(sorted(ctx.items())),
                     symmetries=[("sign", lambda t: [-t[0]])], catalog=[{"type": "lin", "w": 1}])
    obs = [Observation(f"q{u}", "y", 9 * u, {"type": "sq", "u": u}) for u in (1, 2, 3)]   # theta = +-3
    b, c = _both(comp, obs, "F4")
    assert c.verdict == "consistent"
    assert c.identifiability["jacobian_rank"] == 1 == c.identifiability["n_params"]
    assert "not a uniqueness claim" in c.identifiability["claim"]
    assert c.ambiguity.get("discrete_unresolved") == ["sign"]
    sug = [s for s in c.suggestions if s["type"] == "distinguishing-context"]
    assert sug and sug[0]["context"] == {"type": "lin", "w": 1}
    # once the distinguishing observation exists, the sign is resolved
    c2 = AlgebraicChecker().check(comp, obs + [Observation("z", "y", -3, {"type": "lin", "w": 1})])
    assert "discrete_unresolved" not in c2.ambiguity and c2.verdict == "consistent"


def _mover(name="move", version=1, mode_aware=True):
    return Component(name=name, version=version, kind="linear", field="q", params=["p0", "v"],
                     explains=lambda o: o.quantity == "pos",
                     applies=(lambda o: (o.context["mode"] == 0, "mode changed")) if mode_aware
                     else (lambda o: (True, "")),
                     row=lambda ctx: [1, ctx["t"]], design_key=lambda ctx: ctx["t"],
                     context_signature=lambda ctxs: ("mode", 0, ctxs[0]["level"]),
                     applicability_text="mode 0 only" if mode_aware else "all")


def _modes(level=1):
    obs = [Observation(f"p{t}", "pos", 3 + 2 * t, {"t": t, "mode": 0, "level": level}) for t in range(6)]
    obs += [Observation(f"p{t}", "pos", 13 + 5 * (t - 5), {"t": t, "mode": 1, "level": level}) for t in range(6, 11)]
    return obs


def test_f5_changed_context_invalidates_cache():
    cache = StructureCache()
    C = AlgebraicChecker(cache=cache)
    comp = _mover()
    e1 = C.check(comp, _modes())
    assert e1.verdict == "consistent" and len(e1.excluded) == 5
    assert all(r == "mode changed" for _, r in e1.excluded)
    assert cache.stats["misses"] == 1
    C.check(comp, _modes())
    assert cache.stats["hits"] == 1                                  # same version + context: reused
    C.check(comp, _modes(level=2))
    assert cache.stats["misses"] == 2                                # new context: not reused
    assert cache.invalidate(("version", "move")) == 2                # hypothesis revised
    e3 = C.check(_mover(version=2), _modes())
    assert e3.verdict == "consistent" and cache.stats["invalidated"] == 2
    naive = AlgebraicChecker().check(_mover(mode_aware=False), _modes())
    assert naive.verdict == "hypothesis-inadequate"                  # mixing modes is a model error
    assert naive.ambiguity["regimes"]["first_inconsistent"] == "p6"
    b = OrdinaryChecker().check(comp, _modes())
    _log("F5", b, e1)


def test_f6_incomplete_hypothesis_family():
    comp = Component(name="line", version=1, kind="linear", field="q", params=["c0", "c1"],
                     explains=lambda o: True, applies=lambda o: (True, ""),
                     row=lambda ctx: [1, ctx["x"]], design_key=lambda ctx: ctx["x"])
    obs = [Observation(f"x{x}", "y", 1 + 2 * x + 3 * x * x, {"x": x}) for x in range(8)]
    b, c = _both(comp, obs, "F6")
    assert c.verdict == "hypothesis-inadequate"
    assert c.ambiguity["regimes"]["later_self_consistent"] is False
    assert not c.diagnoses


def _tracked(tagged):
    comp = Component(name="trackA", version=1, kind="linear", field="q", params=["p0", "v"],
                     explains=lambda o: o.quantity == "pos_A",
                     applies=lambda o: ((o.context["identity"] == "certain"), "identity uncertain"),
                     row=lambda ctx: [1, ctx["t"]], design_key=lambda ctx: ctx["t"])
    obs = []
    for t in range(12):
        swapped = t >= 6
        v = (43 - 3 * t) if swapped else (10 + 2 * t)               # tracker follows B after t=6 (paths do not cross at t=6)
        ident = "uncertain" if (swapped and tagged) else "certain"
        obs.append(Observation(f"a{t}", "pos_A", v, {"t": t, "identity": ident}))
    return comp, obs


def test_f7_misleading_tracking():
    comp, obs = _tracked(tagged=True)
    b, c = _both(comp, obs, "F7_tagged")
    assert c.verdict == "consistent"
    assert c.ambiguity["identity_uncertain"] == [f"a{t}" for t in range(6, 12)]
    comp, obs = _tracked(tagged=False)
    b, c = _both(comp, obs, "F7_untagged")
    assert c.verdict == "hypothesis-inadequate"
    assert c.suspects == [f"a{t}" for t in range(6, 12)]
    assert c.ambiguity["regimes"]["later_self_consistent"] is True


def test_no_behavioural_authority_and_fallback():
    # the checkers return ledger entries only; an unsupported component falls back to B
    comp = Component(name="opaque", version=1, kind="unsupported", params=["?"],
                     explains=lambda o: True, applies=lambda o: (True, ""))
    e = AlgebraicChecker().check(comp, [Observation("o", "y", 1, {})])
    assert e.fallback and e.verdict == "unsupported"
    bad = Component(name="broken", version=1, kind="linear", field="q", params=["p"],
                    explains=lambda o: True, applies=lambda o: (True, ""),
                    row=lambda ctx: 1 / 0)                            # raises inside C
    e2 = AlgebraicChecker().check(bad, [Observation("o", "y", 1, {})])
    assert e2.fallback and "ZeroDivisionError" in e2.costs["fallback_reason"]
    for k in ("ban", "stop", "end_game", "action"):
        assert not any(k in s.get("type", "") for s in e.suggestions + e2.suggestions)
