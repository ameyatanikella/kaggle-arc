"""Two checkers over the same components and observations.

B  OrdinaryChecker: robust prediction/consistency checking without dependency structure.
   Exact RANSAC consensus over minimal subsets (exhaustive when small, else seeded draws);
   each fit takes a RANDOM particular solution so an undetermined prediction shows up as
   disagreement between fits; ties between different outlier sets are reported (ambiguous),
   never broken silently. Polynomial components are fitted numerically (Levenberg-Marquardt,
   several random starts) and compared with a relative tolerance.
C  AlgebraicChecker: B's fitter plus the parameter-to-observation structure: rank, circuits,
   exact relations (left-null vectors / implicit polynomials), syndrome-matching multi-fault
   diagnosis, coloops (= unverifiable observations), discrete-symmetry ambiguity, and targeted
   suggestions. On an exception, unsupported component or exhausted time budget it falls
   back to B and says so.

Neither checker selects actions, bans actions, interrupts plans or ends games: they return a
LedgerEntry only.
"""
from __future__ import annotations

import math
import random
import time
from fractions import Fraction
from itertools import combinations

import numpy as np

from arc3.algebra.linalg import nullspace, solve
from arc3.algebra.modular import gf2_solve_all
from arc3.diag import structure as S
from arc3.diag.ledger import LedgerEntry


def _split(comp, observations):
    elig, excl = [], []
    for o in observations:
        if not comp.explains(o):
            continue
        ok, why = comp.applies(o)
        if ok:
            elig.append(o)
        else:
            excl.append((o.id, why))
    return elig, excl


def _entry(comp, elig, excl):
    return LedgerEntry(component=comp.name, version=comp.version, kind=comp.kind, params=list(comp.params),
                       applicability=comp.applicability_text, eligible=[o.id for o in elig], excluded=excl,
                       n_params=len(comp.params))


def _val(v, fld):
    return (int(v) & 1) if fld == "gf2" else Fraction(v)


# ------------------------------------------------------------------ fitting (shared)
def _compile(poly):
    E = np.array(list(poly.terms.keys()), dtype=float).reshape(-1, poly.n)
    c = np.array([float(x) for x in poly.terms.values()])
    return E, c


def _peval(comp_poly, th):
    E, c = comp_poly
    if len(c) == 0:
        return 0.0, np.zeros(len(th))
    with np.errstate(all="ignore"):
        mon = np.prod(np.power(th, E), axis=1)
        f = float(c @ mon)
        J = np.zeros(len(th))
        for j in range(len(th)):
            Ej = E.copy()
            Ej[:, j] = np.maximum(Ej[:, j] - 1, 0)
            J[j] = float((c * E[:, j]) @ np.prod(np.power(th, Ej), axis=1))
    return f, J


def _lm_fit(cpolys, ys, n, rng, starts=6, iters=200):
    best, best_err = None, float("inf")
    y = np.array([float(v) for v in ys])
    for _ in range(starts):
        th = rng.uniform(-5, 5, n)
        lam = 1e-3
        for _ in range(iters):
            fJ = [_peval(cp, th) for cp in cpolys]
            r = np.array([f for f, _ in fJ]) - y
            J = np.array([g for _, g in fJ])
            err = float(r @ r)
            if err < 1e-24:
                break
            A = J.T @ J + lam * np.eye(n)
            try:
                step = np.linalg.solve(A, -J.T @ r)
            except np.linalg.LinAlgError:
                break
            th2 = th + step
            r2 = np.array([_peval(cp, th2)[0] for cp in cpolys]) - y
            if float(r2 @ r2) < err:
                th, lam = th2, lam / 3
            else:
                lam *= 5
        err = float(sum((_peval(cp, th)[0] - yy) ** 2 for cp, yy in zip(cpolys, y)))
        if err < best_err:
            best, best_err = th, err
    return best, best_err


class Fitter:
    """Fit a component to a subset of its eligible observations; predict any eligible one."""

    def __init__(self, comp, elig, seed=0, tol=1e-7):
        self.comp, self.elig, self.tol = comp, elig, tol
        self.rng = random.Random(seed)
        self.nprng = np.random.default_rng(seed)
        n = len(comp.params)
        self.n = n
        if comp.kind == "linear":
            self.rows = [comp.row(o.context) for o in elig]
        elif comp.kind == "polynomial":
            self.cpolys = [_compile(comp.poly(o.context)) for o in elig]
        self.ys = [o.value for o in elig]

    def fit(self, idx):
        c = self.comp
        if c.kind == "linear" and c.field == "gf2":
            sol = gf2_solve_all([self.rows[i] for i in idx], [int(self.ys[i]) & 1 for i in idx])
            if sol is None:
                return None
            x, basis = sol
            x = list(x)
            for v in basis:
                if self.rng.random() < 0.5:
                    x = [(a ^ b) for a, b in zip(x, v)]
            return x
        if c.kind == "linear":
            A = [self.rows[i] for i in idx]
            b = [Fraction(self.ys[i]) for i in idx]
            x = solve(A, b)
            if x is None:
                return None
            for v in nullspace(A):
                k = self.rng.randint(-1000, 1000)
                x = [a + k * b_ for a, b_ in zip(x, v)]
            return x
        th, err = _lm_fit([self.cpolys[i] for i in idx], [self.ys[i] for i in idx], self.n, self.nprng)
        scale = 1.0 + sum(float(self.ys[i]) ** 2 for i in idx)
        return th if th is not None and err <= 1e-14 * scale else None

    def predict(self, theta, i):
        c = self.comp
        if c.kind == "linear" and c.field == "gf2":
            return sum(a & b for a, b in zip(self.rows[i], theta)) & 1
        if c.kind == "linear":
            return sum(Fraction(a) * b for a, b in zip(self.rows[i], theta))
        return _peval(self.cpolys[i], theta)[0]

    def agrees(self, pred, i):
        y = self.ys[i]
        if self.comp.kind == "polynomial":
            return abs(pred - float(y)) <= self.tol * (1 + abs(float(y)))
        return pred == _val(y, self.comp.field)


# ------------------------------------------------------------------ B
class OrdinaryChecker:
    def __init__(self, k_max=3, draws=2000, poly_draws=300, seed=0):
        self.k_max, self.draws, self.poly_draws, self.seed = k_max, draws, poly_draws, seed

    def check(self, comp, observations):
        t0 = time.perf_counter()
        elig, excl = _split(comp, observations)
        e = _entry(comp, elig, excl)
        n = len(elig)
        if comp.kind == "unsupported" or n == 0:
            e.verdict = "unsupported" if comp.kind == "unsupported" else "no-observations"
            return e
        F = Fitter(comp, elig, seed=self.seed)
        m = min(len(comp.params), n)
        budget = self.poly_draws if comp.kind == "polynomial" else self.draws
        rng = random.Random(self.seed)
        if math.comb(n, m) <= budget:
            subsets = list(combinations(range(n), m))
        else:
            subsets = [tuple(sorted(rng.sample(range(n), m))) for _ in range(budget)]
        fits = []
        for sub in subsets:
            th = F.fit(sub)
            if th is None:
                continue
            preds = [F.predict(th, i) for i in range(n)]
            inl = frozenset(i for i in range(n) if F.agrees(preds[i], i))
            fits.append((inl, preds))
        if not fits:
            e.verdict = "hypothesis-inadequate"
            e.costs["seconds"] = time.perf_counter() - t0
            return e
        best = max(len(f[0]) for f in fits)
        tops = sorted({tuple(sorted(set(range(n)) - f[0])) for f in fits if len(f[0]) == best})
        ids = [o.id for o in elig]
        if best == n:
            e.verdict = "consistent"
        elif n - best > self.k_max:
            e.verdict = "hypothesis-inadequate"
        elif len(tops) == 1:
            e.verdict = "contradiction-localized"
        else:
            e.verdict = "contradiction-ambiguous"
        if e.verdict.startswith("contradiction"):
            e.diagnoses = [[ids[i] for i in t] for t in tops]
        # undetermined predictions: fits consistent with all others disagree on obs i
        unver = []
        for i in range(n):
            vals = set()
            for inl, preds in fits:
                if inl >= (frozenset(range(n)) - {i}) - frozenset(tops[0]):
                    p = preds[i]
                    vals.add(round(p, 6) if comp.kind == "polynomial" else p)
                if len(vals) > 1:
                    unver.append(ids[i])
                    break
        e.unverifiable = unver
        bad = set(x for d in e.diagnoses for x in d)
        e.supporting = [x for x in ids if x not in bad]
        e.contradicting = sorted(bad) if e.verdict == "contradiction-localized" else []
        e.costs["seconds"] = time.perf_counter() - t0
        e.costs["fits"] = len(subsets)
        return e


# ------------------------------------------------------------------ C
class AlgebraicChecker:
    def __init__(self, k_max=3, cache=None, fallback=None, time_budget=10.0, seed=0, max_subsets=60000):
        self.k_max, self.time_budget, self.seed = k_max, time_budget, seed
        self.cache = cache if cache is not None else S.StructureCache(enabled=True)
        self.fallback = fallback or OrdinaryChecker(k_max=k_max, seed=seed)
        self.max_subsets = max_subsets

    def check(self, comp, observations):
        if comp.kind == "unsupported":
            e = self.fallback.check(comp, observations)
            e.fallback = True
            return e
        try:
            return self._check(comp, observations)
        except Exception as ex:  # safe fallback: ordinary checking
            try:
                e = self.fallback.check(comp, observations)
            except Exception:
                e = _entry(comp, [], [])
                e.verdict = "error"
            e.fallback = True
            e.costs["fallback_reason"] = repr(ex)[:200]
            return e

    # structure depends only on (component version, context signature, design)
    def _structure(self, comp, elig):
        t0 = time.perf_counter()
        n_par = len(comp.params)
        meta = {}
        if comp.kind == "linear":
            rows = [comp.row(o.context) for o in elig]
            fs = None
            fld = comp.field
        else:
            fs = [comp.poly(o.context) for o in elig]
            rows, meta = S.jacobian_rows(fs, n_par, seed=self.seed)
            fld = "q"
        n = len(rows)
        r = S.rank(rows, fld)
        circuits, complete = S.all_circuits(rows, fld, max_subsets=self.max_subsets, seed=self.seed)
        coloops = [i for i in range(n) if S.rank(rows[:i] + rows[i + 1:], fld) < r]
        basis = []
        for i in range(n):
            if S.rank([rows[j] for j in basis + [i]], fld) > len(basis):
                basis.append(i)
        rels = []
        for C in circuits:
            if comp.kind == "linear":
                rels.append(("linear", S.linear_relation(rows, C, fld)))
            else:
                rel = S.polynomial_relation(fs, C)
                rels.append(("poly", rel) if rel is not None else ("none", None))
        pinned = []
        if comp.kind == "linear":
            for j in range(n_par):
                ej = [int(k == j) for k in range(n_par)]
                if S.rank(rows + [ej], fld) == r:
                    pinned.append(comp.params[j])
        return {"rows_rank": r, "n": n, "circuits": circuits, "complete": complete, "coloops": coloops,
                "basis": basis, "relations": rels, "pinned": pinned, "jacobian": meta, "field": fld,
                "setup_s": time.perf_counter() - t0}

    def _check(self, comp, observations):
        t0 = time.perf_counter()
        elig, excl = _split(comp, observations)
        e = _entry(comp, elig, excl)
        n = len(elig)
        if n == 0:
            e.verdict = "no-observations"
            return e
        sig = comp.context_signature([o.context for o in elig]) if comp.context_signature else None
        design = tuple(comp.design_key(o.context) for o in elig) if comp.design_key else tuple(o.id for o in elig)
        key = (comp.name, comp.version, sig, design)
        st = self.cache.get_or_compute(key, comp.assumptions(sig), lambda: self._structure(comp, elig))
        if time.perf_counter() - t0 > self.time_budget:
            raise TimeoutError("structure budget")
        ids = [o.id for o in elig]
        fld = st["field"]
        ys = [_val(o.value, fld) if comp.kind == "linear" else Fraction(o.value) for o in elig]
        e.rank = st["rows_rank"]
        e.independent = [ids[i] for i in st["basis"]]
        e.circuits_complete = st["complete"]
        e.unverifiable = [ids[i] for i in st["coloops"]]
        # --- evaluate relations exactly
        t1 = time.perf_counter()
        violated, usable = [], []
        for ci, (C, (kind, rel)) in enumerate(zip(st["circuits"], st["relations"])):
            if kind == "none":
                e.relations.append({"obs": [ids[i] for i in C], "relation": None, "satisfied": None,
                                    "note": "relation not derived within degree bound"})
                continue
            if kind == "linear":
                if fld == "gf2":
                    v = 0
                    for i in C:
                        v ^= ys[i]
                else:
                    v = sum(rel[i] * ys[i] for i in C)
                txt = "xor" if fld == "gf2" else {ids[i]: str(rel[i]) for i in C}
            else:
                v = rel.eval([ys[i] for i in C])
                txt = repr(rel)
            sat = (v == 0)
            usable.append(ci)
            if not sat:
                violated.append(ci)
            e.relations.append({"obs": [ids[i] for i in C], "relation": txt, "value": str(v), "satisfied": sat})
        eval_s = time.perf_counter() - t1
        # --- identifiability (never "unique" from rank)
        if comp.kind == "linear":
            e.identifiability = {"rank": e.rank, "n_params": e.n_params, "pinned_params": st["pinned"],
                                 "claim": "parameters listed as pinned are determined by the eligible design"}
            if len(st["pinned"]) < e.n_params:
                e.ambiguity["unidentified_params"] = [p for p in comp.params if p not in st["pinned"]]
        else:
            e.identifiability = {"jacobian_rank": e.rank, "n_params": e.n_params, **st["jacobian"],
                                 "claim": ("locally identified at a generic point (not a uniqueness claim)"
                                           if e.rank == e.n_params else "not locally identified")}
            self._discrete_ambiguity(comp, elig, e)
        ident = [x[0] for x in excl if "identity" in x[1]]
        if ident:
            e.ambiguity["identity_uncertain"] = ident
        # --- diagnosis by syndrome matching
        t2 = time.perf_counter()
        self._diagnose(e, st, usable, violated, ids, fld)
        e.costs.update({"structure_s": st["setup_s"], "relation_eval_s": eval_s,
                        "diagnosis_s": time.perf_counter() - t2, "total_s": time.perf_counter() - t0,
                        "circuits": len(st["circuits"]), "cache": dict(self.cache.stats)})
        return e

    def _diagnose(self, e, st, usable, violated, ids, fld):
        circuits = st["circuits"]
        n = len(ids)
        if not violated:
            e.verdict = "consistent"
            e.supporting = sorted({ids[i] for ci in usable for i in circuits[ci]})
            if e.unverifiable:
                e.suggestions.append({"type": "make-checkable", "obs": e.unverifiable,
                                      "why": "in no relation: a fault here cannot be detected"})
            return
        # bit masks of usable circuits per element
        pos = {ci: k for k, ci in enumerate(usable)}
        emask = [0] * n
        for ci in usable:
            for i in circuits[ci]:
                emask[i] |= 1 << pos[ci]
        vmask = sum(1 << pos[ci] for ci in violated)
        cand = sorted({i for ci in violated for i in circuits[ci]})
        found = []
        for s in range(1, self.k_max + 1):
            for D in combinations(cand, s):
                if any(set(f) <= set(D) for f in found):
                    continue
                m = 0
                for i in D:
                    m = (m ^ emask[i]) if fld == "gf2" else (m | emask[i])
                if m == vmask:
                    found.append(D)
        if not found:
            e.verdict = "hypothesis-inadequate"
            # change point (observations are in chronological order): the shortest prefix that
            # contains a violated relation; is everything from there on consistent by itself?
            vsets = [set(circuits[ci]) for ci in violated]
            cut = next(L for L in range(1, n + 1) if any(max(C) < L for C in vsets)) - 1
            suffix_ok = not any(min(C) >= cut for C in vsets)
            e.suspects = ids[cut:]
            e.contradicting = list(e.suspects)
            e.ambiguity["regimes"] = {"first_inconsistent": ids[cut], "earlier_consistent": ids[:cut],
                                      "later_self_consistent": suffix_ok}
            e.suggestions.append({"type": "revise-hypothesis",
                                  "why": f"no set of <= {self.k_max} observations explains the violated relations",
                                  "suspects": e.suspects,
                                  "check": ("two self-consistent regimes: suspect a mode change or identity swap "
                                            f"at {ids[cut]}" if suffix_ok else
                                            "no regime split: the hypothesis family is likely inadequate")})
            return
        e.diagnoses = [[ids[i] for i in D] for D in found]
        if len(found) == 1:
            e.verdict = "contradiction-localized"
            e.contradicting = e.diagnoses[0]
        else:
            e.verdict = "contradiction-ambiguous"
            counts = {}
            for D in found:
                for i in D:
                    counts[i] = counts.get(i, 0) + 1
            best = max(counts, key=lambda i: min(counts[i], len(found) - counts[i]))
            e.suggestions.append({"type": "reobserve", "obs": ids[best],
                                  "why": f"in {counts[best]} of {len(found)} minimal diagnoses; a fresh observation "
                                         "in the same context splits them"})
        bad = {x for d in e.diagnoses for x in d}
        e.supporting = sorted({ids[i] for ci in usable for i in circuits[ci]} - bad)

    def _discrete_ambiguity(self, comp, elig, e):
        from arc3.diag.poly import Poly
        n = len(comp.params)
        th = [Poly.var(n, j) for j in range(n)]
        unresolved = []
        for name, sym in comp.symmetries:
            img = sym(th)
            if all((comp.poly(o.context).substitute(img) - comp.poly(o.context)).is_zero() for o in elig):
                fix = next((ctx for ctx in comp.catalog
                            if not (comp.poly(ctx).substitute(img) - comp.poly(ctx)).is_zero()), None)
                unresolved.append(name)
                e.suggestions.append({"type": "distinguishing-context", "symmetry": name, "context": fix,
                                      "why": "all observations so far are invariant under this symmetry"})
        if unresolved:
            e.ambiguity["discrete_unresolved"] = unresolved
