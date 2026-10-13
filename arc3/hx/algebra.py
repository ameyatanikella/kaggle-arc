"""Contextual action algebra on observed transitions.

A word w = a_1...a_k acts on (latent) state as T_w. On a finite system the induced transformations
form a finite monoid, so every action satisfies a^{k+p} = a^k for some index k >= 0 and period
p >= 1 at every state (orbit of s under a is eventually periodic). Special cases: a = 1 (no-op),
a^2 = a (idempotent), a^2 = 1 (involution), a^m = 1 (cycle). Two-letter relations: ab = 1
(inverse), ab = ba (commute), ab = b (override: the second action erases the first), aba = b
(absorb).

Every relation is stored WITH ITS CONTEXT and is local by default:
  * an instance is an evaluation of the relation at one concrete state p, possible only when all
    transitions it mentions were observed from p;
  * instances are aggregated under context descriptors from coarse to fine
      L0  the word with clicks abstracted to (6, colour under the cursor, size bucket)
      L1  L0 + the object inventory of p (multiset of object signatures: a proxy for mode)
      L2  the concrete word + local patches under each click (keys: + inventory)
    and a prediction at a new state q uses the COARSEST descriptor of q with support >= min_support
    distinct states and ZERO counterexamples. One counterexample at a level disables that level
    for that descriptor; finer levels stay available (context refinement).
  * nothing is generalised from one state: min_support >= 2.

Predictions are used only to (i) predict an untried transition, (ii) shorten plans. Every
prediction is recorded; when the transition is later observed, the prediction is scored, and a
wrong one adds a counterexample. In a system whose dynamics are a function of the visible state
(visible-Markov), a pair of individually ineffective actions cannot have an effect; such a pair
effect (or any same-state/same-action conflict) is evidence of hidden state, which this module
reports (see homing.py) rather than models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

NOOP, IDEM, INVOL, INVERSE, COMMUTE, OVERRIDE, ABSORB, CYCLE = (
    "a=1", "aa=a", "aa=1", "ab=1", "ab=ba", "ab=b", "aba=b", "a^m=1")


@dataclass
class RelStat:
    support: set = field(default_factory=set)
    counter: set = field(default_factory=set)


def _bucket(n: int) -> int:
    return int(n).bit_length()


class Algebra:
    """Transition store + contextual relations.

    ctx(key, action) -> (class_token, local_token): class token abstracts a click (e.g. colour and
    component size bucket under the cursor); local token describes the click's neighbourhood.
    inventory(key) -> token for the object inventory of the state (mode proxy). Both default to
    trivial tokens (then L0 == L1 == L2 up to concrete actions).
    """

    def __init__(self, ctx: Optional[Callable] = None, inventory: Optional[Callable] = None,
                 min_support: tuple = (3, 2, 2), max_cycle: int = 8, scope: str = ""):
        self.T: dict = {}            # key -> {action: key}
        self.P: dict = {}            # key -> set((pred_key, action))
        self.stats: dict = {}        # (rel, level, descriptor) -> RelStat
        self.ctx = ctx or (lambda k, a: (a if a[0] != 6 else (6,), None))
        self.inventory = inventory or (lambda k: None)
        self.min_support = min_support
        self.max_cycle = max_cycle
        self.scope = scope
        self.conflicts: list = []    # (key, action, old, new): same state + action, different result
        self.pred_log: dict = {}     # (key, action) -> (predicted key, rel, level)
        self.n = {"predictions": 0, "pred_checked": 0, "pred_correct": 0, "pred_wrong": 0,
                  "instances": 0, "refuted_levels": 0, "abstain_conflict": 0, "prior_used": 0}
        self.prior: dict = {}        # (rel, descriptor) -> [support, counter] from earlier scopes
        self._inv_cache: dict = {}
        self._ctx_cache: dict = {}

    # ------------------------------------------------------------------ descriptors
    def _cls(self, k, a):
        c = self._ctx_cache.get((k, a))
        if c is None:
            c = self._ctx_cache[(k, a)] = self.ctx(k, a)
        return c

    def _inv(self, k):
        v = self._inv_cache.get(k)
        if v is None:
            v = self._inv_cache[k] = self.inventory(k)
        return v

    def descriptors(self, k, word: tuple) -> tuple:
        cls = tuple(self._cls(k, a)[0] for a in word)
        loc = tuple((a, self._cls(k, a)[1]) if a[0] == 6 else a for a in word)
        inv = self._inv(k)
        return ((0, cls), (1, cls, inv), (2, loc, inv))

    # ------------------------------------------------------------------ observation
    def succ(self, k, a):
        return self.T.get(k, {}).get(a)

    def observe(self, k, a, u) -> dict:
        """Add an observed transition k --a--> u. Returns {'conflict': bool, 'predicted': ...}."""
        info = {"conflict": False, "prediction": None}
        d = self.T.setdefault(k, {})
        old = d.get(a)
        if old is not None and old != u:
            self.conflicts.append((k, a, old, u))
            info["conflict"] = True
        d[a] = u
        self.P.setdefault(u, set()).add((k, a))
        pk = self.pred_log.pop((k, a), None)
        if pk is not None:
            self.n["pred_checked"] += 1
            info["prediction"] = pk
            if pk[0] == u:
                self.n["pred_correct"] += 1
            else:
                self.n["pred_wrong"] += 1
        if old is None or old != u:
            self._evaluate_around(k, a)
        return info

    def _add(self, rel, k, word, holds: bool):
        self.n["instances"] += 1
        for desc in self.descriptors(k, word):
            st = self.stats.setdefault((rel, desc), RelStat())
            if holds:
                st.support.add(k)
            else:
                if not st.counter:
                    self.n["refuted_levels"] += 1
                st.counter.add(k)

    def _evaluate_at(self, p, a, b=None):
        """Evaluate every relation on word (a) / (a, b) at state p whose transitions are known."""
        T = self.T
        u = self.succ(p, a)
        if u is None:
            return
        if b is None:
            self._add(NOOP, p, (a,), u == p)
            if u != p:
                uu = self.succ(u, a)
                if uu is not None:
                    self._add(IDEM, p, (a,), uu == u)
                    self._add(INVOL, p, (a,), uu == p)
                self.note_cycles(p, a)
            return
        if a == b:
            return
        ub = self.succ(u, b)
        if ub is None:
            return
        a_eff = u != p
        vb = self.succ(p, b)
        if a_eff:                                   # precondition: a changed the state
            self._add(INVERSE, p, (a, b), ub == p)
        if vb is not None:
            va = self.succ(vb, a)
            if va is not None and a_eff and vb != p:   # precondition: both effective at p
                self._add(COMMUTE, p, (a, b), ub == va)
            if a_eff:
                self._add(OVERRIDE, p, (a, b), ub == vb)
                uba = self.succ(ub, a)
                if uba is not None:
                    self._add(ABSORB, p, (a, b), uba == vb)

    def _evaluate_around(self, k, a):
        """Re-evaluate the instances that mention the edge (k, a)."""
        T = self.T
        self._evaluate_at(k, a)
        # orbits under a that pass through k may now be complete (cycles)
        cur, seen = k, {k}
        for _ in range(self.max_cycle):
            prev = [q for q, x in self.P.get(cur, ()) if x == a and q not in seen]
            if len(prev) != 1:
                break
            cur = prev[0]
            seen.add(cur)
            self.note_cycles(cur, a)
        for b in list(T.get(k, {})):
            if b != a:
                self._evaluate_at(k, a, b)
                self._evaluate_at(k, b, a)
        u = self.succ(k, a)
        # (k, a) as a second step: p --x--> k --a--> u
        for p, x in list(self.P.get(k, ())):
            if p == k:
                continue
            if x == a:
                self._evaluate_at(p, a)
            else:
                self._evaluate_at(p, x, a)
                # commutation at p also needs the other path p --a--> v --x-->
                v = self.succ(p, a)
                if v is not None:
                    self._evaluate_at(p, a, x)
            # absorb at p's predecessor: q --a--> p --x--> k --a--> u  (word (a, x))
            for q, y in list(self.P.get(p, ())):
                if y == a and q != p:
                    self._evaluate_at(q, a, x)

    # ------------------------------------------------------------------ prediction
    def holds(self, rel, k, word) -> Optional[tuple]:
        """Return (level, support) if the relation is predicted to hold at k for word, else None.
        Direct evaluation at k counts too (L3 = the exact state)."""
        for desc in self.descriptors(k, word):
            st = self.stats.get((rel, desc))
            lvl = desc[0]
            if st is not None and st.counter:
                continue                                   # refuted in this scope at this level
            here = len(st.support) if st is not None else 0
            if here >= self.min_support[lvl]:
                return lvl, here
            # memory: statistics carried from earlier levels count only after one confirmation here,
            # and only if they were never refuted there
            p = self.prior.get((rel, desc)) if self.prior else None
            if p is not None and p[1] == 0 and here >= 1 and p[0] + here >= self.min_support[lvl]:
                self.n["prior_used"] += 1
                return lvl, here + p[0]
        return None

    def predict(self, r, b) -> Optional[tuple]:
        """Predict T(r, b) for an untried (r, b). Returns (key, rel, level) or None. Conflicting
        predictions abstain."""
        if self.succ(r, b) is not None:
            return None
        cands = {}

        def add(key, rel, lvl):
            cands.setdefault(key, (rel, lvl))

        h = self.holds(NOOP, r, (b,))
        if h:
            add(r, NOOP, h[0])
        for q, a in self.P.get(r, ()):
            if q == r:
                continue
            if a == b:
                h = self.holds(INVOL, q, (b,))
                if h:
                    add(q, INVOL, h[0])
                h = self.holds(IDEM, q, (b,))
                if h:
                    add(r, IDEM, h[0])
            else:
                h = self.holds(INVERSE, q, (a, b))
                if h:
                    add(q, INVERSE, h[0])
                v = self.succ(q, b)
                if v is not None:
                    h = self.holds(OVERRIDE, q, (a, b))
                    if h:
                        add(v, OVERRIDE, h[0])
                    w = self.succ(v, a)
                    if w is not None:
                        h = self.holds(COMMUTE, q, (a, b))
                        if h:
                            add(w, COMMUTE, h[0])
            # absorb: q0 --b--> q --a--> r, predict T(r, b) = T(q0, a) when (b a b) = a at q0
            for q0, y in self.P.get(q, ()):
                if y == b and q0 != q and a != b:
                    v0 = self.succ(q0, a)
                    if v0 is not None:
                        h = self.holds(ABSORB, q0, (b, a))
                        if h:
                            add(v0, ABSORB, h[0])
        cyc = self._cycle_predict(r, b)
        if cyc is not None:
            add(cyc[0], CYCLE, cyc[1])
        if not cands:
            return None
        if len(cands) > 1:
            self.n["abstain_conflict"] += 1
            return None
        key, (rel, lvl) = next(iter(cands.items()))
        return key, rel, lvl

    def _cycle_predict(self, r, b):
        """Walk back along b-edges: q0 -b-> q1 ... -b-> r (m-1 steps). If a^m = 1 was seen (with
        the same descriptor) at >= min_support states and never refuted, predict T(r, b) = q0."""
        chain = [r]
        cur = r
        for _ in range(self.max_cycle - 1):
            prev = [q for q, a in self.P.get(cur, ()) if a == b and q != cur]
            if len(prev) != 1:
                break
            cur = prev[0]
            if cur in chain:
                break
            chain.append(cur)
            m = len(chain)
            h = self.holds(("%s|%d" % (CYCLE, m)), cur, (b,))
            if h:
                return cur, h[0]
        return None

    def note_cycles(self, k, a):
        """Evaluate a^m = 1 at k for m = 2..max_cycle when the orbit of k under a is known.
        index/period of the orbit (finite monoid: a^{idx+per} = a^idx) is also returned."""
        orbit = [k]
        cur = k
        for _ in range(self.max_cycle + 1):
            nxt = self.succ(cur, a)
            if nxt is None:
                return None
            if nxt in orbit:
                idx = orbit.index(nxt)
                per = len(orbit) - idx
                for m in range(3, self.max_cycle + 1):         # a^m = 1 at k  <=>  idx == 0 and per | m
                    self._add("%s|%d" % (CYCLE, m), k, (a,), idx == 0 and m % per == 0)
                return idx, per
            orbit.append(nxt)
            cur = nxt
        return None

    def record_prediction(self, k, a, pred):
        self.n["predictions"] += 1
        self.pred_log[(k, a)] = pred

    # ------------------------------------------------------------------ reporting
    def relations(self, min_support: int = 2, limit: int = 30) -> list:
        """Relations that currently hold at some context level with zero counterexamples, plus the
        refuted coarse ones (to show where context matters). Sorted by support."""
        out = []
        for (rel, desc), st in self.stats.items():
            if len(st.support) >= min_support or st.counter:
                out.append({"rel": rel, "level": desc[0], "word": desc[1], "context": desc[2:] if len(desc) > 2 else (),
                            "support": len(st.support), "counter": len(st.counter)})
        out.sort(key=lambda r: (-(r["counter"] == 0), r["level"], -r["support"]))
        return out[:limit]

    def precision(self) -> Optional[float]:
        c = self.n["pred_checked"]
        return None if c == 0 else self.n["pred_correct"] / c
