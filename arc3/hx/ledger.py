"""Prospective forecast ledger over candidate models (HX2).

Every candidate model forecasts the successor state of the action about to be scored BEFORE the
outcome is used (the caller computes forecasts from the pre-state, then reveals the outcome). The
ledger keeps, per model and per action class:
  coverage          share of actions the model made a forecast for
  accuracy          exact successor right / forecasts
  effect accuracy   among actions whose outcome CHANGED the board: exact successor right / forecasts
                    (a correct "nothing happens" is not understanding, so it is reported apart)
  change precision  among forecasts that something would change: the board changed / forecasts
Weights: w(m) proportional to 2^-L(m) * prod P(outcome), P = 1-eps if right, eps if wrong; an
abstention neither credits nor penalises. L(m) (bits) is fixed before any run (MODEL_BITS).
Wrong forecasts are classified: 'binding' (object identity or camera ambiguous on that step),
'latent' (the same state and action gave different outcomes: hidden state), 'transition' (else).
certified(): every forecasting model (>= min_models) agrees, each has >= 0.9 prospective accuracy on
>= 5 checks of that action class, and the action is not history-dependent. Only then may a host
hard-prune; otherwise a prediction is a soft search prior.
"""
from __future__ import annotations

import math
from collections import defaultdict

MODEL_BITS = {"graph": 8, "algebra": 4, "mover": 6, "history": 6, "undo": 2, "undo1": 2, "undo_all": 2}
EPS = 0.05


def _rec():
    return {"n": 0, "right": 0, "eff_n": 0, "eff_right": 0, "chg_n": 0, "chg_did": 0}


class ForecastLedger:
    def __init__(self, bits=None):
        self.bits = dict(bits or MODEL_BITS)
        self.logw = {m: -b * math.log(2) for m, b in self.bits.items()}
        self.tot = {m: _rec() for m in self.bits}
        self.by_cls = defaultdict(lambda: {m: _rec() for m in self.bits})
        self.actions = 0
        self.failures = defaultdict(lambda: defaultdict(int))   # model -> cause -> count
        self.counterexamples = []                               # last few (model, action cls, cause)

    def score(self, pre_key, action_cls, forecasts: dict, outcome_key, cause: str = "transition") -> dict:
        """forecasts: model -> predicted successor key or None (abstain), computed before the outcome."""
        self.actions += 1
        changed = outcome_key != pre_key
        out = {}
        for m, p in forecasts.items():
            if p is None or m not in self.bits:
                continue
            ok = p == outcome_key
            for r in (self.tot[m], self.by_cls[action_cls][m]):
                r["n"] += 1
                r["right"] += ok
                if changed:
                    r["eff_n"] += 1
                    r["eff_right"] += ok
                if p != pre_key:
                    r["chg_n"] += 1
                    r["chg_did"] += changed
            self.logw[m] += math.log(1 - EPS) if ok else math.log(EPS)
            out[m] = ok
            if not ok:
                self.failures[m][cause] += 1
                self.counterexamples.append((m, action_cls, cause))
                del self.counterexamples[:-20]
        return out

    def accuracy(self, m, action_cls=None):
        r = self.tot[m] if action_cls is None else self.by_cls[action_cls][m]
        return (r["right"] / r["n"]) if r["n"] else None, r["n"]

    def certified(self, forecasts: dict, action_cls, history_dependent: bool = False, min_models: int = 1) -> bool:
        """Hard-prune licence relative to the current version space: every model that forecasts agrees,
        each has >= 0.9 prospective accuracy on >= 5 checks of this action class, and the action is not
        history-dependent. Abstaining models do not object; min_models sets how many must forecast."""
        if history_dependent:
            return False
        ps = [(m, p) for m, p in forecasts.items() if p is not None]
        if len(ps) < min_models or len({p for _, p in ps}) != 1:
            return False
        for m, _ in ps:
            acc, n = self.accuracy(m, action_cls)
            if n < 5 or acc < 0.9:
                return False
        return True

    def weights(self) -> dict:
        mx = max(self.logw.values())
        z = sum(math.exp(v - mx) for v in self.logw.values())
        return {m: math.exp(v - mx) / z for m, v in self.logw.items()}

    def summary(self) -> dict:
        def fmt(r):
            return {"forecasts": r["n"], "accuracy": round(r["right"] / r["n"], 3) if r["n"] else None,
                    "effect_accuracy": round(r["eff_right"] / r["eff_n"], 3) if r["eff_n"] else None,
                    "effect_n": r["eff_n"],
                    "change_precision": round(r["chg_did"] / r["chg_n"], 3) if r["chg_n"] else None,
                    "coverage": round(r["n"] / self.actions, 3) if self.actions else None}
        return {"actions": self.actions, "weights": {m: round(w, 3) for m, w in self.weights().items()},
                "models": {m: fmt(self.tot[m]) for m in self.bits},
                "failures": {m: dict(c) for m, c in self.failures.items()}}
