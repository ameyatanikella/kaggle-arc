"""Predictive-history state splitting and an undo-stack model (HX2).

Two histories may share an abstract state only while short action continuations predict the same
outcomes. When the same visible state and action produced different outcomes (a conflict), the
apparent state omitted something. HistorySplitter looks for the SIMPLEST feature, from a fixed,
game-independent list written before any run, that makes the outcome a function of (state, action,
feature) on every observed instance:
    prev_action      the previous action (exact, incl. click target)
    prev_same        whether the previous action equals this one
    prev_effective   whether the previous action changed the board
    count_mod2/3/4   times this action was taken on this level, mod k (counters, toggles, cycles)
    parity:<a>       times ANOTHER key action a was taken on this level, mod 2 (a latch it toggles)
A feature is adopted only when it is consistent with every instance, takes >= 2 values and each value
was seen >= 2 times (no claim from single observations). A split predicts only for feature values
already observed with that (state, action); otherwise it abstains. Unexplained conflicts are kept as such (no randomness is claimed).

UndoModel: UNDO's result depends on the undo stack, not on the board. Prediction = the decision
state before the most recent effective non-UNDO action (the stack is popped when UNDO is observed).
"""
from __future__ import annotations

from collections import Counter, defaultdict

FEATURES = ("prev_same", "prev_effective", "count_mod2", "prev_action", "count_mod3", "count_mod4")


class HistoryTracker:
    """Per-level action history needed for the features."""

    def __init__(self):
        self.reset_level()

    def reset_level(self):
        self.prev_action = None
        self.prev_effective = None
        self.counts = Counter()

    def features(self, action) -> dict:
        c = self.counts[action]
        f = {"prev_action": self.prev_action, "prev_same": self.prev_action == action,
             "prev_effective": self.prev_effective, "count_mod2": c % 2, "count_mod3": c % 3,
             "count_mod4": c % 4}
        for a, n in self.counts.items():            # latch / toggle set by another action
            if a != action and a[0] != 6:
                f["parity:%s" % (a,)] = n % 2
        return f

    def update(self, action, effective: bool):
        self.counts[action] += 1
        self.prev_action = action
        self.prev_effective = effective


class HistorySplitter:
    def __init__(self):
        self.inst = defaultdict(list)      # (key, action) -> [(features, outcome)]
        self.split = {}                    # (key, action) -> feature name | None (unexplained)
        self.by_action = defaultdict(Counter)   # action -> feature -> number of explained conflicts

    def observe(self, key, action, feats: dict, outcome) -> dict:
        lst = self.inst[(key, action)]
        lst.append((feats, outcome))
        if len({o for _, o in lst}) < 2:
            return {"conflict": False}
        first = (key, action) not in self.split
        found = None
        names = list(FEATURES) + sorted({k for fe, _ in lst for k in fe if k.startswith("parity:")})
        for f in names:                         # simplest first; a split needs >= 2 observations per value
            m, cnt = {}, Counter()
            ok = True
            for fe, o in lst:
                v = fe.get(f)
                cnt[v] += 1
                if m.setdefault(v, o) != o:
                    ok = False
                    break
            if ok and len(m) >= 2 and min(cnt.values()) >= 2:
                found = f
                break
        if found is not None and self.split.get((key, action)) != found:
            self.by_action[action][found] += 1
        self.split[(key, action)] = found
        return {"conflict": True, "feature": found, "first": first}

    @property
    def counts(self) -> dict:
        vals = list(self.split.values())
        return {"conflicts": len(vals), "explained": sum(v is not None for v in vals),
                "unexplained": sum(v is None for v in vals)}

    def history_dependent(self, key, action) -> bool:
        return (key, action) in self.split

    def predict(self, key, action, feats: dict):
        f = self.split.get((key, action))
        if f is None:
            return None
        for fe, o in reversed(self.inst[(key, action)]):
            if fe.get(f) == feats.get(f):
                return o
        return None

    def explain(self, key, action) -> str:
        f = self.split.get((key, action))
        n = len(self.inst[(key, action)])
        if f is None:
            return f"no single history feature explains the {n} observations yet"
        return {"prev_same": "whether the previous action was the same",
                "prev_effective": "whether the previous action changed the board",
                "count_mod2": "how many times this action was taken on this level (even/odd)",
                "count_mod3": "how many times this action was taken on this level (mod 3)",
                "count_mod4": "how many times this action was taken on this level (mod 4)",
                "prev_action": "which action came just before"}.get(
            f, f"how many times {f.split(':', 1)[-1]} was pressed on this level (even/odd)") + \
            f" ({n} of {n} observations consistent)"


class UndoModel:
    UNDO = 7

    def __init__(self):
        self.stack = []
        self.n = {"predictions": 0, "correct": 0}

    def reset_level(self, start_key=None):
        self.stack = []

    def predict(self, key, action):
        if action[0] != self.UNDO:
            return None
        return self.stack[-1] if self.stack else key

    def observe(self, pre_key, action, post_key):
        if action[0] == self.UNDO:
            if self.stack and self.stack[-1] == post_key:
                self.stack.pop()
            elif not self.stack and post_key != pre_key:
                pass
            return
        if post_key != pre_key:
            self.stack.append(pre_key)


class UndoLastModel:
    """Single-level UNDO: returns to the decision state before the most recent action (of any kind,
    UNDO included, so a second UNDO re-does). Competes with UndoModel (stack) in the ledger."""
    UNDO = 7

    def __init__(self):
        self.last_pre = None

    def reset_level(self, start_key=None):
        self.last_pre = None

    def predict(self, key, action):
        if action[0] != self.UNDO:
            return None
        return self.last_pre if self.last_pre is not None else key

    def observe(self, pre_key, action, post_key):
        self.last_pre = pre_key


class UndoAllModel(UndoModel):
    """UNDO stack that records EVERY action (no-ops too): UNDO after a no-op changes nothing."""

    def observe(self, pre_key, action, post_key):
        if action[0] == self.UNDO:
            if self.stack:
                self.stack.pop()
            return
        self.stack.append(pre_key)
