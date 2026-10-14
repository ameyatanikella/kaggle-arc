"""Goal ledger: temporal goal hypotheses that separate PREREQUISITES from SUFFICIENT win conditions,
refuted live by non-winning states (HX2; docs/prereg_hx2.md).

Hypotheses (over the fact grammar of goals.py):
  ('fact', f)         f holds in the frame
  ('and', f, g)       f and g hold together
  ('then', f, g)      f was achieved earlier on this level (bounded event memory) and g holds now

Evidence rules (deterministic, local to a level; a win supports a hypothesis but never proves it):
  * proposed from a WIN frame: true in the win frame and false at the level's start;
  * a hypothesis that holds in a non-winning decision state is NOT sufficient: its status becomes
    'prerequisite' (it may still be necessary). This is the live refutation the brief asks for;
  * a hypothesis false in a later level's win frame is not necessary there ('win_false' counted);
  * a hypothesis already true in a new level's first frame cannot be sufficient there.
Statuses: 'sufficient?' (never refuted), 'prerequisite' (refuted as sufficient). Nothing is deleted.
"""
from __future__ import annotations

from itertools import combinations
from typing import Optional

import numpy as np

from .goals import describe as describe_fact
from .goals import distance as fact_distance
from .goals import facts as frame_facts

KIND_COST = {"fact": 0, "and": 1, "then": 2}


def describe(h: tuple) -> str:
    if h[0] == "fact":
        return describe_fact(h[1])
    if h[0] == "and":
        return f"{describe_fact(h[1])} AND {describe_fact(h[2])}"
    return f"{describe_fact(h[1])} achieved earlier, THEN {describe_fact(h[2])}"


class GoalLedger:
    def __init__(self, mask_fn=None, max_pairs: int = 10, max_new: int = 40):
        self.mask_fn = mask_fn or (lambda: None)
        self.max_pairs = max_pairs
        self.max_new = max_new
        self.h: dict = {}                 # hypothesis -> record
        self.level = None
        self.start_facts: set = set()
        self.achieved: set = set()        # facts achieved on this level (true at some state, not at start)
        self.seen: set = set()            # state keys already evaluated on this level
        self.order: list = []             # (key, facts) of distinct states on this level, in order
        self.events: list = []            # refutations for reporting: (level, hypothesis, kind)
        self.n = {"states": 0, "wins": 0, "proposed": 0, "refuted_sufficient": 0, "win_false": 0,
                  "transfer_explained": 0, "transfer_checked": 0, "abstain": 0}

    # ------------------------------------------------------------------ evaluation
    def holds(self, h: tuple, fs: set, achieved: set) -> bool:
        if h[0] == "fact":
            return h[1] in fs
        if h[0] == "and":
            return h[1] in fs and h[2] in fs
        return h[1] in achieved and h[2] in fs

    def start_level(self, level: int, grid: np.ndarray, key=None):
        self.level = level
        fs = set(frame_facts(grid, self.mask_fn()))
        self.start_facts = fs
        self.achieved = set()
        self.seen = set()
        self.order = []
        # already true at a level's first frame -> cannot be sufficient here
        for h, r in self.h.items():
            if r["status"] == "sufficient?" and self.holds(h, fs, set()):
                self._refute(h, "true at level start")
        if key is not None:
            self.seen.add(key)
            self.order.append((key, fs))

    def observe_state(self, key, grid: np.ndarray) -> list:
        """A decision state that did NOT end the level. Returns hypotheses refuted as sufficient now."""
        if self.level is None or key in self.seen:
            return []
        self.seen.add(key)
        fs = set(frame_facts(grid, self.mask_fn()))
        if not fs:
            self.n["abstain"] += 1
            return []
        self.n["states"] += 1
        out = []
        for h, r in self.h.items():
            if r["status"] == "sufficient?" and self.holds(h, fs, self.achieved):
                self._refute(h, "held in a non-winning state")
                out.append(h)
        self.achieved |= (fs - self.start_facts)
        self.order.append((key, fs))
        return out

    def _refute(self, h, why):
        r = self.h[h]
        r["status"] = "prerequisite"
        r["nonwin"] += 1
        r["refuted_level"] = self.level
        r["why"] = why
        self.n["refuted_sufficient"] += 1
        self.events.append((self.level, h, why))

    # ------------------------------------------------------------------ wins
    def observe_win(self, level: int, win: Optional[np.ndarray]) -> list:
        """The level was completed; `win` is the faithful WIN frame (None = abstain). Returns the new
        hypotheses proposed from this win (sufficient? first)."""
        self.n["wins"] += 1
        if win is None:
            self.n["abstain"] += 1
            return []
        fw = set(frame_facts(win, self.mask_fn()))
        if not fw:
            self.n["abstain"] += 1
            return []
        # existing hypotheses: explained or not necessary here
        for h, r in self.h.items():
            if level in r["support"]:
                continue
            self.n["transfer_checked"] += 1
            if self.holds(h, fw, self.achieved):
                r["support"].append(level)
                self.n["transfer_explained"] += 1
            else:
                r["win_false"] += 1
                self.n["win_false"] += 1
        new_facts = sorted(fw - self.start_facts)
        states = [fs for _, fs in self.order]
        def nonwin(pred):
            return sum(1 for i, fs in enumerate(states) if pred(i, fs))
        cand = []
        single_cnt = {f: nonwin(lambda i, fs, f=f: f in fs) for f in new_facts}
        for f in new_facts:
            cand.append((("fact", f), single_cnt[f]))
        # conjunctions of facts that are individually not sufficient
        weak = sorted([f for f in new_facts if single_cnt[f] > 0], key=lambda f: (single_cnt[f], f))[:self.max_pairs]
        for f, g in combinations(weak, 2):
            cand.append((("and", f, g), nonwin(lambda i, fs, f=f, g=g: f in fs and g in fs)))
        # temporal: f achieved before the win (not at start), then g at the win
        earlier = sorted(self.achieved - set(new_facts))[:self.max_pairs]
        prefix_ach = []
        acc = set()
        for fs in states:
            prefix_ach.append(set(acc))
            acc |= (fs - self.start_facts)
        for f in earlier:
            for g in weak[:5]:
                cnt = sum(1 for i, fs in enumerate(states) if f in prefix_ach[i] and g in fs)
                cand.append((("then", f, g), cnt))
        cand.sort(key=lambda hc: (hc[1] > 0, KIND_COST[hc[0][0]], hc[1], hc[0]))
        out = []
        for h, cnt in cand[:self.max_new]:
            if h in self.h:
                continue
            self.h[h] = {"support": [level], "status": "sufficient?" if cnt == 0 else "prerequisite",
                         "nonwin": cnt, "win_false": 0, "proposed_level": level, "why": "" if cnt == 0 else
                         "held earlier on its own level"}
            self.n["proposed"] += 1
            out.append(h)
        return out

    # ------------------------------------------------------------------ queries
    def ranked(self, status: Optional[str] = "sufficient?", limit: int = 8) -> list:
        items = [(h, r) for h, r in self.h.items() if status is None or r["status"] == status]
        items.sort(key=lambda hr: (-len(hr[1]["support"]), hr[1]["win_false"], KIND_COST[hr[0][0]],
                                   hr[1]["nonwin"], hr[0]))
        return items[:limit]

    def distance(self, h: tuple, grid: np.ndarray) -> float:
        mask = self.mask_fn()
        if h[0] == "fact":
            return fact_distance(h[1], grid, mask)
        if h[0] == "and":
            return fact_distance(h[1], grid, mask) + fact_distance(h[2], grid, mask)
        done = h[1] in self.achieved
        return (0.0 if done else fact_distance(h[1], grid, mask)) + fact_distance(h[2], grid, mask)

    def score(self, grid: np.ndarray, top: int = 3) -> float:
        r = self.ranked("sufficient?", top)
        return float(sum(self.distance(h, grid) for h, _ in r)) if r else 0.0

    def take_events(self) -> list:
        ev, self.events = self.events, []
        return ev
