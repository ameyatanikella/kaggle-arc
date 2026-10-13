"""HX session for an LLM host: faithful events, contextual algebra, goal evidence, triggers.

Host-agnostic (numpy + scipy, no harness imports) so it is unit-testable; the harness binding is in
kaggle/b2/hx_patch.py. One HXSession per game. It never acts. It produces:
  * reports, only when a trigger fires (never on every turn):
      level   - a level was completed: what changed in the winning step (object level, from the
                faithful WIN frame), facts true at the win and in none of the level's earlier
                states, and how those candidates stand on the new level's first frame;
      hidden  - the same board and action gave different results (state the board does not show);
      noops   - several consecutive actions changed nothing inside the board: the relations
                verified on this level that cover the actions tried (with their contexts);
  * answers to sandbox requests (goals, relations, wins, status).
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np

from ..perception import VolatileMask
from .algebra import COMMUTE, IDEM, INVERSE, INVOL, NOOP, OVERRIDE, Algebra
from .context import GridContext
from .events import DEATH, GAME_WIN, MOVE, RESET, WIN, EventLog
from .goals import GoalModel, describe
from .objects import describe_match, match, parse

NOOP_STREAK = 4          # consecutive interior no-ops (across calls) before the noops report
MAX_REPORT_LINES = 9
REL_TEXT = {NOOP: "changed nothing", IDEM: "a second press changed nothing more", INVOL: "a second press undid it",
            INVERSE: "the second undid the first", COMMUTE: "order did not matter",
            OVERRIDE: "the second erased the first's effect"}


def action_text(a: tuple) -> str:
    if a[0] == 6:
        return f"MOUSE(row={a[2]}, col={a[1]})"
    if a[0] == 0:
        return "RESET"
    if a[0] == 7:
        return "UNDO"
    return f"ACTION{a[0]}"


def word_text(word) -> str:
    out = []
    for t in word:
        if isinstance(t, tuple) and t and t[0] == 6:
            out.append("click(colour %s, size~2^%s)" % (t[1], t[2]) if len(t) >= 3 else "click")
        elif isinstance(t, tuple):
            out.append(action_text(t))
        else:
            out.append(str(t))
    return " then ".join(out)


class HXSession:
    def __init__(self, triggers: bool = True):
        self.log = EventLog()
        self.vm = VolatileMask()
        self.ctx = GridContext(self.log, self.vm)
        self.alg: Optional[Algebra] = None
        self.alg_level = -1
        self.goals = GoalModel(mask_fn=lambda: self.vm.mask)
        self.triggers = triggers
        self.pending: list = []
        self.noop_streak = 0
        self.noop_fired = False
        self.conflicts_reported: set = set()
        self.last_win_report: dict = {}
        self.stats = Counter()
        self.started = False

    # ------------------------------------------------------------------ recording
    def start(self, grid, level: int = 0):
        self.log.start(np.asarray(grid, dtype=np.uint8), level)
        self.started = True

    def _algebra(self, level: int, rebuild: bool):
        if self.alg is None or self.alg_level != level or rebuild:
            self.alg = Algebra(ctx=self.ctx.ctx, inventory=self.ctx.inventory)
            self.alg_level = level
            for p0, a0, p1, _ in self.log.transitions(level=level):
                self.alg.observe(self.ctx.key(p0), a0, self.ctx.key(p1))
            self.stats["algebra_rebuilds"] += 1
        return self.alg

    def record(self, action: tuple, frames, state: str, levels_before: int, levels_after: int,
               legal=(), pre_grid=None, automatic: bool = False) -> dict:
        if not self.started:
            self.start(pre_grid if pre_grid is not None else frames[0], levels_before)
        pre = np.asarray(pre_grid, dtype=np.uint8) if pre_grid is not None else None
        st = self.log.record(action, frames, state, levels_before, levels_after, legal=legal, pre_grid=pre,
                             automatic=automatic)
        self.stats["steps"] += 1
        self.stats["kind_" + st.kind] += 1
        out = {"kind": st.kind}
        if st.kind == MOVE:
            g0, g1 = self.log.grid(st.pre), self.log.grid(st.post)
            old = self.vm.mask.copy()
            self.vm.update(g0, g1)
            grown = not np.array_equal(old, self.vm.mask)
            k0, k1 = self.ctx.key(st.pre), self.ctx.key(st.post)
            alg = self._algebra(st.level, grown)
            if not grown:
                info = alg.observe(k0, st.action, k1)
            else:
                info = {"conflict": False}          # the rebuild already contains this step
                alg.observe(k0, st.action, k1)
            if info.get("conflict"):
                self.stats["conflicts"] += 1
                self._hidden_report(st, alg)
            if k0 == k1:
                self.noop_streak += 1
                if self.noop_streak >= NOOP_STREAK and not self.noop_fired:
                    self._noop_report(st, alg)
                    self.noop_fired = True
            else:
                self.noop_streak, self.noop_fired = 0, False
        elif st.kind in (WIN, GAME_WIN):
            self.noop_streak, self.noop_fired = 0, False
            self._win_report(st)
        elif st.kind == DEATH:
            self.noop_streak, self.noop_fired = 0, False
            if st.terminal is not None:
                self.goals.observe_death(self.log.grid(st.pre), self.log.grid(st.terminal))
        elif st.kind == RESET:
            self.noop_streak, self.noop_fired = 0, False
        return out

    # ------------------------------------------------------------------ triggers
    def _push(self, kind: str, lines: list):
        if not self.triggers or not lines:
            return
        lines = lines[:MAX_REPORT_LINES]
        self.pending.append((kind, lines))
        self.stats["trigger_" + kind] += 1
        self.stats["report_chars"] += sum(len(x) + 1 for x in lines)

    def _win_report(self, st):
        lv = st.level
        visited = [self.log.grid(s.pre) for s in self.log.level_steps(lv)]
        start = self.log.grid(self.log.level_start.get(lv))
        win = self.log.grid(st.terminal)
        explain = self.goals.observe_win(lv, start, visited, win)
        lines = []
        if win is None:
            lines.append(f"HX: level {lv + 1} completed by {action_text(st.action)}; the engine returned no separate "
                         "win frame, so no win-state facts are available.")
        else:
            m = match(parse(self.log.grid(st.pre), mask=self.vm.mask), parse(win, mask=self.vm.mask))
            ch = describe_match(m, limit=3)[:5]
            lines.append(f"HX evidence: level {lv + 1} was completed by {action_text(st.action)} after "
                         f"{self.log.actions_on_level(lv)} actions on that level. In the winning step: "
                         + ("; ".join(ch) if ch else "no object-level change outside the masked border") + ".")
            if explain:
                n_prev = max(0, len(visited) - 1)
                cnt = self.goals.last_counter
                lines.append(f"Facts true in the win frame but false at that level's start [number of its {n_prev} "
                             "earlier states where already true]: " + ", ".join(
                                 f"{describe(f)} [{cnt.get(f, 0)}]" for f in explain[:5]) + ".")
        if st.kind == WIN and st.init is not None:
            rep = self.goals.at_level_start(lv + 1, self.log.grid(st.init), limit=4)
            if rep["open"]:
                lines.append(f"On level {lv + 2}'s first frame these candidates are false (heuristic distance): "
                             + ", ".join(f"{describe(f)} ({d:.0f})" for f, d in rep["open"]) + ".")
            if rep["already_true"]:
                lines.append("Already true there, so not sufficient on their own (or the goal changed): "
                             + ", ".join(describe(f) for f in rep["already_true"][:4]) + ".")
            lines.append("These are evidence, not a verified goal: a win supports a goal hypothesis but does not prove it.")
        self.last_win_report = {"level": lv, "lines": lines}
        self._push("level", lines)

    def _hidden_report(self, st, alg):
        k = (self.ctx.key(st.pre), st.action)
        if k in self.conflicts_reported:
            return
        self.conflicts_reported.add(k)
        earlier = [s.t for s in self.log.steps[:-1] if s.kind == MOVE and s.level == st.level
                   and self.ctx.key(s.pre) == k[0] and s.action == st.action]
        self._push("hidden", [
            f"HX evidence: {action_text(st.action)} from the same board (ignoring the masked border) gave a different "
            f"result now than at step(s) {earlier[-3:]} of this level. The outcome depends on something the board "
            "does not show (an earlier action, a counter, a timer, or the masked border). A model of the current "
            "frame alone cannot predict it; include what happened before."])

    def _noop_report(self, st, alg):
        lines = [f"HX evidence: the last {self.noop_streak} actions changed nothing inside the board."]
        rels = [r for r in alg.relations(min_support=2, limit=200) if r["counter"] == 0 and r["rel"] == NOOP]
        rels.sort(key=lambda r: (r["level"], -r["support"]))
        shown = 0
        for r in rels:
            if r["level"] != 0:
                continue
            lines.append(f"- {word_text(r['word'])}: changed nothing in {r['support']} different states of this level, "
                         "never changed anything there.")
            shown += 1
            if shown >= 4:
                break
        eff = Counter()
        for p0, a0, p1, _ in self.log.transitions(level=st.level):
            if self.ctx.key(p0) != self.ctx.key(p1):
                eff[a0 if a0[0] != 6 else (6,)] += 1
        if eff:
            lines.append("Actions that did change the board on this level: " + ", ".join(
                f"{'clicks' if a == (6,) else action_text(a)} x{n}" for a, n in eff.most_common(5)) + ".")
        self._push("noops", lines)

    def take_report(self) -> str:
        if not self.pending:
            return ""
        out = "\n".join(line for _, lines in self.pending for line in lines)
        self.pending.clear()
        return out

    # ------------------------------------------------------------------ sandbox requests
    def request(self, method: str, args: dict) -> dict:
        self.stats["req_" + method] += 1
        if method == "goals":
            items = []
            for f, r in self.goals.ranked(int(args.get("limit", 8))):
                items.append({"fact": describe(f), "levels_explained": [x + 1 for x in r["support"]],
                              "earlier_states_true_without_win": r["counter"]})
            cur = self.log.grid(self.log.current)
            status = self.goals.at_level_start(self.log.level, cur, limit=8) if cur is not None else {"open": [], "already_true": []}
            return {"candidates": items,
                    "false_now": [{"fact": describe(f), "distance": round(d, 1)} for f, d in status["open"]],
                    "true_now": [describe(f) for f in status["already_true"]],
                    "hazards": [describe(f) for f, _ in sorted(self.goals.hazard.items(), key=lambda kv: -kv[1])[:5]],
                    "last_win": self.last_win_report.get("lines", [])}
        if method == "relations":
            alg = self.alg if self.alg_level == self.log.level else None
            if alg is None:
                return {"level": self.log.level + 1, "relations": [], "note": "no transitions on this level yet"}
            rows = []
            for r in alg.relations(min_support=int(args.get("min_support", 2)), limit=int(args.get("limit", 20))):
                rows.append({"relation": r["rel"], "meaning": REL_TEXT.get(r["rel"], r["rel"]),
                             "actions": word_text(r["word"]),
                             "context": ["any state", "same objects present", "same click neighbourhood"][min(2, r["level"])],
                             "states_supporting": r["support"], "counterexamples": r["counter"]})
            return {"level": self.log.level + 1, "relations": rows,
                    "conflicts": len([c for c in alg.conflicts]),
                    "note": "relations are local to their context; one counterexample disables a context"}
        if method == "wins":
            out = []
            for w in self.log.wins():
                out.append({"level": w["level"] + 1, "action": action_text(w["step"].action),
                            "win_grid": None if w["win"] is None else self.log.grid(w["win"]).tolist(),
                            "start_grid": self.log.grid(w["start"]).tolist() if w["start"] is not None else None,
                            "actions_on_level": self.log.actions_on_level(w["level"])})
            return {"wins": out}
        if method == "status":
            return {"level": self.log.level + 1, "steps": len(self.log.steps), "stats": dict(self.stats),
                    "masked_border_cells": int(self.vm.mask.sum())}
        return {"error": f"unknown hx method {method!r}"}
