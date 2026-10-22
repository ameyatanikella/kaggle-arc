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

from ..learning.causal_beliefs import CausalBeliefBank
from ..perception import VolatileMask
from .algebra import COMMUTE, IDEM, INVERSE, INVOL, NOOP, OVERRIDE, Algebra
from .context import GridContext
from .events import DEATH, GAME_WIN, MOVE, RESET, WIN, EventLog
from .goal_ledger import GoalLedger
from .goal_ledger import describe as describe_h
from .goals import GoalModel, describe
from .history import HistorySplitter, HistoryTracker, UndoAllModel, UndoLastModel, UndoModel
from .ledger import ForecastLedger
from .objects import describe_match, match, parse
from .scene import camera_shift

NOOP_STREAK = 4          # consecutive interior no-ops (across calls) before the noops report
MAX_REPORT_LINES = 9
REL_TEXT = {NOOP: "changed nothing", IDEM: "a second press changed nothing more", INVOL: "a second press undid it",
            INVERSE: "the second undid the first", COMMUTE: "order did not matter",
            OVERRIDE: "the second erased the first's effect"}


class _PrecommittedModel:
    """Adapter feeding sandbox-committed 1-step forecasts into CausalBeliefBank."""

    def __init__(self, name: str, claims_by_action: dict):
        self.name = name
        self._claims = claims_by_action

    def predict(self, before, action, history):
        claim = self._claims.get(str(action))
        return dict(claim) if isinstance(claim, dict) else None


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


def action_cls(a: tuple) -> str:
    return "click" if a[0] == 6 else action_text(a)


class HXSession:
    def __init__(self, triggers: bool = True, ledger: bool = False, stall: bool = False,
                 stall_neutral: bool = False, stall_turns: int = 5, stall_tokens: int = 20000):
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
        self.use_ledger = ledger
        if ledger:                       # HX2: per-game hypothesis ledger (docs/prereg_hx2.md)
            self.fl = ForecastLedger()
            self.gl = GoalLedger(mask_fn=lambda: self.vm.mask)
            self.hist = HistoryTracker()
            self.split = HistorySplitter()
            self.undo = UndoModel()
            self.undo1 = UndoLastModel()
            self.undo_all = UndoAllModel()
            self.camera: list = []
            self.goal_reports: Counter = Counter()
            self.camera_reported: set = set()
        # Stage 1: switchable in-loop stall intervention & cost-matched neutral control
        self.use_stall = bool(stall)
        self.use_stall_neutral = bool(stall_neutral)
        self.stall_turns = max(1, int(stall_turns))
        self.stall_tokens = max(1000, int(stall_tokens))
        self.stall_level = -1
        self.turns_on_level = 0
        self.next_stall_turn = self.stall_turns
        self.next_stall_tokens = self.stall_tokens
        self.stall_active_turn = False
        self.stall_reminded_this_turn = False
        self.stall_invoked_this_turn = False
        self.stall_triggered_levels: set[int] = set()
        self.stall_changed_levels: set[int] = set()
        self.last_probe_record: Optional[dict] = None
        self.active_plan_on_level: Optional[dict] = None
        self.intervention_log: list[dict] = []

    # ------------------------------------------------------------------ recording
    def start(self, grid, level: int = 0):
        self.log.start(np.asarray(grid, dtype=np.uint8), level)
        self.started = True
        self._reset_stall_level(level)
        if self.use_ledger:
            self.gl.start_level(level, self.log.grid(self.log.current), self.ctx.key(self.log.current))

    def _reset_stall_level(self, level: int):
        self.stall_level = int(level)
        self.turns_on_level = 0
        self.next_stall_turn = self.stall_turns
        self.next_stall_tokens = self.stall_tokens
        self.stall_active_turn = False
        self.stall_reminded_this_turn = False
        self.stall_invoked_this_turn = False
        self.last_probe_record = None
        self.active_plan_on_level = None

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
            fresh = self.alg is None or self.alg_level != st.level or grown
            forecasts = None
            if self.use_ledger and not fresh:
                forecasts = self._forecasts(k0, st.action)          # from the pre-state only
            alg = self._algebra(st.level, grown)
            if not grown:
                info = alg.observe(k0, st.action, k1)
            else:
                info = {"conflict": False}          # the rebuild already contains this step
                alg.observe(k0, st.action, k1)
            split_info = None
            if self.use_ledger:
                split_info = self._ledger_move(st, g0, g1, k0, k1, forecasts)
            if info.get("conflict"):
                self.stats["conflicts"] += 1
                self._hidden_report(st, alg, split_info)
            if k0 == k1:
                self.noop_streak += 1
                if self.noop_streak >= NOOP_STREAK and not self.noop_fired:
                    self._noop_report(st, alg)
                    self.noop_fired = True
            else:
                self.noop_streak, self.noop_fired = 0, False
        elif st.kind in (WIN, GAME_WIN):
            self.noop_streak, self.noop_fired = 0, False
            if st.level in self.stall_triggered_levels:
                self.stats["stall_levels_won"] += 1
            if (self.active_plan_on_level is not None
                    and self.active_plan_on_level.get("level") == st.level
                    and self.active_plan_on_level.get("following")):
                self.stats["levels_won_via_executed_plan"] += 1
            if self.use_ledger:
                self._ledger_win(st)
            else:
                self._win_report(st)
            if st.kind == WIN:
                self._reset_stall_level(st.level + 1)
        elif st.kind == DEATH:
            self.noop_streak, self.noop_fired = 0, False
            if st.terminal is not None:
                self.goals.observe_death(self.log.grid(st.pre), self.log.grid(st.terminal))
        elif st.kind == RESET:
            self.noop_streak, self.noop_fired = 0, False
            if self.use_ledger:
                self.hist.reset_level()
                self.undo.reset_level()
                self.undo1.reset_level()
                self.undo_all.reset_level()
                if st.post is not None:
                    self.gl.observe_state(self.ctx.key(st.post), self.log.grid(st.post))
        return out

    # ------------------------------------------------------------------ HX2 ledger
    def _forecasts(self, k0, a) -> dict:
        alg = self.alg
        known = alg.succ(k0, a)
        feats = self.hist.features(a)
        self._feats = feats
        hist_dep = a[0] == UndoModel.UNDO
        return {"graph": known,
                "algebra": None if (known is not None or hist_dep) else (alg.predict(k0, a) or (None,))[0],
                "history": self.split.predict(k0, a, feats),
                "undo": self.undo.predict(k0, a),
                "undo1": self.undo1.predict(k0, a),
                "undo_all": self.undo_all.predict(k0, a)}

    def _ledger_move(self, st, g0, g1, k0, k1, forecasts):
        a = st.action
        feats = getattr(self, "_feats", None) or self.hist.features(a)
        self._feats = None
        sp = self.split.observe(k0, a, feats, k1)
        cam = None
        if float((g0 != g1).mean()) > 0.05:
            cam = camera_shift(g0, g1, self.vm.mask)
            if cam is not None:
                self.camera.append((st.level, st.t, cam[0], cam[1]))
                self.stats["camera_shifts"] += 1
                if st.level not in self.camera_reported:
                    self.camera_reported.add(st.level)
                    self._push("camera", [
                        f"HX evidence: after {action_text(a)} the whole view shifted by {cam[0]} rows and {cam[1]} "
                        f"columns ({int(cam[2] * 100)}% of the board matches the shifted previous frame). Positions "
                        "in frames are SCREEN coordinates: the same tile is now at (row + dy, col + dx). Track "
                        "positions in world coordinates and convert back before you click."])
        if forecasts is not None:
            cause = "binding" if cam is not None else ("latent" if sp.get("conflict") else "transition")
            self.fl.score(k0, action_cls(a), forecasts, k1, cause)
        self.undo.observe(k0, a, k1)
        self.undo1.observe(k0, a, k1)
        self.undo_all.observe(k0, a, k1)
        self.hist.update(a, k0 != k1)
        refuted = self.gl.observe_state(k1, g1)
        if refuted:
            self.stats["goal_refutations"] += len(refuted)
            if self.goal_reports[st.level] < 2:
                self.goal_reports[st.level] += 1
                self._push("goal", [
                    "HX evidence: these goal hypotheses from earlier wins now hold on this level WITHOUT a level-up, "
                    "so none of them is sufficient on its own (each may still be a prerequisite): "
                    + "; ".join(describe_h(h) for h in refuted[:4]) + "."])
        return sp

    def _ledger_win(self, st):
        lv = st.level
        win = self.log.grid(st.terminal)
        new = self.gl.observe_win(lv, win)
        lines = []
        if win is None:
            lines.append(f"HX: level {lv + 1} completed by {action_text(st.action)}; no separate win frame was "
                         "returned, so no win-state evidence is available.")
        else:
            m = match(parse(self.log.grid(st.pre), mask=self.vm.mask), parse(win, mask=self.vm.mask))
            ch = describe_match(m, limit=3)[:5]
            lines.append(f"HX evidence: level {lv + 1} was completed by {action_text(st.action)} after "
                         f"{self.log.actions_on_level(lv)} actions on that level. In the winning step: "
                         + ("; ".join(ch) if ch else "no object-level change outside the masked border") + ".")
        suff = self.gl.ranked("sufficient?", 4)
        if suff:
            lines.append("Goal hypotheses never yet seen without a level-up (candidates for 'sufficient'; "
                         "[levels explained]): " + "; ".join(f"{describe_h(h)} [{len(r['support'])}]" for h, r in suff) + ".")
        pre = [h for h, r in self.gl.ranked("prerequisite", 40) if lv in r["support"]][:3]
        if pre:
            lines.append("True at this win but also earlier without a win (at most prerequisites): "
                         + "; ".join(describe_h(h) for h in pre) + ".")
        if st.kind == WIN and st.init is not None:
            self.gl.start_level(lv + 1, self.log.grid(st.init), self.ctx.key(st.init))
            open_ = [(h, self.gl.distance(h, self.log.grid(st.init))) for h, _ in self.gl.ranked("sufficient?", 4)]
            if open_:
                lines.append(f"On level {lv + 2}'s first frame (heuristic distance): "
                             + "; ".join(f"{describe_h(h)} ({d:.0f})" for h, d in open_) + ".")
            lines.append("A win supports a goal hypothesis but does not prove it; a hypothesis that holds without a "
                         "level-up is refuted as sufficient.")
        self.hist.reset_level()
        self.undo.reset_level()
        self.undo1.reset_level()
        self.undo_all.reset_level()
        self.last_win_report = {"level": lv, "lines": lines}
        self._push("level", lines)

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

    def _hidden_report(self, st, alg, split_info=None):
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
            "frame alone cannot predict it; include what happened before."
            + ((" So far the outcome is consistent with depending on " + self.split.explain(k[0], st.action) + ".")
               if self.use_ledger and split_info and split_info.get("feature") else "")])

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

    # ------------------------------------------------------------------ Stage 1 stall turn
    def on_turn_start(self, level: int, actions_on_level: int = 0, tokens_on_level: int = 0) -> str:
        """Called at the start of each analyzer turn. Fires the Stage 1 stall intervention
        (or cost-matched neutral control) when the level exceeds the registered stall threshold."""
        if not (self.use_stall or self.use_stall_neutral):
            return ""
        lv = int(level)
        if lv != self.stall_level:
            self._reset_stall_level(lv)
        self.turns_on_level += 1
        # If the previous turn was a stall turn and B0 never called hx_probe or action, close it out
        if self.stall_active_turn and not self.stall_invoked_this_turn:
            self.stats["stall_proposal_failures"] += 1
        self.stall_active_turn = False
        self.stall_reminded_this_turn = False
        self.stall_invoked_this_turn = False

        tok = int(tokens_on_level)
        acts = int(actions_on_level)
        level_trans = list(self.log.transitions(level=self.log.level))
        n_moves = len(level_trans)
        n_effective = sum(1 for p0, _, p1, _ in level_trans if self.ctx.key(p0) != self.ctx.key(p1))
        # A causal transition check requires non-trivial observed motion on the current level.
        # If no move has changed the interior board yet (rank-0 design), let B0 explore first.
        if n_moves == 0 or n_effective == 0:
            return ""
        fire_turn = self.turns_on_level >= self.next_stall_turn
        fire_tok = (tok >= self.next_stall_tokens) and (n_effective >= 2)
        if not (fire_turn or fire_tok):
            return ""
        self.next_stall_turn = self.turns_on_level + self.stall_turns
        self.next_stall_tokens = max(tok, self.next_stall_tokens) + self.stall_tokens
        self.stall_triggered_levels.add(lv)
        self.stats["stall_triggers"] += 1
        self.stats["stall_triggered_levels"] = len(self.stall_triggered_levels)

        tok_k = max(1, tok // 1000) if tok else 0
        lv_disp = lv if lv >= 1 else self.log.level + 1
        if self.use_stall:
            self.stall_active_turn = True
            msg = (
                f"HX stall check (level {lv_disp}, {self.turns_on_level} turns / {acts} actions / "
                f"~{tok_k}k tokens on this level without a win; {n_moves} same-level moves in `moves`, "
                f"{n_effective} changed the interior board):\n"
                "Before executing `action(...)` on this turn, test two competing hypotheses in Python with `hx_probe` "
                "(call the `python` tool directly without long prose speculation):\n"
                "1. Define `state_of(frame)` returning a dict or tuple of the board features that matter.\n"
                "2. Define at least two candidate transition functions `step1(s, a)`, `step2(s, a)` and goal predicates `goal1(s)`, `goal2(s)`.\n"
                "3. Run `res = hx_probe({'m1': step1, 'm2': step2}, {'g1': goal1, 'g2': goal2}, state_of=state_of)` and print `res`.\n"
                "`hx_probe` checks models against `moves`, excludes non-winning goals and static/refuted models, "
                "reports `empirical_invariants` (exact per-action deltas & relations) and `state_diagnostics` "
                "(Markov conflicts or blind features), and returns `'decision'`, `'action'`, and `'probe'`/`'plan'`. "
                "If it returns `'none'`, use `empirical_invariants`/`state_diagnostics` to repair the model or run "
                "`distinguishing_probe`/`frontier_probe`."
            )
        else:
            msg = (
                f"Progress review (level {lv_disp}, {self.turns_on_level} turns / {acts} actions / "
                f"~{tok_k}k tokens on this level without a win; {n_moves} same-level moves in `moves`, "
                f"{n_effective} changed the interior board):\n"
                "Before executing `action(...)` on this turn, review your current progress in Python:\n"
                "1. Inspect the current board state and summarize the objects and coordinates that have changed so far.\n"
                "2. Consider whether your current approach has made measurable progress toward clearing the level or is repeating states.\n"
                "3. Check the available actions and recent transitions in Python and print a short status summary of what you observe.\n"
                "After printing your status summary from the current board state, choose and execute your next action sequence as usual."
            )
        self.stats["stall_prompt_chars"] += len(msg)
        return msg

    # ------------------------------------------------------------------ sandbox requests
    def request(self, method: str, args: dict) -> dict:
        self.stats["req_" + method] += 1
        if method == "before_action":
            if (self.use_stall and self.stall_active_turn
                    and not self.stall_invoked_this_turn
                    and not self.stall_reminded_this_turn):
                self.stall_reminded_this_turn = True
                self.stats["stall_reminders"] += 1
                return {
                    "require_probe": True,
                    "message": (
                        "Stall check active on this turn (0 actions spent): before calling action(...), "
                        "call hx_probe(models, goals, state_of=state_of) with at least two candidate "
                        "step(state, action) models and goal(state) predicates to test against `moves`. "
                        "Calling action(...) after hx_probe (or on your next tool step) will execute immediately."
                    ),
                }
            if (self.use_stall and self.stall_active_turn
                    and not self.stall_invoked_this_turn
                    and self.stall_reminded_this_turn):
                self.stall_active_turn = False
                self.stats["stall_proposal_failures"] += 1
            return {"require_probe": False}

        if method == "probe_logged":
            self.stats["probe_calls"] += 1
            if self.stall_active_turn or (self.stall_level in self.stall_triggered_levels):
                if not self.stall_invoked_this_turn:
                    self.stall_invoked_this_turn = True
                    self.stats["stall_invocations"] += 1
                self.stall_active_turn = False
            n_valid = int(args.get("valid_models") or 0)
            sole_ok = bool(args.get("single_supported_model"))
            if n_valid >= 2 or sole_ok:
                self.stats["valid_model_pairs"] += 1
            else:
                self.stats["stall_proposal_failures"] += 1
            self.last_probe_record = dict(args, level=self.log.level)
            return {"ok": True}

        if method == "action_logged":
            actual_actions = [str(a) for a in (args.get("actions") or [])]
            before_obs = args.get("before_obs")
            after_obs = args.get("after_obs")
            prev_act = args.get("prev_action")
            rec = self.last_probe_record
            if rec is not None and rec.get("level") == self.log.level and actual_actions:
                proposed_action = rec.get("action")
                proposed_plan = rec.get("plan") or rec.get("probe") or ([] if proposed_action is None else [proposed_action])
                followed = bool(proposed_action is not None and str(actual_actions[0]) == str(proposed_action))
                # A followed recommendation that departs from the preceding action is an
                # observable redirection proxy, not a counterfactual score comparison.
                changed = bool(followed and prev_act is not None
                               and str(actual_actions[0]) != str(prev_act))
                self.stats["probe_actions_evaluated"] += 1
                if followed:
                    self.stats["probe_actions_followed"] += 1
                    self.active_plan_on_level = {"level": self.log.level, "plan": list(proposed_plan), "following": True}
                if changed:
                    self.stats["stall_actions_changed"] += 1
                    self.stall_changed_levels.add(self.log.level)
                    self.stats["stall_changed_levels"] = len(self.stall_changed_levels)

                weights_after = {}
                model_outcomes = {}
                forecasts = rec.get("forecasts_from_s0") or {}
                # Sandbox forecasts describe one step. A batched action() call returns only
                # the final observation, so grading it as the first step would be false.
                one_step_observed = len(actual_actions) == 1
                if one_step_observed and isinstance(before_obs, dict) and isinstance(after_obs, dict) and forecasts:
                    try:
                        models = [_PrecommittedModel(m, fc) for m, fc in forecasts.items() if isinstance(fc, dict)]
                        if models:
                            priors = {m.name: max(1e-6, float((rec.get("weights") or {}).get(m.name, 1.0))) for m in models}
                            bank = CausalBeliefBank(models, priors=priors)
                            bank.commit(before_obs, str(actual_actions[0]))
                            bank.observe(after_obs)
                            weights_after = {k: round(float(v), 4) for k, v in bank.weights().items()}
                            for m_name, ev in bank.evidence.items():
                                model_outcomes[m_name] = {
                                    "correct": ev.correct,
                                    "wrong": ev.wrong,
                                    "vacuous": ev.vacuous,
                                    "abstained": ev.abstained,
                                    "event_coverage": ev.event_coverage,
                                }
                            self.stats["belief_updates"] += 1
                            self.stats["belief_correct_fields"] += sum(ev.correct for ev in bank.evidence.values())
                            self.stats["belief_wrong_fields"] += sum(ev.wrong for ev in bank.evidence.values())
                    except Exception:
                        pass
                if len(self.intervention_log) < 200:
                    self.intervention_log.append({
                        "level": self.log.level + 1,
                        "decision": rec.get("decision"),
                        "proposed_action": proposed_action,
                        "proposed_plan": proposed_plan,
                        "actual_actions": actual_actions[:5],
                        "followed_tool": followed,
                        "action_changed": changed,
                        "valid_models": rec.get("valid_models", 0),
                        "weights_before": rec.get("weights"),
                        "weights_after": weights_after,
                        "model_outcomes": model_outcomes,
                        "belief_update_skipped": "batched_actions" if not one_step_observed else None,
                    })
                self.last_probe_record = None
            return {"ok": True}

        if method == "interventions":
            return {"interventions": list(self.intervention_log), "stats": dict(self.stats)}
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
        if method == "ledger" and self.use_ledger:
            gl = {"sufficient_candidates": [{"hypothesis": describe_h(h), "levels_explained": [x + 1 for x in r["support"]],
                                             "later_wins_where_false": r["win_false"]}
                                            for h, r in self.gl.ranked("sufficient?", int(args.get("limit", 8)))],
                  "prerequisites": [{"hypothesis": describe_h(h), "held_without_win": r["nonwin"],
                                     "levels_explained": [x + 1 for x in r["support"]]}
                                    for h, r in self.gl.ranked("prerequisite", int(args.get("limit", 8)))],
                  "counts": dict(self.gl.n)}
            splits = [{"state": k.hex()[:8], "action": action_text(a), "explanation": self.split.explain(k, a)}
                      for (k, a) in list(self.split.split)[-6:]]
            return {"forecasts": self.fl.summary(), "goals": gl, "hidden_state": {**self.split.counts, "examples": splits},
                    "camera_shifts": [{"level": lv + 1, "dy": dy, "dx": dx} for lv, t, dy, dx in self.camera[-5:]]}
        if method == "status":
            return {"level": self.log.level + 1, "steps": len(self.log.steps), "stats": dict(self.stats),
                    "masked_border_cells": int(self.vm.mask.sum())}
        return {"error": f"unknown hx method {method!r}"}
