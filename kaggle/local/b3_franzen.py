"""B3 + Franzen reasoning integration (docs/prereg_b3_franzen.md).

One game, one authoritative history, one budget. The Franzen/Duck harness (ToolAgent: prompts,
Python sandbox, retained functions, guards, history) is used UNCHANGED as the reasoning layer.
B3 is attached at the harness's only engine boundary:

  * _HarnessGameSession._execute_action is wrapped. Every action (the model's own action() calls,
    B3's autonomous actions, B3 actions requested by the model, automatic resets) passes through
    it. B3 observes the outcome, checks its prediction (a contradiction is recorded), and the
    action is attributed to its source.
  * The session's turn loop becomes B3-first. B3 acts autonomously (frontier exploration, cached
    plans, which continue through expected unchanged steps) with NO LLM call per action. An LLM
    turn (one ToolAgent.analyze call, exactly as Franzen's loop makes it) is invoked on a
    registered trigger: level start; no reachable unexplored state; contradiction; autonomy
    allowance; delegation finished; repeated guard refusals.
  * A `b3(method, **args)` channel is added to the sandbox. The model's Python can query B3's
    model, frontier, candidates and planner, delegate autonomous execution, and revise B3's
    representation. B3 actions requested from inside a snippet (`b3_run`) execute through the
    snippet's normal action() path, so every harness guard and history update applies.
  * Inference failure never ends the game: after repeated analyzer failures, B3 continues
    autonomously and the LLM is retried later. Only WIN, the action cap or the wall-time budget
    ends a game.

Arms: --arm b3 (frozen B3, no LLM), b3_franzen (integrated), franzen (unchanged harness).
Run with the harness venv, e.g.
  build/b0src/ARC3-Inference/.venv/bin/python kaggle/local/b3_franzen.py --arm b3_franzen \\
      --games ls20 --minutes 20 --base-url http://localhost:11434/v1 --model qwen3.5:35b --out runs/b3f/x
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kaggle" / "local"))

B3_CONFIG = {"plan_cache": "projected", "noop_order": True, "reset_purpose": True}   # frozen baseline
AUTONOMY_ALLOWANCE = 300        # autonomous actions per level between LLM turns without a level-up
CONTRADICTION_MIN_GAP = 50      # autonomous actions between contradiction-triggered turns
REFUSAL_LIMIT = 20
FAILURE_BACKOFF_ACTIONS = 200   # autonomous actions before retrying the LLM after repeated failures
MAX_FAILURES = 3

B3_TOOL_DOC = """- An executor called B3 shares this game with you. It explores autonomously between your turns: it keeps a graph of observed transitions, tries untried actions, follows cached plans, and resets the level only when its recorded graph shows a purpose. It hands control to you when a level starts, when it has no reachable unexplored state, when its model is contradicted, after 300 autonomous actions without level progress, or when actions you delegated are done. Every action, yours or B3's, appears in `history`; the turn opener says why you were called and what B3 did.
- B3 tools in the Python sandbox. Each returns a dict with exactly these keys:
  - `b3("status")` -> {'level', 'state_id', 'states_known', 'frontier_states', 'untried_here', 'plan_to_frontier_len', 'historically_unchanged', 'recent_contradictions', 'actions_by_source', 'trigger'}
  - `b3("candidates")` -> {'untried': list of actions, 'tried_count'}: actions B3 has not yet tried from the current state (action dicts usable with action()).
  - `b3("transitions")` -> {'from_here': list of {'action', 'to_state', 'changed'}}: what each tried action did from this state.
  - `b3("state_frame", state_id=...)` -> {'state_id', 'rows'}: the stored board of a known state (rows of hex colour digits).
  - `b3("plan_to_frontier")` and `b3("plan_to_state", state_id=...)` -> {'actions': list or None, 'length', 'reason'}: shortest known action sequence in B3's graph (execute it with action()).
  - `b3_run(n=50, stop_on_change=False)` -> {'executed', 'stopped'}: let B3 choose and execute up to n actions now, inside this snippet.
  - `b3("delegate", n=...)` -> {'autonomous_actions'}: after this turn, B3 acts on its own for up to n actions before calling you again.
  - `b3("revise", mask_rows=[...], mask_cols=[...], extra_clicks=[[row, col], ...], undefer=True)` -> {'applied'}: change B3's representation when it cannot explain the evidence (ignore rows/columns for state identity, add click targets, re-enable actions it deprioritised as historically unchanged).
- B3 cannot judge goals. Use your analysis to decide what to aim for, then use B3's graph and planner to get there cheaply."""

_HOST = threading.local()          # the current B3 session of this thread
_INSTALLED = {"done": False}
STATS = Counter()

_CHILD_ANCHOR = '    runtime_globals["action"] = action'
_CHILD_SRC = '''
    def b3(method, **kwargs):
        _send({"type": "b3", "request": {"method": str(method), "args": kwargs}})
        reply = _recv()
        if reply.get("type") != "b3_result":
            raise RuntimeError("Invalid b3 response from sandbox host.")
        res = reply.get("result") or {}
        if isinstance(res, dict) and res.get("error"):
            raise RuntimeError("b3(" + repr(method) + "): " + str(res["error"]))
        return res

    def b3_run(n=50, stop_on_change=False):
        done = []
        for _ in range(int(n)):
            nxt = b3("next")
            a = nxt.get("action")
            if a is None or a == "RESET":
                return {"executed": done, "stopped": nxt.get("why") or "B3 has no action here"}
            try:
                r = action([a])
            except Exception as exc:
                b3("blocked", action=a, error=type(exc).__name__ + ": " + str(exc))
                return {"executed": done, "stopped": type(exc).__name__ + ": " + str(exc)}
            done.append(a)
            for flag in ("level_completed", "game_over", "run_complete", "done"):
                if r.get(flag):
                    return {"executed": done, "stopped": flag}
            if stop_on_change and r.get("gameplay_changed"):
                return {"executed": done, "stopped": "gameplay_changed"}
        return {"executed": done, "stopped": "n reached"}

    runtime_globals["b3"] = b3
    runtime_globals["b3_run"] = b3_run'''

_HOST_ANCHOR = '            if msg_type == "action":\n'
_HOST_SRC = '''            if msg_type == "b3":
                _h = getattr(_B3_HOST, "session", None)
                try:
                    _rep = _h.b3_request(dict(message.get("request") or {})) if _h is not None else {
                        "error": "B3 tools are not available in this session."}
                except Exception as _exc:  # noqa: BLE001
                    _rep = {"error": type(_exc).__name__ + ": " + str(_exc)}
                _send_json_line(process.stdin, {"type": "b3_result", "result": _rep})
                continue
'''


def install():
    """Patch the harness at runtime (no file is edited). Idempotent. Not used by the 'franzen' arm."""
    if _INSTALLED["done"]:
        return
    import inference.agent.python_tool_sandbox as S
    import inference.agent.tool_agent as T
    b = S._SANDBOX_BOOTSTRAP
    assert b.count(_CHILD_ANCHOR) == 1, "sandbox bootstrap changed"
    S._SANDBOX_BOOTSTRAP = b.replace(_CHILD_ANCHOR, _CHILD_ANCHOR + _CHILD_SRC, 1)
    src = inspect.getsource(S.run_sandboxed_python)
    assert src.count(_HOST_ANCHOR) == 1, "sandbox host loop changed"
    src = src.replace(_HOST_ANCHOR, _HOST_SRC + _HOST_ANCHOR, 1)
    S._B3_HOST = _HOST
    ns = S.__dict__
    exec(compile(src, S.__file__, "exec"), ns)          # redefines run_sandboxed_python in the module
    T.run_sandboxed_python = S.run_sandboxed_python      # tool_agent imported it by name
    inner_sp = T._build_system_prompt

    def _build_system_prompt(*args, **kwargs):
        p = inner_sp(*args, **kwargs)
        anchor = "\n\nTool session rules:\n"
        if B3_TOOL_DOC not in p:
            p = p.replace(anchor, "\n" + B3_TOOL_DOC + anchor, 1) if anchor in p else p + "\n" + B3_TOOL_DOC
        return p
    T._build_system_prompt = _build_system_prompt
    A = T.ToolAgent
    inner_up = A._build_user_prompt

    def _build_user_prompt(self, action_num, *args, **kwargs):
        out = inner_up(self, action_num, *args, **kwargs)
        sess = getattr(_HOST, "session", None)
        if sess is not None:
            try:
                out += "\n\n" + sess.turn_report()
            except Exception as exc:  # noqa: BLE001
                STATS["report_errors"] += 1
                out += f"\n\nB3 report unavailable: {type(exc).__name__}"
        return out
    for f in dir(inner_up):
        if f.endswith("_wrapped"):
            setattr(_build_user_prompt, f, getattr(inner_up, f))
    A._build_user_prompt = _build_user_prompt
    _INSTALLED["done"] = True


def _sid(node) -> str:
    return hashlib.blake2b(node, digest_size=4).hexdigest() if isinstance(node, (bytes, bytearray)) else str(node)


def make_session_class():
    import arcengine
    from inference.agent.action_names import to_model_action
    from inference.framework import solver as SV
    from arc3.agents.b3.agent import B3Agent
    from arc3.env import Observation

    class B3Session(SV._HarnessGameSession):
        # attributes set by the solver after construction: b3_arm, b3_seed
        def _b3_setup(self, arm: str, seed: int):
            self.b3_arm = arm
            self.b3 = B3Agent(self.game.game_run.game_id if self.game.game_run else "game", dict(B3_CONFIG), seed=seed)
            self.b3_source = None
            self.b3_pending_tool = None
            self.b3_level_actions = 0
            self.b3_level = None
            self.b3_trigger = "game start" if arm == "b3_franzen" else None
            self.b3_auto_since_turn = 0
            self.b3_auto_level = 0
            self.b3_delegated = 0
            self.b3_refusals = 0
            self.b3_llm_blocked_until = 0
            self.b3_contradictions: list = []
            self.b3_last_contra_at = -10 ** 9
            self.b3_by_source = Counter()
            self.b3_level_sources: list = []          # (level, source of the completing action)
            self.b3_since_turn = Counter()
            self.b3_revisions: list = []
            self.b3_t = Counter()                      # wall time: llm_turns, b3_decide, engine
            self.b3_turns = 0
            self.b3_turn_triggers = Counter()
            self.b3_failures = 0
            self.b3_last_obs = None

        # ---------------------------------------------------------------- observations
        def _b3_obs(self):
            st = self.game.current_state
            raw = st.raw.state
            return Observation(frames=(np.array(SV._grid_from_state(st), dtype=np.uint8),),
                               state=getattr(raw, "name", str(raw)), levels_completed=int(st.levels_completed),
                               win_levels=int(self.game.number_of_levels),
                               available_actions=tuple(int(a) for a in (st.available_actions or ())),
                               actions_used=self.action_count, level_actions_used=self.b3_level_actions)

        @staticmethod
        def _tuple(action):
            if action.id == arcengine.GameAction.ACTION6:
                return (6, int(action.data.get("x", 0)), int(action.data.get("y", 0)))
            return (int(action.id.value),)

        @staticmethod
        def _harness(choice):
            if choice[0] == 6:
                return {"action": "MOUSE", "row": int(choice[2]), "col": int(choice[1])}
            return {"action": to_model_action(arcengine.GameAction.from_id(choice[0]).name)}

        def _execute_action(self, action, **kwargs):
            choice = self._tuple(action)
            src = self.b3_source or "llm"
            if src == "llm" and self.b3_pending_tool is not None and choice == self.b3_pending_tool:
                src = "b3_tool"                        # B3's choice, requested by the model's b3_run
            if src not in ("b3_auto", "b3_tool"):
                self.b3.note_external(choice if action.id != arcengine.GameAction.RESET else (0,))
            self.b3_pending_tool = None
            t0 = time.perf_counter()
            payload = super()._execute_action(action, **kwargs)
            self.b3_t["engine_and_harness"] += time.perf_counter() - t0
            lv_before = self.b3_level
            obs = self._b3_obs()
            if action.id == arcengine.GameAction.RESET:
                self.b3_level_actions = 0 if obs.levels_completed != lv_before else self.b3_level_actions + 1
            else:
                self.b3_level_actions += 1
            info = self.b3.observe(obs)
            self.b3_last_obs = obs
            self.b3_by_source[src] += 1
            self.b3_since_turn[src] += 1
            if lv_before is not None and obs.levels_completed > lv_before:
                self.b3_level_sources.append((lv_before + 1, src))
                self.b3_level_actions = 0
                self.b3_auto_level = 0
                self._set_trigger(f"level {obs.levels_completed + 1} started (the previous level was completed by {src})")
            self.b3_level = obs.levels_completed
            if info.get("contradiction"):
                self.b3_contradictions.append({"state_id": _sid(info.get("node")), "action": self._harness(choice),
                                               "source": src})
                self.b3_contradictions = self.b3_contradictions[-10:]
                if src == "b3_auto" and self.action_count - self.b3_last_contra_at >= CONTRADICTION_MIN_GAP:
                    self.b3_last_contra_at = self.action_count
                    self._set_trigger("B3's model was contradicted: a known transition produced a different state")
            return payload

        def _execute_auto_reset(self):
            self.b3_source = "auto_reset"
            try:
                super()._execute_auto_reset()
            finally:
                self.b3_source = None

        def _set_trigger(self, why):
            if self.b3_arm == "b3_franzen" and self.b3_trigger is None:
                self.b3_trigger = why

        # ---------------------------------------------------------------- B3 autonomy
        def _b3_step(self):
            obs = self.b3_last_obs or self._b3_obs()
            t0 = time.perf_counter()
            act = self.b3.decide(obs)
            self.b3_t["b3_decide"] += time.perf_counter() - t0
            if act.why == "fallback":
                self._set_trigger("B3 has no reachable unexplored state in its recorded graph")
            choice = (act.action,) if act.action != 6 else (6, act.x, act.y)
            self.b3_source = "b3_reset" if act.action == 0 else "b3_auto"
            try:
                if act.action == 0:
                    self._execute_action(arcengine.ActionInput(id=arcengine.GameAction.RESET, data={}),
                                         batch_index=1, batch_size=1)
                    self.last_engine_action = "RESET"
                    return
                payload = self.step_env({"actions": [self._harness(choice)]})
            finally:
                self.b3_source = None
            if not payload.get("executed"):
                node = self.b3._obs_node
                self.b3.graph.tried.setdefault(node, set()).add(choice)
                self.b3.prev = None
                self.b3_refusals += 1
                if self.b3_refusals >= REFUSAL_LIMIT:
                    self.b3_refusals = 0
                    self._set_trigger("the harness guards refused B3's actions repeatedly")
                return
            self.b3_refusals = 0
            self.b3_auto_since_turn += 1
            self.b3_auto_level += 1
            if self.b3_delegated > 0:
                self.b3_delegated -= 1
                if self.b3_delegated == 0:
                    self._set_trigger("the autonomous actions you delegated are done")
            elif self.b3_auto_since_turn >= AUTONOMY_ALLOWANCE:
                self._set_trigger(f"{AUTONOMY_ALLOWANCE} autonomous B3 actions on this level without progress")

        # ---------------------------------------------------------------- one LLM turn
        def _llm_turn(self, retry_step):
            if retry_step is None:
                self.analysis_step += 1
                step = self.analysis_step
            else:
                step = retry_step
            self.write_runtime_state()
            before = self._read_transcript_bytes()
            t0 = time.perf_counter()
            try:
                result = self.analyzer.analyze(
                    self.state_path, self.action_count, valid_actions=SV._engine_action_names(self.game),
                    step_env=self.step_env, transcript_path=self.transcript_path, analysis_step=step,
                    request_timeout_seconds=self.request_timeout_seconds(), should_stop=self.should_stop)
            finally:
                self.b3_t["llm_turns"] += time.perf_counter() - t0
                delta = self._transcript_delta_since(before)
                if delta.strip():
                    self._append_analysis_viewer_event(step, delta)
                    self.write_viewer_payload()
            return result, step

        def turn_report(self) -> str:
            src = ", ".join(f"{k} {v}" for k, v in sorted(self.b3_since_turn.items())) or "none"
            node = self.b3._obs_node
            fr = sum(1 for n in self.b3.graph.edges if self.b3._untried(n, self.b3_last_obs.available_actions)) \
                if self.b3_last_obs else 0
            return (f"B3 executor report: you were called because {self.b3_last_trigger}. "
                    f"Actions since your last turn by source: {src}. Current state {_sid(node)}; "
                    f"states known on this level {len(self.b3.graph.edges)}, with untried actions {fr}; "
                    f"recent contradictions {len(self.b3_contradictions)}. Use b3(...) tools to query or delegate.")

        # ---------------------------------------------------------------- B3-first loop
        def _play_inner(self):
            run = self.game.game_run
            _HOST.session = self
            self.b3_level = int(self.game.current_state.levels_completed)
            self.b3_last_obs = self._b3_obs()
            self.b3.observe(self.b3_last_obs)
            self.b3_last_trigger = "the game started"
            retry_step = None
            try:
                while not self.should_stop():
                    if SV._is_engine_game_over(self.game) and self.last_engine_action != "RESET":
                        self._execute_auto_reset()
                        continue
                    llm_ok = self.b3_arm == "b3_franzen" and self.action_count >= self.b3_llm_blocked_until
                    if llm_ok and (self.b3_trigger is not None or retry_step is not None) and self.b3_delegated == 0:
                        if retry_step is None:
                            self.b3_last_trigger = self.b3_trigger or self.b3_last_trigger
                            self.b3_turn_triggers[self.b3_last_trigger.split(":")[0][:60]] += 1
                        self.b3_trigger = None
                        result, step = self._llm_turn(retry_step)
                        self.b3_turns += 1
                        if result is None or getattr(result, "retryable_failure", False):
                            self.b3_failures += 1
                            retry_step = step
                            if self.b3_failures >= MAX_FAILURES:
                                # inference exhaustion: B3 continues; the LLM is retried later
                                self.b3_failures, retry_step = 0, None
                                self.b3_llm_blocked_until = self.action_count + FAILURE_BACKOFF_ACTIONS
                                self.b3_trigger = "the reasoning service failed repeatedly; B3 continued meanwhile"
                            continue
                        self.b3_failures = 0
                        if getattr(result, "yielded_control", False):
                            retry_step = None if result.step_executed else step
                            if retry_step is not None:
                                continue
                        retry_step = None
                        self.b3_since_turn = Counter()
                        self.b3_auto_since_turn = 0
                        continue
                    self._b3_step()
            except Exception as exc:
                if run.final_score is None:
                    run.solver_note = f"error: {type(exc).__name__}: {exc}"
                    if run.state == "playing":
                        run.state = "crashed"
                    self._finish_if_needed()
            finally:
                _HOST.session = None
                if run.solver_note is None:
                    run.solver_note = f"tokens={SV._analyzer_reported_tokens(self.analyzer)}"
                self._finish_if_needed()
                self.state_path.unlink(missing_ok=True)
                self._write_analysis_html()
                self.write_viewer_payload()
                self._b3_write_summary()

        def _b3_write_summary(self):
            run = self.game.game_run
            out = {"game_id": run.game_id if run else None, "arm": self.b3_arm,
                   "levels_completed": int(self.game.current_state.levels_completed),
                   "actions": self.action_count, "actions_by_source": dict(self.b3_by_source),
                   "level_completions_by_source": self.b3_level_sources, "llm_turns": self.b3_turns,
                   "turn_triggers": dict(self.b3_turn_triggers), "contradictions": self.b3.stats.get("predictions", 0)
                   - self.b3.stats.get("pred_correct", 0), "revisions": self.b3_revisions,
                   "wall_seconds": dict(self.b3_t), "b3_stats": {k: v for k, v in self.b3.stats.items()
                                                                 if isinstance(v, (int, float))}}
            p = self.solver.job_dir / "b3_session" / f"{run.game_id}_p{self.pass_index}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(out, indent=1, default=str))

        # ---------------------------------------------------------------- sandbox channel
        def b3_request(self, req):
            m = str(req.get("method", ""))
            a = req.get("args") or {}
            b3 = self.b3
            obs = self.b3_last_obs
            node = b3._obs_node
            avail = obs.available_actions if obs else ()
            STATS[f"b3_{m}"] += 1
            if m == "status":
                fr = sum(1 for n in b3.graph.edges if b3._untried(n, avail))
                plan = b3.graph.bfs(node, lambda n: bool(b3._untried(n, avail)))[0]
                hu = [list(c) for c, v in b3.cls_stats.items() if v[1] == 0 and len(v[0]) >= 3]
                return {"level": int(obs.levels_completed) + 1, "state_id": _sid(node), "states_known": len(b3.graph.edges),
                        "frontier_states": fr, "untried_here": len(b3._untried(node, avail)),
                        "plan_to_frontier_len": len(plan) if plan else None, "historically_unchanged": hu[:20],
                        "recent_contradictions": list(self.b3_contradictions), "actions_by_source": dict(self.b3_by_source),
                        "trigger": self.b3_last_trigger}
            if m == "candidates":
                u = b3._untried(node, avail)
                return {"untried": [self._harness(c) for c in u[:60]], "tried_count": len(b3.graph.tried.get(node, ()))}
            if m == "transitions":
                return {"from_here": [{"action": self._harness(c), "to_state": _sid(v) if v != ("FATAL",) else "GAME_OVER",
                                       "changed": v != node} for c, v in b3.graph.edges.get(node, {}).items()][:80]}
            if m == "state_frame":
                want = str(a.get("state_id"))
                for n, g in b3.rep.items():
                    if _sid(n) == want:
                        return {"state_id": want, "rows": ["".join("%x" % int(v) for v in row) for row in g]}
                return {"error": f"unknown state_id {want!r}"}
            if m in ("plan_to_frontier", "plan_to_state"):
                if m == "plan_to_frontier":
                    goal = lambda n: bool(b3._untried(n, avail))
                else:
                    want = str(a.get("state_id"))
                    goal = lambda n: _sid(n) == want
                acts, nodes, _ = b3.graph.bfs(node, goal)
                if acts is None:
                    return {"actions": None, "length": 0, "reason": "no path in B3's recorded graph"}
                return {"actions": [self._harness(c) for c in acts], "length": len(acts), "reason": "shortest known path"}
            if m == "next":
                act = b3.decide(obs)
                if act.action == 0:
                    return {"action": "RESET", "why": "B3 would reset the level here; end the snippet to let it"}
                c = (act.action,) if act.action != 6 else (6, act.x, act.y)
                self.b3_pending_tool = c
                return {"action": self._harness(c), "why": act.why}
            if m == "blocked":
                b3.graph.tried.setdefault(node, set())
                if self.b3_pending_tool is not None:
                    b3.graph.tried[node].add(self.b3_pending_tool)
                self.b3_pending_tool = None
                b3.prev = None
                return {"ok": True}
            if m == "delegate":
                n = max(0, min(int(a.get("n", 100)), 1000))
                self.b3_delegated = n
                return {"autonomous_actions": n}
            if m == "revise":
                applied = {}
                rows, cols = [int(r) for r in a.get("mask_rows") or []], [int(c) for c in a.get("mask_cols") or []]
                if rows or cols:
                    mask = b3.vmask.mask.copy()
                    for r in rows:
                        mask[r, :] = True
                    for c in cols:
                        mask[:, c] = True
                    b3.vmask.mask = mask
                    b3._remap()
                    b3._obs_node = b3._key(b3._obs_grid)
                    applied["masked"] = {"rows": rows, "cols": cols}
                if a.get("extra_clicks"):
                    pts = [(int(c), int(r)) for r, c in a["extra_clicks"]]
                    b3.extra_clicks = list(dict.fromkeys(getattr(b3, "extra_clicks", []) + pts))
                    b3.cands = {}
                    applied["extra_clicks"] = len(pts)
                if a.get("undefer"):
                    b3.cls_stats = {}
                    applied["undefer"] = True
                self.b3_revisions.append({"at_action": self.action_count, **applied})
                return {"applied": applied}
            return {"error": f"unknown b3 method {m!r}"}

    B3Session.__qualname__ = "B3Session"
    globals()["B3Session"] = B3Session                 # picklable by module-level name
    return B3Session


def make_solver(arm: str, seed: int, **kw):
    from inference.framework.solver import HarnessSolver
    if arm == "franzen":
        return HarnessSolver(**kw)
    Session = make_session_class()

    class B3HarnessSolver(HarnessSolver):
        def _play_one(self, game, index, pass_index, local_server=None):
            from inference.framework import solver as SV
            try:
                run = game.game_run
                stem = self._run_stem(run.game_id, pass_index)
                analyzer = self._make_analyzer(game, index, local_server)
                session = Session(solver=self, game=game, analyzer=analyzer, game_index=index, pass_index=pass_index,
                                  state_path=self._artifacts_dir() / f"{stem}_{SV.RUNTIME_STATE_FILENAME}",
                                  transcript_path=self._transcripts_dir() / f"{stem}.txt",
                                  analysis_html_relpath=f"solver_analysis/{stem}.html", stop_event=self._stop_event,
                                  viewer_data_path=self._artifacts_dir() / f"{stem}_viewer_data.json")
                session._b3_setup(arm, seed)
                session.play()
            except Exception as exc:
                self._finish_after_error(game, exc)
    B3HarnessSolver.__qualname__ = "B3HarnessSolver"
    B3HarnessSolver.__module__ = __name__
    globals()["B3HarnessSolver"] = B3HarnessSolver     # TAAF pickles the solver
    return B3HarnessSolver(**kw)


def main():
    from run_harness_local import B0_PRIORITY_ENV, B0_SETUP_ENV
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["b3", "b3_franzen", "franzen"], required=True)
    p.add_argument("--src", default=str(ROOT / "build" / "b0src"))
    p.add_argument("--games", default="ls20")
    p.add_argument("--minutes", type=float, default=20.0)
    p.add_argument("--max-actions", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base-url", default="http://localhost:11434/v1")
    p.add_argument("--model", default="qwen3.5:35b")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    env = dict(B0_SETUP_ENV)
    env.update(B0_PRIORITY_ENV)
    env["ARC3_MAX_ACTIVE_STREAMS"] = 1
    env.update({"LOCAL_ANALYZER_BASE_URL": a.base_url, "OPENAI_BASE_URL": a.base_url,
                "LOCAL_ANALYZER_MODEL_ID": a.model, "INFERENCE_ANALYZER_MODEL": a.model, "MPLBACKEND": "Agg",
                "TAAF_RUN_AS_SUBMISSION": "0", "TAAF_MINIMAL_DIAGNOSTICS": "0", "OPERATION_MODE": "offline"})
    os.environ.update({k: str(v) for k, v in env.items()})
    os.environ["ONLY_RESET_LEVELS"] = "true"
    src = Path(a.src)
    sys.path.insert(0, str(src / "ARC3-Inference"))
    sys.path.insert(0, str(src / "tufa-arc-agi-framework" / "src"))
    import arc_agi
    import taaf.benchmark
    import taaf.game_api
    if a.arm != "franzen":
        install()
    env_dir = str(ROOT / "data" / "environment_files")
    spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    ids = [e.game_id for e in arc.available_environments if any(e.game_id.startswith(w) for w in a.games.split(","))]
    games = [taaf.game_api.GameAPI(env_name=g, arcade_spec=spec) for g in ids]
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    solver = make_solver(a.arm, a.seed, label=f"{a.arm}-local", model="local", analyzer_timeout=900.0,
                         max_actions_per_game=a.max_actions, max_runtime_s_per_game=a.minutes * 60,
                         max_generated_tokens_per_game=None, concurrency=1, save_request_logs=True)
    bm = taaf.benchmark.Benchmark(label=a.arm, games=games, solver=solver, n_passes=1, job_dir=out)
    (out / "run_config.json").write_text(json.dumps({"arm": a.arm, "env": env, "games": ids, "args": vars(a),
                                                     "b3_config": B3_CONFIG}, indent=1, default=str))
    t0 = time.time()
    soft_end = datetime.now() + timedelta(minutes=a.minutes * len(games) + 5)
    asyncio.run(bm.run(soft_end_time=soft_end, runtime_environment=None, minimal_diagnostics=False))
    (out / "b3_channel_stats.json").write_text(json.dumps(dict(STATS), indent=1))
    print(f"done in {time.time() - t0:.1f}s; outputs in {out}")


if __name__ == "__main__":
    main()
