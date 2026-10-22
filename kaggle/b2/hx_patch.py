"""B2_HX: bind arc3.hx to the Duck/Franzen harness at runtime (no harness file is edited).

Layers (env switches, read by b2_patches.install; B2_HX=1 turns on all three):
  B2_HX_EVENTS    faithful events in the sandbox: transitions[i].kind / .level, `moves`, guarded
                  frame_diff default (hx_child.py); the host keeps a typed event log with the true
                  WIN frames (frames[-2] of a level-completing step).
  B2_HX_TOOLS     sandbox helpers hx_check / hx_probe (run in the sandbox on model-written callables)
                  and hx_goals / hx_relations / hx_wins (answered by the host session).
  B2_HX_TRIGGERS  short evidence reports appended to the next user prompt ONLY when a trigger fires:
                  level completed, same board + action gave a different result, >= 4 consecutive
                  interior no-ops. Never on every turn.

Every hook is wrapped so an HX failure can never break a game (errors are counted in hx_summary()).
Requires arc3/ (perception + hx) importable: on Kaggle the notebook writes it to /tmp/arc3.
"""
from __future__ import annotations

import inspect
import json
import os
import threading
from collections import Counter

_HOST = threading.local()
STATS = Counter()
INTERVENTIONS: list[dict] = []
CAUSAL_REPORTS: list[dict] = []
_LOCK = threading.Lock()
_STATE = {"installed": False, "events": False, "tools": False, "triggers": False,
          "ledger": False, "stall": False, "stall_neutral": False,
          "causal_report": False, "causal_neutral": False}

_CHILD_SRC_ANCHOR = "HOST_STDOUT = sys.stdout"
_CHILD_REFRESH_ANCHOR = '        runtime_globals["last_action_call_result"] = action_result\n'
_HOST_ANCHOR = '            if msg_type == "action":\n'
_HOST_SRC = '''            if msg_type == "hx":
                _h = getattr(_HX_HOST, "session", None)
                try:
                    _rep = _HX_REQUEST(_h, dict(message.get("request") or {}))
                except Exception as _exc:  # noqa: BLE001
                    _rep = {"error": type(_exc).__name__ + ": " + str(_exc)}
                _send_json_line(process.stdin, {"type": "hx_result", "result": _rep})
                continue
'''
_TOOLS_ON_LINE = "_HX_TOOLS_ON = True      # hx_patch rewrites this line when only the events layer is installed"
_LEDGER_ON_LINE = "_HX_LEDGER_ON = False    # hx_patch rewrites this line when the HX2 ledger is installed"
_STALL_ON_LINE = "_HX_STALL_ON = False     # hx_patch rewrites this line when the Stage 1 stall intervention is installed"


def _bump(k, n=1):
    with _LOCK:
        STATS[k] += n


def _json_safe(v):
    if v is None or isinstance(v, (str, bool, int, float)):
        return v
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_json_safe(x) for x in v]
    try:
        import numpy as np
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
    except Exception:  # noqa: BLE001
        pass
    return str(v)


def _request(session, req: dict) -> dict:
    if session is None:
        return {"error": "HX is not available in this session."}
    method = str(req.get("method") or "")
    if method not in ("goals", "relations", "wins", "status", "ledger", "before_action",
                      "probe_logged", "action_logged", "interventions"):
        return {"error": f"unknown hx method {method!r}"}
    _bump("req_" + method)
    return _json_safe(session.request(method, dict(req.get("args") or {})))


def _child_source(tools_on: bool, ledger_on: bool = False, stall_on: bool = False) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.join(here, "hx_child.py")).read()
    if not tools_on:
        assert src.count(_TOOLS_ON_LINE) == 1
        src = src.replace(_TOOLS_ON_LINE, "_HX_TOOLS_ON = False")
    if ledger_on:
        assert src.count(_LEDGER_ON_LINE) == 1
        src = src.replace(_LEDGER_ON_LINE, "_HX_LEDGER_ON = True")
    if stall_on:
        assert src.count(_STALL_ON_LINE) == 1
        src = src.replace(_STALL_ON_LINE, "_HX_STALL_ON = True")
    return src


def system_doc() -> str:
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("hx_child_doc", os.path.join(here, "hx_child.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    doc = mod.HX_EVENTS_DOC if _STATE["events"] else ""
    if _STATE["tools"]:
        doc += mod.HX_DOC
    if _STATE["ledger"]:
        doc += mod.HX_LEDGER_DOC
    return doc


def install(events: bool = True, tools: bool = True, triggers: bool = True, ledger: bool = False,
            quiet: bool = False, stall: bool = False, stall_neutral: bool = False,
            stall_turns: int = 5, stall_tokens: int = 20000,
            causal_report: bool = False, causal_neutral: bool = False) -> dict:
    """Idempotent. Any layer implies the host event log (needed by tools and triggers). The HX2 ledger
    (B2_HX_LEDGER) needs the sandbox channel and the triggers, so it switches both on."""
    if ledger:
        tools = triggers = True
    if stall or stall_neutral:
        events = tools = True
    if causal_report and causal_neutral:
        raise ValueError("choose exactly one causal feedback arm")
    if causal_report or causal_neutral:
        events = tools = True
    if quiet:
        triggers = False
    if _STATE["installed"]:
        return dict(_STATE)
    import numpy as np
    import inference.agent.python_tool_sandbox as S
    import inference.agent.tool_agent as T
    from inference.framework import solver as SV
    from arc3.hx.session import HXSession

    _STATE.update(events=bool(events), tools=bool(tools), triggers=bool(triggers),
                  ledger=bool(ledger), stall=bool(stall), stall_neutral=bool(stall_neutral),
                  causal_report=bool(causal_report), causal_neutral=bool(causal_neutral))

    # ---- sandbox child: helpers + refresh hook (events and/or tools)
    if events or tools:
        b = S._SANDBOX_BOOTSTRAP
        assert b.count(_CHILD_SRC_ANCHOR) == 1 and b.count(_CHILD_REFRESH_ANCHOR) == 1, "sandbox bootstrap changed"
        b = b.replace(_CHILD_SRC_ANCHOR, _child_source(tools, ledger, stall) + "\n\n" + _CHILD_SRC_ANCHOR, 1)
        b = b.replace(_CHILD_REFRESH_ANCHOR, _CHILD_REFRESH_ANCHOR + "        _hx_refresh(runtime_globals)\n", 1)
        S._SANDBOX_BOOTSTRAP = b

    # ---- sandbox host loop: 'hx' requests (tools only)
    if tools:
        src = inspect.getsource(S.run_sandboxed_python)
        assert src.count(_HOST_ANCHOR) == 1, "sandbox host loop changed"
        src = src.replace(_HOST_ANCHOR, _HOST_SRC + _HOST_ANCHOR, 1)
        S._HX_HOST = _HOST
        S._HX_REQUEST = _request
        exec(compile(src, S.__file__, "exec"), S.__dict__)
        T.run_sandboxed_python = S.run_sandboxed_python

    # ---- system prompt: tool text (events doc and/or tools doc)
    if events or tools:
        doc = system_doc()
        inner_sp = T._build_system_prompt

        def _build_system_prompt(*args, **kwargs):
            p = inner_sp(*args, **kwargs)
            anchor = "\n\nTool session rules:\n"
            if doc and doc not in p:
                p = p.replace(anchor, "\n" + doc + anchor, 1) if anchor in p else p + "\n" + doc
            return p
        T._build_system_prompt = _build_system_prompt

    # ---- session binding: one HXSession per game, recorded at the engine boundary
    G = SV._HarnessGameSession
    inner_play = G.play
    inner_exec = G._execute_action

    def play(self):
        try:
            self._hx = HXSession(triggers=triggers, ledger=ledger, stall=stall,
                                 stall_neutral=stall_neutral, stall_turns=stall_turns,
                                 stall_tokens=stall_tokens)
            st = self.game.current_state
            self._hx.start(np.asarray(SV._grid_from_state(st), dtype=np.uint8), int(st.levels_completed))
            _bump("sessions")
        except Exception as exc:  # noqa: BLE001
            self._hx = None
            _bump("errors_start")
            STATS["last_error"] = f"start: {type(exc).__name__}: {exc}"[:300]
        prev = getattr(_HOST, "session", None)
        _HOST.session = self._hx
        try:
            return inner_play(self)
        finally:
            _HOST.session = prev
            if getattr(self, "_hx", None) is not None:
                for k, v in self._hx.stats.items():
                    _bump("s_" + k, int(v))
                if self._hx.intervention_log:
                    with _LOCK:
                        room = max(0, 200 - len(INTERVENTIONS))
                        INTERVENTIONS.extend(self._hx.intervention_log[:room])

    def _execute_action(self, action, **kwargs):
        hx = getattr(self, "_hx", None)
        pre = lb = legal = None
        if hx is not None:
            try:
                st0 = self.game.current_state
                pre = np.asarray(SV._grid_from_state(st0), dtype=np.uint8)
                lb = int(st0.levels_completed)
                legal = tuple(int(a) for a in (st0.available_actions or ()))
            except Exception:  # noqa: BLE001
                hx = None
        payload = inner_exec(self, action, **kwargs)
        if hx is not None:
            try:
                st1 = self.game.current_state
                raw = getattr(st1, "raw", None)
                frames = list(getattr(raw, "frame", None) or [])
                if not frames:
                    frames = [SV._grid_from_state(st1)]
                aid = int(action.id.value)
                if aid == 6:
                    d = dict(action.data or {})
                    act = (6, int(d.get("x", 0)), int(d.get("y", 0)))
                else:
                    act = (aid,)
                state = getattr(getattr(raw, "state", None), "name", "NOT_FINISHED")
                hx.record(act, frames, state, lb, int(st1.levels_completed), legal=legal, pre_grid=pre,
                          automatic=bool(kwargs.get("automatic", False)))
            except Exception as exc:  # noqa: BLE001
                _bump("errors_record")
                STATS["last_error"] = f"record: {type(exc).__name__}: {exc}"[:300]
        return payload

    G.play = play
    G._execute_action = _execute_action

    # ---- triggered reports in the next user prompt
    if triggers or stall or stall_neutral or causal_report or causal_neutral:
        A = T.ToolAgent
        inner_up = A._build_user_prompt

        def _build_user_prompt(self, action_num, *args, **kwargs):
            out = inner_up(self, action_num, *args, **kwargs)
            sess = getattr(_HOST, "session", None)
            if sess is not None:
                try:
                    rep = sess.take_report() if triggers else ""
                    if rep:
                        out += "\n\n" + rep
                        _bump("reports_delivered")
                        _bump("report_chars_delivered", len(rep))
                    if causal_report or causal_neutral:
                        from arc3.hx.causal_feedback import render
                        feedback, facts = render(sess, neutral=causal_neutral)
                        out += "\n\n" + feedback
                        _bump("causal_reports_delivered")
                        _bump("causal_report_chars", len(feedback))
                        with _LOCK:
                            CAUSAL_REPORTS.append({"turn": len(CAUSAL_REPORTS) + 1,
                                                   "facts": _json_safe(None if facts is None else {
                                                       "level": facts["level"],
                                                       "recorded_steps": facts["recorded_steps"],
                                                       "actions": facts["actions"],
                                                       "changed": facts["changed"],
                                                       "clicks": facts["clicks"],
                                                       "click_rank": facts["click_rank"],
                                                       "shown_laws": facts["shown_laws"]}),
                                                   "chars": len(feedback)})
                    if stall or stall_neutral:
                        frame = kwargs.get("current_frame")
                        lv = max(0, int(getattr(frame, "level", 1) or 1) - 1)
                        summ = getattr(self, "_last_step_summary", None) or {}
                        acts = max(0, int(T._priority_action_count(summ))
                                   - int(getattr(self, "_actions_at_level_start", 0)))
                        tok = max(0, int(getattr(self, "_session_generated_tokens", 0))
                                   - int(getattr(self, "_tokens_at_level_start", 0)))
                        intervention = sess.on_turn_start(lv, acts, tok)
                        if intervention:
                            out += "\n\n" + intervention
                            _bump("stall_prompts_delivered")
                except Exception as exc:  # noqa: BLE001
                    _bump("errors_report")
                    STATS["last_error"] = f"report: {type(exc).__name__}: {exc}"[:300]
            return out
        for f in dir(inner_up):
            if f.endswith("_wrapped"):
                setattr(_build_user_prompt, f, getattr(inner_up, f))
        A._build_user_prompt = _build_user_prompt

    _STATE["installed"] = True
    return dict(_STATE)


def hx_summary() -> dict:
    with _LOCK:
        return {**{k: v for k, v in _STATE.items()}, **dict(STATS),
                "interventions": list(INTERVENTIONS), "causal_reports": list(CAUSAL_REPORTS)}
