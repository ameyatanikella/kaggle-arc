"""B2: individually switchable patches on top of the B0 harness (Franzen M2).

Install in the notebook/driver process AFTER the harness is importable and
BEFORE the benchmark runs:

    import b2_patches; b2_patches.install()

Every switch defaults OFF; with all switches off the harness is byte-identical
in behaviour to B0. Switches (env vars, read at install time):

  B2_BAR_FIX=1        delete the system-prompt line telling the model to ignore the
                      edge bar ("do not get distracted by it ..."). B0's own game-over
                      text says deaths come from that bar depleting; the two lines
                      contradict. Correction only, no new advice.
  B2_SOLVED_MEMORY=1  pin facts about each cleared level (action count, winning
                      action tail, the model's own reasoning tail, and its stated
                      "Rule for level N:") at the end of the system prompt for the
                      rest of the game. Port of rellik13's M85 (Apache-2.0,
                      github.com/LohitSiriki/arc-agi-3-milestone2-solution).

`b2_summary()` returns counters showing whether each switch actually fired.
"""
from __future__ import annotations

import json
import os
import re
import threading

_LOCK = threading.Lock()
STATS = {"installed": [], "bar_fix_applied": 0, "levels_recorded": 0, "rules_captured": 0,
         "rules_missing": 0, "asks": 0, "block_chars_max": 0, "errors": 0, "last_error": ""}


def _on(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "on", "yes"}


def _err(exc: BaseException) -> None:
    with _LOCK:
        STATS["errors"] += 1
        STATS["last_error"] = f"{type(exc).__name__}: {exc}"[:300]


# --------------------------------------------------------------------------- bar fix
BAR_LINE = ("- In many games, a long horizontal or vertical line near an edge is a timer or "
            "remaining-steps bar. It often shrinks or changes each step. If you identify such a "
            "bar, do not get distracted by it or treat it as core gameplay state unless there is "
            "concrete evidence that it interacts with the puzzle mechanics.\n")


def _install_bar_fix(T) -> None:
    inner = T._build_system_prompt
    if getattr(inner, "_b2_bar_fix", False):
        return

    def _build_system_prompt(*args, **kwargs):
        p = inner(*args, **kwargs)
        if BAR_LINE in p:
            p = p.replace(BAR_LINE, "")
            with _LOCK:
                STATS["bar_fix_applied"] += 1
        return p

    _build_system_prompt._b2_bar_fix = True
    T._build_system_prompt = _build_system_prompt


# --------------------------------------------------------------------------- prompt fixes
_EPHEMERAL_CODE_DESC = "Python code to run. The snippet is ephemeral and is not saved across tool calls."
_RETAINED_CODE_DESC = ("Python code to run. Variables reset between calls; functions you define are "
                       "retained as described in the system prompt.")
_FUSED_BULLETS = "Sometime they might also be 1x1 tokens.- Some games"
_SPLIT_BULLETS = "Sometimes they might also be 1x1 tokens.\n- Some games"


def _install_prompt_fixes(T) -> None:
    """Corrections only: (1) the python tool schema calls snippets 'ephemeral and not
    saved' although B0 retains model-defined functions for the whole game
    (ARC3_PERSISTENT_FUNCTIONS=1) - the two statements contradict; (2) two system-prompt
    bullets are fused by a missing newline."""
    persistent = os.environ.get("ARC3_PERSISTENT_FUNCTIONS", "0").strip().lower() in {"1", "true", "on", "yes"}
    if persistent and T._PYTHON_TOOL_DESCRIPTION.startswith("Run one ephemeral Python snippet"):
        T._PYTHON_TOOL_DESCRIPTION = T._PYTHON_TOOL_DESCRIPTION.replace(
            "Run one ephemeral Python snippet", "Run one Python snippet", 1)
    A = T.ToolAgent
    inner_tools = A._tools

    def _tools(self, state_path):
        tools = inner_tools(self, state_path)
        if persistent:
            for t in tools:
                props = t.get("function", {}).get("parameters", {}).get("properties", {})
                code = props.get("code")
                if isinstance(code, dict) and code.get("description") == _EPHEMERAL_CODE_DESC:
                    code["description"] = _RETAINED_CODE_DESC
        return tools

    A._tools = _tools
    inner_sp = T._build_system_prompt

    def _build_system_prompt(*args, **kwargs):
        return inner_sp(*args, **kwargs).replace(_FUSED_BULLETS, _SPLIT_BULLETS)

    T._build_system_prompt = _build_system_prompt


# --------------------------------------------------------------------------- stuck nudge
STUCK_TOKENS = int(os.environ.get("B2_STUCK_TOKENS", "45000"))
STUCK_MAX_ACTIONS = int(os.environ.get("B2_STUCK_MAX_ACTIONS", "40"))
# Trigger: "abs" = tokens >= STUCK_TOKENS and actions <= STUCK_MAX_ACTIONS (first arm);
# "tpa" = tokens >= STUCK_TPA_MIN_TOKENS and tokens/(actions+1) >= STUCK_TPA (calibrated on B0's
# log: completed levels median ~695 tokens/action, failed final levels ~1384).
STUCK_TRIGGER = os.environ.get("B2_STUCK_TRIGGER", "abs").strip().lower()
STUCK_TPA = float(os.environ.get("B2_STUCK_TPA", "1200"))
STUCK_TPA_MIN_TOKENS = int(os.environ.get("B2_STUCK_TPA_MIN_TOKENS", "30000"))
STUCK_TEXT = ("\n\nProgress check: about {k}K generated tokens and {m} actions spent on this level "
              "without clearing it. Experiments are cheap compared with extended reasoning: if your "
              "current idea has not worked, test an untried mechanic or goal hypothesis directly with "
              "a few actions and update from what changes.")


def _install_stuck_nudge(T) -> None:
    """Measured motivation: on the level where B0 runs fail, they had used a median of
    0.73x the human baseline actions (Franzen v3 log), and in a real Flash-Next run
    several stuck levels used 0.2-0.3x baseline actions after ~30K reasoning tokens.
    Fires only when a level has consumed >= STUCK_TOKENS generated tokens with
    <= STUCK_MAX_ACTIONS actions, and at most once per STUCK_TOKENS tokens."""
    A = T.ToolAgent
    inner = A._build_user_prompt
    if getattr(inner, "_b2_stuck", False):
        return

    def _build_user_prompt(self, action_num, *args, **kwargs):
        out = inner(self, action_num, *args, **kwargs)
        try:
            tok = int(getattr(self, "_session_generated_tokens", 0)) - int(getattr(self, "_tokens_at_level_start", 0))
            summ = getattr(self, "_last_step_summary", None) or {}
            acts = max(0, int(T._priority_action_count(summ)) - int(getattr(self, "_actions_at_level_start", 0)))
            start = int(getattr(self, "_tokens_at_level_start", 0))
            if getattr(self, "_b2_stuck_level_start", None) != start:
                self._b2_stuck_level_start = start
                self._b2_stuck_next = STUCK_TOKENS if STUCK_TRIGGER != "tpa" else STUCK_TPA_MIN_TOKENS
            if STUCK_TRIGGER == "tpa":
                fire = tok >= max(self._b2_stuck_next, STUCK_TPA_MIN_TOKENS) and tok / (acts + 1) >= STUCK_TPA
            else:
                fire = tok >= self._b2_stuck_next and acts <= STUCK_MAX_ACTIONS
            if fire:
                out += STUCK_TEXT.format(k=tok // 1000, m=acts)
                self._b2_stuck_next = tok + STUCK_TOKENS
                with _LOCK:
                    STATS["stuck_nudges"] = STATS.get("stuck_nudges", 0) + 1
        except Exception as exc:
            _err(exc)
        return out

    for f in dir(inner):
        if f.endswith("_wrapped"):
            setattr(_build_user_prompt, f, getattr(inner, f))
    _build_user_prompt._b2_stuck = True
    A._build_user_prompt = _build_user_prompt


# --------------------------------------------------------------------------- solved memory
M_ACTIONS_KEPT = 30
M_REASONING_CHARS = 900
M_RULE_CHARS = 600
M_FULL_LEVELS = 3
M_ASK_TURNS = 2
M_HEADER = "=== SOLVED LEVELS (facts from this game + your own conclusions; kept for the whole game) ==="
M_FOOTER = "=== END SOLVED LEVELS ==="


def _msg_text(msg) -> str:
    parts = []
    for k in ("content", "reasoning_content", "reasoning"):
        v = msg.get(k) if isinstance(msg, dict) else None
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, list):
            parts += [p.get("text", "") for p in v if isinstance(p, dict) and isinstance(p.get("text"), str)]
    return "\n".join(p for p in parts if p)


def level_actions(history_entries, level: int) -> list[str]:
    """Actions taken on `level` (history[i].frame is the frame AFTER history[i].action)."""
    out, prev = [], None
    for e in history_entries or []:
        act = str(getattr(e, "action", "") or "").strip()
        if act and prev == level:
            out.append(act)
        lv = getattr(getattr(e, "frame", None), "level", None)
        if isinstance(lv, int):
            prev = lv
    return out


def find_rule(text: str, level: int):
    m = re.search(r"Rule for level\s*%d\s*:[ \t]*" % int(level), text or "", re.I)
    if not m:
        return None
    lines = text[m.end():].split("\n")
    first = lines[0].strip()
    if len(first) >= 40:
        rule = first
    else:
        out = [first] if first else []
        for ln in lines[1:6]:
            if not ln.strip():
                if out:
                    break
                continue
            out.append(ln.strip())
        rule = " ".join(out)
    rule = " ".join(rule.split())[:M_RULE_CHARS]
    return rule if len(rule) >= 20 else None


def memory_block(levels: list[dict]) -> str:
    if not levels:
        return ""
    lines = [M_HEADER]
    full_from = max(0, len(levels) - M_FULL_LEVELS)
    for i, rec in enumerate(levels):
        lines.append(f"Level {rec['level']} - cleared after {rec['n_actions']} actions.")
        lines.append(f"  Your rule: {rec['rule'] if rec.get('rule') else 'not stated'}")
        if i >= full_from:
            tail = rec["actions"]
            skipped = rec["n_actions"] - len(tail)
            lines.append("  Last actions before it cleared" + (f" ({skipped} earlier omitted)" if skipped > 0 else "")
                         + ": " + (", ".join(tail) if tail else "none recorded"))
            if rec.get("reasoning"):
                lines.append("  Your reasoning in the turn that cleared it: " + rec["reasoning"])
    lines.append("Reuse these where the new level has the same controls; verify with one probe before relying on them.")
    lines.append(M_FOOTER)
    return "\n".join(lines)


def _install_solved_memory(T) -> None:
    A = T.ToolAgent
    inner = A._build_user_prompt
    if getattr(inner, "_b2_memory", False):
        return

    def _set_block(self):
        block = memory_block(getattr(self, "_b2_levels", []))
        sp = getattr(self, "_system_prompt", "") or ""
        old = getattr(self, "_b2_block_text", "")
        if old and old in sp:
            sp = sp.replace(old, block)
        elif block:
            sp = sp.rstrip("\n") + "\n\n" + block
        self._system_prompt = sp
        self._b2_block_text = block
        with _LOCK:
            STATS["block_chars_max"] = max(STATS["block_chars_max"], len(block))

    def _capture(self) -> bool:
        pending = [r for r in getattr(self, "_b2_levels", [])
                   if not r.get("rule") and r.get("asks", 0) > r.get("checks", 0)]
        if not pending:
            return False
        texts = [_msg_text(m) for m in (getattr(self, "_history_messages", []) or [])
                 if isinstance(m, dict) and m.get("role") == "assistant"]
        changed = False
        for rec in pending:
            rec["checks"] = rec.get("checks", 0) + 1
            for t in reversed(texts):
                rule = find_rule(t, rec["level"])
                if rule:
                    rec["rule"] = rule
                    changed = True
                    with _LOCK:
                        STATS["rules_captured"] += 1
                    break
            if not rec.get("rule") and rec["checks"] >= M_ASK_TURNS:
                with _LOCK:
                    STATS["rules_missing"] += 1
        return changed

    def _build_user_prompt(self, action_num, *args, **kwargs):
        out = inner(self, action_num, *args, **kwargs)
        try:
            if not hasattr(self, "_b2_levels"):
                self._b2_levels = []
                self._b2_block_text = ""
            changed = _capture(self)
            summary = kwargs.get("previous_step_summary") or {}
            level_now = getattr(kwargs.get("current_frame"), "level", None)
            if summary.get("level_transition") and not summary.get("run_complete") and isinstance(level_now, int):
                solved = level_now - 1
                if solved >= 1 and all(r["level"] != solved for r in self._b2_levels):
                    acts = level_actions(kwargs.get("history_entries") or [], solved)
                    reasoning = ""
                    for m in reversed(getattr(self, "_history_messages", []) or []):
                        if isinstance(m, dict) and m.get("role") == "assistant":
                            t = " ".join(_msg_text(m).split())
                            if t:
                                reasoning = t[-M_REASONING_CHARS:]
                                break
                    self._b2_levels.append(dict(level=solved, n_actions=len(acts), actions=acts[-M_ACTIONS_KEPT:],
                                                reasoning=reasoning, rule=None, asks=0, checks=0))
                    changed = True
                    with _LOCK:
                        STATS["levels_recorded"] += 1
            last = self._b2_levels[-1] if self._b2_levels else None
            if last is not None and not last.get("rule") and last["asks"] < M_ASK_TURNS:
                last["asks"] += 1
                with _LOCK:
                    STATS["asks"] += 1
                out += ("\n\nLEVEL %d SOLVED. In your reply this turn, before acting, write one short paragraph that "
                        "starts exactly with 'Rule for level %d:' -- the goal, what each control/action did, and the "
                        "plan that won (at most 5 lines). It is pinned for every later level."
                        % (last["level"], last["level"]))
            if changed:
                _set_block(self)
        except Exception as exc:  # never break the game loop
            _err(exc)
        return out

    for f in dir(inner):
        if f.endswith("_wrapped"):
            setattr(_build_user_prompt, f, getattr(inner, f))
    _build_user_prompt._b2_memory = True
    A._build_user_prompt = _build_user_prompt


# --------------------------------------------------------------------------- sandbox tools
TOOLS_DOC = (
    "- Preloaded compute-only helpers (they never act; no action cost): "
    "`bfs_plan(start, step, is_goal, actions)` returns the shortest action list in YOUR simulator "
    "`step(state, action) -> next_state` (None = blocked). "
    "`distinguish(sims, starts, actions)` returns the shortest action sequence whose predicted "
    "observations differ between your candidate simulators `sim(state, action) -> (next_state, obs)` "
    "- use it to choose one probe that separates competing hypotheses. "
    "`gf2_solve(A, b, cost=None)` gives the cheapest press set for toggle/XOR systems "
    "(A[i][j]=1 if press j flips cell i); `zm_solve(A, b, m)` the same for cyclic counters mod m.\n"
)
_TOOLS_REG_ANCHOR = '        runtime_globals["last_action_call_result"] = action_result'
_TOOLS_SRC_ANCHOR = "HOST_STDOUT = sys.stdout"


def _install_tools(T) -> None:
    import inference.agent.python_tool_sandbox as S
    here = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.join(here, "sandbox_tools.py")).read()
    b = S._SANDBOX_BOOTSTRAP
    if "B2_TOOL_NAMES" in b:
        return
    assert b.count(_TOOLS_SRC_ANCHOR) == 1 and b.count(_TOOLS_REG_ANCHOR) == 1, "sandbox bootstrap changed"
    names = ("bfs_plan", "distinguish", "gf2_solve", "zm_solve")
    reg = "".join(f'\n        runtime_globals["{n}"] = {n}' for n in names)
    b = b.replace(_TOOLS_SRC_ANCHOR, src + "\n\n" + _TOOLS_SRC_ANCHOR, 1)
    b = b.replace(_TOOLS_REG_ANCHOR, _TOOLS_REG_ANCHOR + reg, 1)
    S._SANDBOX_BOOTSTRAP = b
    inner = T._build_system_prompt

    def _build_system_prompt(*args, **kwargs):
        p = inner(*args, **kwargs)
        anchor = "\n\nTool session rules:\n"
        if TOOLS_DOC not in p:
            p = p.replace(anchor, "\n" + TOOLS_DOC + anchor, 1) if anchor in p else p + "\n" + TOOLS_DOC
        return p

    T._build_system_prompt = _build_system_prompt


def tool_usage_from_requests(path: str) -> dict:
    """Count how often generated code called each helper (for 'does it control behaviour?')."""
    counts = {n: 0 for n in ("bfs_plan", "distinguish", "gf2_solve", "zm_solve")}
    calls = 0
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            for m in r.get("messages", [])[-1:]:
                for tc in (m.get("tool_calls") or []):
                    calls += 1
                    code = json.loads(tc.get("function", {}).get("arguments", "{}")).get("code", "")
                    for n in counts:
                        counts[n] += code.count(n + "(")
    return {"tool_calls_seen": calls, **counts}


# --------------------------------------------------------------------------- install
def install() -> dict:
    import inference.agent.tool_agent as T
    if _on("B2_BAR_FIX"):
        _install_bar_fix(T)
        STATS["installed"].append("bar_fix")
    if _on("B2_SOLVED_MEMORY"):
        _install_solved_memory(T)
        STATS["installed"].append("solved_memory")
    if _on("B2_TOOLS"):
        _install_tools(T)
        STATS["installed"].append("tools")
    if _on("B2_PROMPT_FIXES"):
        _install_prompt_fixes(T)
        STATS["installed"].append("prompt_fixes")
    if _on("B2_STUCK_NUDGE"):
        _install_stuck_nudge(T)
        STATS["installed"].append("stuck_nudge")
    print("B2 patches installed:", json.dumps(STATS["installed"]), flush=True)
    return dict(STATS)


def b2_summary() -> dict:
    with _LOCK:
        return dict(STATS)
