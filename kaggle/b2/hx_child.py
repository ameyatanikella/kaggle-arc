"""HX helpers preloaded into the harness Python sandbox (B2_HX). Pure Python, stdlib only.

Injected into the sandbox bootstrap (module level) by hx_patch.py; `_hx_refresh(runtime_globals)`
runs on every state refresh. HX_DOC is the single source of truth for the model-facing text
(hx_patch inserts it into the system prompt; tests/test_hx_child.py checks every documented key).

Faithful events (B2_HX_EVENTS):
  * every item of `transitions` gets `.kind` in {'move', 'level_up', 'death', 'reset', 'won'} and
    `.level`; only 'move' items are consecutive decision states of one level;
  * `moves` = the 'move' transitions of the CURRENT level (what a dynamics model must explain);
  * `frame_diff()` with no arguments refuses to diff across a level change, death or reset.
Tools (B2_HX_TOOLS): hx_check, hx_probe (run here, on model-written callables), and
hx_goals / hx_relations / hx_wins (answered by the host from its faithful event log).
"""
import collections as _hxc
import itertools as _hxit
import math as _hxmath
import time as _hxtime

HX_NAMES = ("hx_check", "hx_probe", "hx_goals", "hx_relations", "hx_wins")

HX_EVENTS_DOC = (
    "- Faithful transitions: every item of `transitions` has `.kind` ('move', 'level_up', 'death', 'reset', "
    "'won') and `.level`. Only 'move' items are one action between two consecutive boards of the same level; "
    "a 'level_up' item's after_frame is already the NEXT level's first board and a 'reset' item is not game "
    "dynamics. `moves` lists the 'move' transitions of the current level: validate models on `moves`, not on "
    "all `transitions`.\n"
)

HX_DOC = (
    "- HX helpers (compute only, no game actions; each stops after `time_limit` s, default 8). Each returns a dict:\n"
    "  - `hx_check(predict, observe=None, level=None)` -> {'checked', 'correct', 'mismatches', 'errors'}: replays "
    "your model on the recorded moves: predict(before_frame, action) must equal observe(after_frame) (default "
    "observe: the frame's ascii). Each mismatch has 'step', 'action', 'predicted', 'observed', 'cells' (up to 5 "
    "differing (row, col, predicted, actual) when both are ascii) and 'cause': 'hidden_state' (same "
    "observe(before) and action earlier gave another result), 'border' (differences only in the 4-cell border) or "
    "'dynamics'.\n"
    "  - `hx_probe(models, goals, actions, depth=2)` -> {'decision', 'action', 'probe', 'gain', 'plans', "
    "'groups'}: models = {name: (state, step)} with step(state, action) -> next state (None = cannot predict), "
    "all over the same state representation; goals = {name: is_goal(state)}. It plans for every model x goal. "
    "decision 'act': all plans start with the same action ('action'). 'probe': the action list 'probe' (length "
    "<= depth) splits the hypotheses ('groups' = which predict what) and is worth more than its actions "
    "('gain' > 0); run it and keep the hypotheses that predicted what you observe. 'follow': no probe pays, "
    "'action' starts the plan that succeeds under the most hypotheses. 'none': no hypothesis reaches a goal - "
    "write a better model rather than forcing one. Optional weights={name: w}.\n"
    "  - On a stall turn, use `hx_probe({'m1': step1, 'm2': step2}, {'g1': goal1, 'g2': goal2}, "
    "state_of=state_of)`: here each step is `step(state, action)` and `state_of(frame)` extracts the "
    "same state representation from observed frames. This form checks candidates on recorded same-level "
    "moves, excludes models refuted by those moves and goals already true on non-winning frames, reports "
    "`state_diagnostics` (Markov conflicts, blind gameplay moves) and `empirical_invariants` (exact "
    "per-action coordinate/set deltas and algebraic relations on `state_of(moves)`), precommits next-action "
    "forecasts, and logs whether you follow its recommendation. One surviving model may plan after two "
    "correct recorded predictions (or one non-noop prediction when a competing model was falsified). "
    "A 'none' result means repair the models or goals before acting on a plan.\n"
    "  - `hx_goals()` -> {'candidates', 'false_now', 'true_now', 'hazards', 'last_win'}: facts that became true "
    "when earlier levels were completed (evidence from the actual win frames, not a verified goal) and whether "
    "each is true on the current board. `hx_relations()` -> {'relations', 'conflicts'}: action relations "
    "verified on this level (no effect, undo, order irrelevant, ...) with the number of states supporting them "
    "and counterexamples; a relation holds only in its context. `hx_wins()` -> {'wins'}: per completed level, "
    "the winning action and the board at the moment of the win.\n"
)

HX_LEDGER_DOC = (
    "  - `hx_ledger()` -> {'forecasts', 'goals', 'hidden_state', 'camera_shifts'}: the per-game hypothesis "
    "ledger. 'forecasts': how often each built-in model (graph = transitions seen before, algebra, history, "
    "undo variants) predicted your actions' outcomes BEFORE they happened, with 'effect_accuracy' counted only on "
    "actions that changed the board. 'goals': 'sufficient_candidates' (true at earlier wins, never yet true "
    "without a level-up) and 'prerequisites' (true at wins but also seen without one, so not enough alone). "
    "'hidden_state': same-board/same-action conflicts and which history feature explains them, if any. "
    "'camera_shifts': detected view translations (frames are screen coordinates).\n"
)

_HX_TL = 8.0
_HX_TOOLS_ON = True      # hx_patch rewrites this line when only the events layer is installed
_HX_LEDGER_ON = False    # hx_patch rewrites this line when the HX2 ledger is installed
_HX_STALL_ON = False     # hx_patch rewrites this line when the Stage 1 stall intervention is installed
_HX_RG = None
_HX_LAST_STATE_OF = None


def _hx_stable_value(v):
    """Canonicalize structured feature values before sandbox/host comparison."""
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, (str, int, float, type(None))):
        return v
    if isinstance(v, (set, frozenset)):
        return ("set", tuple(sorted((_hx_stable_value(x) for x in v), key=repr)))
    if isinstance(v, dict):
        return ("dict", tuple(sorted(((str(k), _hx_stable_value(x)) for k, x in v.items()),
                                     key=lambda pair: pair[0])))
    if isinstance(v, (tuple, list)):
        return (type(v).__name__, tuple(_hx_stable_value(x) for x in v))
    try:
        return _hx_stable_value(v.tolist())
    except (AttributeError, TypeError, ValueError):
        return repr(v)


def _hx_obs_dict(s):
    if s is None:
        return None
    if isinstance(s, dict):
        return {str(k): v if isinstance(v, (str, int, float, bool, type(None)))
                else repr(_hx_stable_value(v)) for k, v in s.items()}
    if isinstance(s, (tuple, list)):
        return {f"f{i}": repr(_hx_stable_value(v)) if not isinstance(v, (str, int, float, bool, type(None))) else v
                for i, v in enumerate(s)}
    return {"state": repr(_hx_stable_value(s)) if not isinstance(s, (str, int, float, bool)) else s}


def _hx_interior_changed(t):
    r = getattr(t, "result", None)
    if isinstance(r, dict) and "gameplay_changed" in r:
        return bool(r["gameplay_changed"])
    b = _hx_ascii(getattr(t, "before_frame", None))
    a = _hx_ascii(getattr(t, "after_frame", None))
    if isinstance(b, str) and isinstance(a, str) and "\n" in b and "\n" in a:
        br, ar = b.split("\n"), a.split("\n")
        h, w = len(br), len(br[0]) if br else 0
        if h > 8 and w > 8 and len(ar) == h:
            return any(br[r][4:w - 4] != ar[r][4:w - 4] for r in range(4, h - 4))
        return b != a
    return False


def _hx_delta_val(b, a):
    if isinstance(b, (int, float)) and not isinstance(b, bool) and isinstance(a, (int, float)) and not isinstance(a, bool):
        return round(a - b, 4) if isinstance(a, float) or isinstance(b, float) else a - b
    if isinstance(b, (tuple, list)) and isinstance(a, (tuple, list)) and len(b) == len(a) and 1 <= len(b) <= 8:
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in b) and all(
                isinstance(x, (int, float)) and not isinstance(x, bool) for x in a):
            return tuple(round(y - x, 4) if isinstance(x, float) or isinstance(y, float) else y - x for x, y in zip(b, a))
    if isinstance(b, (set, frozenset)) and isinstance(a, (set, frozenset)) and len(b) == len(a) and 1 <= len(b) <= 128:
        try:
            sb = sorted(b)
            sa = sorted(a)
            if all(isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], (int, float)) and isinstance(p[1], (int, float)) for p in sb) and all(
                    isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], (int, float)) and isinstance(p[1], (int, float)) for p in sa):
                dr, dc = sa[0][0] - sb[0][0], sa[0][1] - sb[0][1]
                if all((p[0] + dr, p[1] + dc) == q for p, q in zip(sb, sa)):
                    return ("translate", (dr, dc))
        except Exception:
            pass
    return None


def _hx_state_diagnostics_and_invariants(moves, state_of, s0, acts):
    diag = {
        "moves_checked": 0,
        "non_noop_moves": 0,
        "markov_conflicts": [],
        "blind_gameplay_moves": [],
        "untested_actions_from_s0": [],
    }
    inv = {"action_deltas": {}, "relations": []}
    seen_trans = {}
    action_records = {}
    seq_records = []
    visited_keys = {_hx_stable_value(s0)}
    k0 = _hx_stable_value(s0)
    tested_from_s0 = set()
    for idx, t in enumerate(moves):
        try:
            sb = state_of(t.before_frame)
            sa = state_of(t.after_frame)
        except Exception:
            continue
        diag["moves_checked"] += 1
        kb = _hx_stable_value(sb)
        ka = _hx_stable_value(sa)
        visited_keys.add(kb)
        visited_keys.add(ka)
        act = str(t.action)
        step_idx = getattr(t.after_frame, "step", idx + 1)
        if kb == k0:
            tested_from_s0.add(act)
        if kb != ka:
            diag["non_noop_moves"] += 1
        elif _hx_interior_changed(t) and len(diag["blind_gameplay_moves"]) < 4:
            diag["blind_gameplay_moves"].append({"step": step_idx, "action": act})
        prev = seen_trans.get((kb, act))
        if prev is not None and prev[0] != ka and len(diag["markov_conflicts"]) < 3:
            diag["markov_conflicts"].append({
                "action": act,
                "steps": [prev[1], step_idx],
                "prev_after": repr(prev[2])[:80],
                "after": repr(sa)[:80],
            })
        else:
            seen_trans[(kb, act)] = (ka, step_idx, sa)
        action_records.setdefault(act, []).append((sb, sa, kb, ka))
        seq_records.append((act, sb, sa, kb, ka))
    diag["untested_actions_from_s0"] = [str(a) for a in acts if str(a) not in tested_from_s0]
    if isinstance(s0, dict):
        for k in list(s0.keys())[:10]:
            f_deltas = {}
            for act, recs in action_records.items():
                ds = []
                ok = True
                for sb, sa, _, _ in recs:
                    if not isinstance(sb, dict) or not isinstance(sa, dict) or k not in sb or k not in sa:
                        ok = False
                        break
                    d = _hx_delta_val(sb[k], sa[k])
                    if d is None:
                        ok = False
                        break
                    ds.append(d)
                if ok and ds:
                    uniq = []
                    for d in ds:
                        if d not in uniq:
                            uniq.append(d)
                    f_deltas[act] = uniq[0] if len(uniq) == 1 else {"consistent": False, "observed": uniq[:3]}
            if f_deltas:
                inv["action_deltas"][str(k)] = f_deltas
    else:
        s_deltas = {}
        for act, recs in action_records.items():
            ds = [_hx_delta_val(sb, sa) for sb, sa, _, _ in recs]
            if ds and all(d is not None for d in ds):
                uniq = []
                for d in ds:
                    if d not in uniq:
                        uniq.append(d)
                s_deltas[act] = uniq[0] if len(uniq) == 1 else {"consistent": False, "observed": uniq[:3]}
        if s_deltas:
            inv["action_deltas"]["state"] = s_deltas
    for act, recs in action_records.items():
        if recs and all(kb == ka for _, _, kb, ka in recs):
            inv["relations"].append({"rel": "noop", "word": [act], "support": len(recs)})
    pair_counts = _hxc.Counter()
    pair_Viol = set()
    for i in range(len(seq_records) - 1):
        a1, _, _, kb1, ka1 = seq_records[i]
        a2, _, _, kb2, ka2 = seq_records[i + 1]
        if ka1 != kb2:
            continue
        if kb1 == ka2 and kb1 != ka1:
            pair_counts[("inverse", a1, a2)] += 1
        elif kb1 != ka2:
            pair_Viol.add(("inverse", a1, a2))
        if a1 == a2:
            if ka1 == ka2 and kb1 != ka1:
                pair_counts[("idempotent", a1, a2)] += 1
            elif ka1 != ka2:
                pair_Viol.add(("idempotent", a1, a2))
            if kb1 == ka2 and kb1 != ka1:
                pair_counts[("involution", a1, a2)] += 1
            elif kb1 != ka2:
                pair_Viol.add(("involution", a1, a2))
    for (rel, a1, a2), cnt in pair_counts.items():
        if (rel, a1, a2) not in pair_Viol and len(inv["relations"]) < 8:
            inv["relations"].append({"rel": rel, "word": [a1, a2], "support": cnt})
    return diag, inv, visited_keys, diag["non_noop_moves"]


def _hx_act_names(actions):
    items = [actions] if isinstance(actions, (str, dict)) else list(actions or ())
    out = []
    for it in items:
        if isinstance(it, str):
            out.append(it.strip())
        elif isinstance(it, dict):
            out.append(str(it.get("action", "")).strip())
    return [x for x in out if x]


def _hx_kind(entry):
    r = getattr(entry, "result", None) or {}
    act = str(getattr(entry, "action", "") or "")
    if r.get("action_name") == "RESET" or act == "RESET":
        return "reset"
    if r.get("run_complete") or r.get("done"):
        return "won"
    if r.get("level_completed"):
        return "level_up"
    if r.get("game_over"):
        return "death"
    return "move"


def _hx_refresh(rg):
    """Annotate transitions with kind/level, build `moves`, guard frame_diff's default."""
    global _HX_RG
    _HX_RG = rg
    trs = rg.get("transitions") or []
    for t in trs:
        try:
            t.kind = _hx_kind(t)
            t.level = getattr(t.after_frame, "level", None)
            if t.kind == "move" and t.before_frame is not None and getattr(t.before_frame, "level", None) != t.level:
                t.kind = "level_up"            # defensive: never a move across levels
        except Exception:
            pass
    cur = rg.get("current_frame")
    lvl = getattr(cur, "level", None)
    rg["moves"] = [t for t in trs if getattr(t, "kind", "") == "move" and getattr(t, "level", None) == lvl
                   and t.before_frame is not None]
    fd = rg.get("frame_diff")
    if fd is not None and not getattr(fd, "_hx", False):
        last = trs[-1] if trs else None

        def frame_diff(before=None, after=None, _fd=fd, _last=last):
            if before is None and after is None and _last is not None and getattr(_last, "kind", "move") != "move":
                return {"note": "the last action was a '%s': previous_frame and current_frame are not one game "
                                "transition. Pass frames explicitly, e.g. frame_diff(moves[-1].before_frame, "
                                "moves[-1].after_frame)." % _last.kind}
            return _fd(before, after)
        frame_diff._hx = True
        rg["frame_diff"] = frame_diff
    if _HX_TOOLS_ON:
        for n in HX_NAMES:
            f = globals().get(n)
            if f is not None:
                rg[n] = f
    if _HX_LEDGER_ON:
        rg["hx_ledger"] = hx_ledger
    act_fn = rg.get("action")
    if (_HX_TOOLS_ON or _HX_STALL_ON) and act_fn is not None and not getattr(act_fn, "_hx_wrapped", False):
        def _hx_action_wrapper(actions, _inner=act_fn, _rg=rg):
            names = _hx_act_names(actions)
            has_host = callable(globals().get("_send")) and callable(globals().get("_recv"))
            if _HX_STALL_ON and has_host:
                try:
                    chk = _hx_call("before_action", actions=names)
                except Exception:
                    chk = {}
                if chk.get("require_probe"):
                    err_cls = _rg.get("KnownNoOpActionError") or RuntimeError
                    raise err_cls(str(chk.get("message") or "call hx_probe before action() on a stall turn"))
            obs_fn = _HX_LAST_STATE_OF or _rg.get("state_of") or _rg.get("observe")
            before_obs = None
            if callable(obs_fn) and _rg.get("current_frame") is not None:
                try:
                    before_obs = _hx_obs_dict(obs_fn(_rg.get("current_frame")))
                except Exception:
                    before_obs = None
            prev_act = _rg.get("last_action")
            res = _inner(actions)
            if has_host:
                after_obs = None
                if callable(obs_fn) and _rg.get("current_frame") is not None:
                    try:
                        after_obs = _hx_obs_dict(obs_fn(_rg.get("current_frame")))
                    except Exception:
                        after_obs = None
                try:
                    _hx_call("action_logged", actions=names, before_obs=before_obs, after_obs=after_obs,
                             prev_action=prev_act,
                             level_completed=bool(isinstance(res, dict) and (
                                 res.get("level_completed") or res.get("run_complete") or res.get("done"))))
                except Exception:
                    pass
            return res
        _hx_action_wrapper._hx_wrapped = True
        rg["action"] = _hx_action_wrapper


def _hx_ascii(x):
    if hasattr(x, "ascii"):
        return x.ascii
    return x


def _hx_moves(level=None):
    g = globals().get("_HX_RG") or {}
    trs = g.get("transitions") or []
    if level is None:
        return list(g.get("moves") or [])
    return [t for t in trs if getattr(t, "kind", "") == "move" and getattr(t, "level", None) == level]


def hx_check(predict, observe=None, level=None, limit=5, time_limit=_HX_TL):
    t_end = _hxtime.time() + float(time_limit)
    obs_fn = observe or _hx_ascii
    ms = _hx_moves(level)
    seen = {}
    out = {"checked": 0, "correct": 0, "mismatches": [], "errors": {}}
    for t in ms:
        if _hxtime.time() > t_end:
            out["stopped"] = "time_limit"
            break
        try:
            pred = predict(t.before_frame, t.action)
        except Exception as exc:  # noqa: BLE001
            k = type(exc).__name__ + ": " + str(exc)[:120]
            out["errors"][k] = out["errors"].get(k, 0) + 1
            continue
        try:
            got = obs_fn(t.after_frame)
            key = (repr(obs_fn(t.before_frame)), t.action)
        except Exception as exc:  # noqa: BLE001
            k = "observe: " + type(exc).__name__ + ": " + str(exc)[:120]
            out["errors"][k] = out["errors"].get(k, 0) + 1
            continue
        out["checked"] += 1
        prev = seen.get(key)
        seen.setdefault(key, repr(got))
        if _hx_ascii(pred) == got or pred == got:
            out["correct"] += 1
            continue
        if len(out["mismatches"]) >= limit:
            continue
        mm = {"step": getattr(t.after_frame, "step", None), "action": t.action,
              "predicted": repr(_hx_ascii(pred))[:160], "observed": repr(got)[:160]}
        p, g = _hx_ascii(pred), got
        cause = "dynamics"
        if isinstance(p, str) and isinstance(g, str):
            pr, gr = p.split("\n"), g.split("\n")
            cells = [(r, c, pr[r][c], gr[r][c]) for r in range(min(len(pr), len(gr)))
                     for c in range(min(len(pr[r]), len(gr[r]))) if pr[r][c] != gr[r][c]]
            mm["cells"] = cells[:5]
            mm["n_cells"] = len(cells)
            h, w = len(gr), len(gr[0]) if gr else 0
            if cells and all(r < 4 or c < 4 or r >= h - 4 or c >= w - 4 for r, c, _, _ in cells):
                cause = "border"
        if prev is not None and prev != repr(got):
            cause = "hidden_state"
        mm["cause"] = cause
        out["mismatches"].append(mm)
    out["accuracy"] = round(out["correct"] / out["checked"], 3) if out["checked"] else None
    return out


def _hx_plan(step, s0, goal, actions, max_nodes, t_end, key=_hx_stable_value):
    """BFS shortest plan in one model; returns (plan or None, exhausted)."""
    try:
        if goal(s0):
            return [], False
    except Exception:  # noqa: BLE001
        return None, False
    seen = {key(s0)}
    q = _hxc.deque([(s0, [])])
    nodes = 0
    while q:
        if nodes >= max_nodes or _hxtime.time() > t_end:
            return None, False
        s, path = q.popleft()
        for a in actions:
            nodes += 1
            try:
                ns = step(s, a)
            except Exception:  # noqa: BLE001
                ns = None
            if ns is None:
                continue
            k = key(ns)
            if k in seen:
                continue
            seen.add(k)
            try:
                ok = goal(ns)
            except Exception:  # noqa: BLE001
                ok = False
            if ok:
                return path + [a], False
            q.append((ns, path + [a]))
    return None, True


def _hx_run(step, s, word):
    for a in word:
        try:
            s = step(s, a)
        except Exception:  # noqa: BLE001
            return None
        if s is None:
            return None
    return s


def _hx_solves(step, s, goal, plan):
    for a in plan:
        try:
            s = step(s, a)
            if s is None:
                return False
            if goal(s):
                return True
        except Exception:  # noqa: BLE001
            return False
    try:
        return bool(goal(s)) if not plan else False
    except Exception:  # noqa: BLE001
        return False


def hx_probe(models, goals, actions=None, depth=2, weights=None, cost_per_action=0.02, max_nodes=4000,
             max_probes=60, time_limit=_HX_TL, key=_hx_stable_value, state_of=None):
    """Decision-valued probe selection over hypotheses (model x goal); see HX_DOC."""
    if state_of is not None:
        return _hx_probe_live(models, goals, actions, depth, weights, cost_per_action,
                              max_nodes, max_probes, time_limit, key, state_of)
    t_end = _hxtime.time() + float(time_limit)
    w = weights or {}
    hyps = [(m, g) for m in models for g in goals]
    wt = {h: float(w.get(h[0], 1.0)) * float(w.get(h[1], 1.0)) for h in hyps}
    W = sum(wt.values()) or 1.0
    actions = list(actions or ())
    plans = {}
    exhausted_map = {}
    for (m, g) in hyps:
        s0, step = models[m]
        pl, ex = _hx_plan(step, s0, goals[g], actions, max_nodes, t_end, key)
        plans[(m, g)] = pl
        exhausted_map[(m, g)] = bool(ex)

    def value(plan, hs, offset=0):
        tot = sum(wt[h] for h in hs) or 1.0
        ok = sum(wt[h] for h in hs if _hx_solves(models[h[0]][1], models[h[0]][0], goals[h[1]], plan))
        return ok / tot - cost_per_action * (len(plan) + offset)

    out = {
        "plans": {f"{m}|{g}": p for (m, g), p in plans.items()},
        "exhausted_plans": [f"{m}|{g}" for (m, g), ex in exhausted_map.items() if ex],
        "groups": {},
        "gain": 0.0,
    }
    cands = [p for p in plans.values() if p]
    if not cands and not any(p == [] for p in plans.values()):
        all_exhausted = bool(exhausted_map) and all(exhausted_map.values())
        note = ("no model reaches any goal (reachable state space exhausted)"
                if all_exhausted else "no model reaches any goal within the search budget")
        out.update(decision="none", action=None, probe=None, note=note)
        if len(models) >= 2 and actions:
            m_wt = {m: float(w.get(m, 1.0)) for m in models}
            m_W = sum(m_wt.values()) or 1.0
            best_dist = None
            words = []
            for d in range(1, int(depth) + 1):
                for wd in _hxit.product(actions, repeat=d):
                    words.append(list(wd))
                    if len(words) >= max_probes:
                        break
            for q in words:
                if _hxtime.time() > t_end:
                    break
                m_cells = {}
                for m_name, (ms0, mstep) in models.items():
                    s = _hx_run(mstep, ms0, q)
                    m_cells.setdefault(key(s) if s is not None else None, []).append(m_name)
                if len(m_cells) < 2:
                    continue
                ent = 0.0
                for ms in m_cells.values():
                    p_cell = sum(m_wt[m] for m in ms) / m_W
                    if p_cell > 0:
                        ent -= p_cell * _hxmath.log2(p_cell)
                score = ent - cost_per_action * len(q)
                if best_dist is None or score > best_dist[0]:
                    best_dist = (score, ent, q, m_cells)
            if best_dist is not None:
                out["distinguishing_probe"] = best_dist[2]
                out["distinguishing_groups"] = {repr(o)[:80]: ms for o, ms in best_dist[3].items()}
                out["partition_entropy"] = round(best_dist[1], 3)
        return out
    firsts = {tuple(p[:1]) for p in plans.values() if p is not None}
    if len(firsts) == 1 and all(p is not None for p in plans.values()):
        first = next(iter(firsts))
        out.update(decision="act", action=first[0] if first else None, probe=None)
        return out
    best_now = max(cands, key=lambda p: value(p, hyps)) if cands else []
    v_now = value(best_now, hyps) if cands else 0.0
    words = []
    for d in range(1, int(depth) + 1):
        for wd in _hxit.product(actions, repeat=d):
            words.append(list(wd))
            if len(words) >= max_probes:
                break
    for p in cands:
        for d in range(1, min(len(p), int(depth)) + 1):
            if p[:d] not in words:
                words.append(p[:d])
    best = None
    for q in words:
        if _hxtime.time() > t_end:
            out["stopped"] = "time_limit"
            break
        cells = {}
        for h in hyps:
            s = _hx_run(models[h[0]][1], models[h[0]][0], q)
            cells.setdefault(key(s) if s is not None else None, []).append(h)
        if len(cells) < 2:
            continue
        ev = 0.0
        for o, hs in cells.items():
            pw = sum(wt[h] for h in hs) / W
            best_c = -1e9
            for h in hs:
                s = _hx_run(models[h[0]][1], models[h[0]][0], q)
                if s is None:
                    continue
                pl, _ = _hx_plan(models[h[0]][1], s, goals[h[1]], actions, max_nodes // 4, t_end, key)
                if pl is None:
                    continue
                full = list(q) + pl
                best_c = max(best_c, value(full, hs))
            ev += pw * (best_c if best_c > -1e9 else -cost_per_action * len(q))
        gain = ev - v_now
        if best is None or gain > best[0]:
            best = (gain, q, cells)
    if best is not None and best[0] > 0:
        out.update(decision="probe", probe=best[1], action=best[1][0], gain=round(best[0], 3),
                   groups={repr(o)[:80]: [f"{m}|{g}" for m, g in hs] for o, hs in best[2].items()})
    else:
        out.update(decision="follow", action=best_now[0] if best_now else None, probe=None,
                   gain=round(best[0], 3) if best else 0.0, plan=best_now)
    return out


def _hx_probe_live(models, goals, actions, depth, weights, cost_per_action,
                   max_nodes, max_probes, time_limit, key, state_of):
    """Turn frame-based candidate functions into checked, precommitted plans."""
    global _HX_LAST_STATE_OF
    if not callable(state_of) or not isinstance(models, dict) or not isinstance(goals, dict):
        return {"decision": "none", "action": None, "error": "supply state_of and model/goal dictionaries"}
    rg = _HX_RG or {}
    frame = rg.get("current_frame")
    if frame is None:
        return {"decision": "none", "action": None, "error": "current frame unavailable"}
    try:
        s0 = state_of(frame)
    except Exception as exc:
        return {"decision": "none", "action": None, "error": "state_of: " + type(exc).__name__ + ": " + str(exc)[:120]}
    _HX_LAST_STATE_OF = state_of
    acts = list(actions or rg.get("valid_actions") or ())
    if not acts:
        return {"decision": "none", "action": None, "error": "no valid actions"}
    moves = list(rg.get("moves") or [])[-24:]
    diag, inv, visited_keys, non_noop_moves = _hx_state_diagnostics_and_invariants(moves, state_of, s0, acts)
    checked = {}
    forecasts = {}
    usable = {}
    executable = 0
    derived_weights = {}

    def mismatch_example(action, predicted, observed):
        fields = []
        if isinstance(predicted, dict) and isinstance(observed, dict):
            for field in list(dict.fromkeys(list(predicted) + list(observed))):
                p, o = predicted.get(field), observed.get(field)
                if _hx_stable_value(p) != _hx_stable_value(o):
                    fields.append({"field": str(field)[:40], "predicted": repr(p)[:80],
                                   "observed": repr(o)[:80]})
                if len(fields) == 3:
                    break
        else:
            fields = [{"field": "state", "predicted": repr(predicted)[:80],
                       "observed": repr(observed)[:80]}]
        return {"action": str(action), "fields": fields}

    for name, step in list(models.items())[:6]:
        if not callable(step):
            continue
        correct = wrong = abstained = non_noop_correct = 0
        examples = []
        errors = []
        for t in moves:
            try:
                sb = state_of(t.before_frame)
                observed = state_of(t.after_frame)
                is_non_noop = _hx_stable_value(sb) != _hx_stable_value(observed)
                predicted = step(sb, t.action)
                if predicted is None:
                    abstained += 1
                    if len(errors) < 2:
                        errors.append({"action": str(t.action), "error": "step returned None"})
                elif _hx_stable_value(predicted) == _hx_stable_value(observed) or predicted == observed:
                    correct += 1
                    if is_non_noop:
                        non_noop_correct += 1
                else:
                    wrong += 1
                    if len(examples) < 2:
                        examples.append(mismatch_example(t.action, predicted, observed))
            except Exception as exc:
                abstained += 1
                if len(errors) < 2:
                    errors.append({"action": str(t.action), "error":
                                   (type(exc).__name__ + ": " + str(exc))[:120]})
        fc = {}
        for act in acts:
            try:
                pred = step(s0, act)
                if pred is not None:
                    fc[str(act)] = _hx_obs_dict(pred)
            except Exception as exc:
                if len(errors) < 2:
                    errors.append({"action": str(act), "error":
                                   (type(exc).__name__ + ": " + str(exc))[:120]})
        observed_n = correct + wrong
        if not moves and observed_n == 0:
            supported = True
        elif non_noop_moves > 0:
            supported = bool(correct > 0 and correct / observed_n >= 0.6 and non_noop_correct >= 1)
        else:
            supported = bool(correct > 0 and correct / observed_n >= 0.6)
        checked[str(name)] = {
            "correct": correct,
            "wrong": wrong,
            "abstained": abstained,
            "non_noop_correct": non_noop_correct,
            "current_actions_predicted": len(fc),
            "supported": bool(supported),
            "mismatch_examples": examples,
            "abstention_examples": errors,
        }
        if fc:
            executable += 1
            if supported:
                usable[str(name)] = (s0, step)
                forecasts[str(name)] = fc
                derived_weights[str(name)] = max(0.05, (1 + correct) / (2 + observed_n))

    non_win_states = []
    seen_nw = set()
    for t in moves:
        for fr in (t.before_frame, t.after_frame):
            try:
                st = state_of(fr)
                kst = _hx_stable_value(st)
                if kst not in seen_nw:
                    seen_nw.add(kst)
                    non_win_states.append(st)
            except Exception:
                pass
    k0 = _hx_stable_value(s0)
    if k0 not in seen_nw:
        seen_nw.add(k0)
        non_win_states.append(s0)

    checked_goals = {}
    valid_goals = {}
    for gname, fn in list(goals.items())[:6]:
        if not callable(fn):
            continue
        g_key = str(gname)
        falsified = False
        true_on_s0 = False
        g_errors = []
        for st in non_win_states:
            try:
                val = bool(fn(st))
                if val:
                    falsified = True
                    if _hx_stable_value(st) == k0:
                        true_on_s0 = True
            except Exception as exc:
                if len(g_errors) < 2:
                    g_errors.append((type(exc).__name__ + ": " + str(exc))[:120])
        g_supported = (not falsified) and (not g_errors)
        checked_goals[g_key] = {
            "supported": bool(g_supported),
            "falsified_on_non_win": bool(falsified),
            "true_on_s0": bool(true_on_s0),
            "errors": g_errors,
        }
        if g_supported:
            valid_goals[g_key] = fn

    valid_count = len(usable)
    any_falsified = any(
        c["wrong"] > 0 or (non_noop_moves > 0 and c.get("non_noop_correct", 0) == 0)
        for k, c in checked.items() if k not in usable
    )
    sole_supported = False
    if valid_count == 1:
        sole_info = checked[next(iter(usable))]
        sole_supported = bool(
            sole_info["correct"] >= 2
            or (sole_info.get("non_noop_correct", 0) >= 1 and any_falsified)
        )
    if (valid_count >= 2 or sole_supported) and valid_goals:
        out = hx_probe(usable, valid_goals, acts, depth=depth,
                       weights=weights or derived_weights, cost_per_action=cost_per_action,
                       max_nodes=max_nodes, max_probes=max_probes, time_limit=time_limit, key=key)
    else:
        if not valid_goals:
            note = "all candidate goals were falsified on observed non-winning states (or errored); define a goal predicate that is False on current/past non-winning frames"
        else:
            note = "need two supported models, or one verified survivor (with >=2 correct moves or >=1 non-noop move against a falsified alternative), and one supported goal predicate"
        out = {"decision": "none", "action": None, "probe": None, "plans": {}, "note": note}

    if out.get("decision") == "none" and usable:
        best_m = max(usable.keys(), key=lambda m: derived_weights.get(m, 1.0))
        ms0, mstep = usable[best_m]
        for d in range(1, int(depth) + 1):
            found_fp = None
            for wd in _hxit.product(acts, repeat=d):
                sn = _hx_run(mstep, ms0, wd)
                if sn is not None and key(sn) not in visited_keys:
                    found_fp = list(wd)
                    break
            if found_fp is not None:
                out["frontier_probe"] = found_fp
                break

    out["checked_models"] = checked
    out["checked_goals"] = checked_goals
    out["state_diagnostics"] = diag
    out["empirical_invariants"] = inv
    out["executable_models"] = executable
    out["valid_models"] = valid_count
    out["single_supported_model"] = bool(sole_supported)
    try:
        _hx_call("probe_logged", decision=out.get("decision"), action=out.get("action"),
                 probe=out.get("probe"), plan=out.get("plan"), valid_models=valid_count,
                 single_supported_model=bool(sole_supported),
                 weights=weights or derived_weights, forecasts_from_s0=forecasts)
    except Exception:
        pass
    return out


def _hx_call(method, **kwargs):
    _send({"type": "hx", "request": {"method": str(method), "args": kwargs}})
    reply = _recv()
    if reply.get("type") != "hx_result":
        raise RuntimeError("Invalid hx response from sandbox host.")
    res = reply.get("result") or {}
    if isinstance(res, dict) and res.get("error"):
        raise RuntimeError("hx: " + str(res["error"]))
    return res


def hx_goals(limit=8):
    return _hx_call("goals", limit=limit)


def hx_relations(min_support=2, limit=20):
    return _hx_call("relations", min_support=min_support, limit=limit)


def hx_ledger(limit=6):
    return _hx_call("ledger", limit=limit)


def hx_wins():
    res = _hx_call("wins")
    for w in res.get("wins", []):
        for k in ("win_grid", "start_grid"):
            g = w.get(k)
            if g:
                w[k.replace("_grid", "_ascii")] = "\n".join("".join(COLOR_CHARS[max(0, min(15, int(v)))] for v in row)
                                                            for row in g)
    return res
