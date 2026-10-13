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
    "  - `hx_goals()` -> {'candidates', 'false_now', 'true_now', 'hazards', 'last_win'}: facts that became true "
    "when earlier levels were completed (evidence from the actual win frames, not a verified goal) and whether "
    "each is true on the current board. `hx_relations()` -> {'relations', 'conflicts'}: action relations "
    "verified on this level (no effect, undo, order irrelevant, ...) with the number of states supporting them "
    "and counterexamples; a relation holds only in its context. `hx_wins()` -> {'wins'}: per completed level, "
    "the winning action and the board at the moment of the win.\n"
)

_HX_TL = 8.0
_HX_TOOLS_ON = True      # hx_patch rewrites this line when only the events layer is installed
_HX_RG = None


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


def _hx_plan(step, s0, goal, actions, max_nodes, t_end, key=repr):
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


def hx_probe(models, goals, actions, depth=2, weights=None, cost_per_action=0.02, max_nodes=4000,
             max_probes=60, time_limit=_HX_TL, key=repr):
    """Decision-valued probe selection over hypotheses (model x goal); see HX_DOC."""
    t_end = _hxtime.time() + float(time_limit)
    w = weights or {}
    hyps = [(m, g) for m in models for g in goals]
    wt = {h: float(w.get(h[0], 1.0)) * float(w.get(h[1], 1.0)) for h in hyps}
    W = sum(wt.values()) or 1.0
    actions = list(actions)
    plans = {}
    for (m, g) in hyps:
        s0, step = models[m]
        plans[(m, g)] = _hx_plan(step, s0, goals[g], actions, max_nodes, t_end, key)[0]

    def value(plan, hs, offset=0):
        tot = sum(wt[h] for h in hs) or 1.0
        ok = sum(wt[h] for h in hs if _hx_solves(models[h[0]][1], models[h[0]][0], goals[h[1]], plan))
        return ok / tot - cost_per_action * (len(plan) + offset)

    out = {"plans": {f"{m}|{g}": p for (m, g), p in plans.items()}, "groups": {}, "gain": 0.0}
    cands = [p for p in plans.values() if p]
    if not cands and not any(p == [] for p in plans.values()):
        out.update(decision="none", action=None, probe=None,
                   note="no model reaches any goal within the search budget")
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


def hx_wins():
    res = _hx_call("wins")
    for w in res.get("wins", []):
        for k in ("win_grid", "start_grid"):
            g = w.get(k)
            if g:
                w[k.replace("_grid", "_ascii")] = "\n".join("".join(COLOR_CHARS[max(0, min(15, int(v)))] for v in row)
                                                            for row in g)
    return res
