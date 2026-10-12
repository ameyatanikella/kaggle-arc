"""Compute-only helpers preloaded into the harness Python sandbox (B2_TOOLS).

Pure Python, no third-party imports, bounded work (node caps plus a wall-clock
cap below the sandbox's 30 s limit). None of these execute game actions.

  bfs_plan(start, step, is_goal, actions, max_nodes=200000, key=None, time_limit=20)
  distinguish(sims, starts, actions, max_depth=6, max_nodes=100000, time_limit=20)
  gf2_solve(A, b, cost=None)
  zm_solve(A, b, m, cost=None, max_states=2000000)
"""
import collections as _b2c
import itertools as _b2it
import time as _b2time


def bfs_plan(start, step, is_goal, actions, max_nodes=200000, key=None, time_limit=20.0):
    """Shortest action list from `start` to a state with is_goal(state) True, in YOUR
    deterministic simulator step(state, action) -> next_state (None = illegal/blocked).
    Returns {'plan': [...] or None, 'nodes': n, 'exhausted': bool}. exhausted=True with
    plan None means no goal is reachable *in your model*; False means a cap was hit."""
    t_end = _b2time.time() + float(time_limit)
    k = key or (lambda s: s)
    if is_goal(start):
        return {"plan": [], "nodes": 1, "exhausted": False}
    seen = {k(start)}
    q = _b2c.deque([(start, None)])
    parent = {}
    nodes = 0
    capped = False
    while q:
        s, _ = q.popleft()
        if nodes >= max_nodes or _b2time.time() > t_end:
            capped = True
            break
        for a in actions:
            ns = step(s, a)
            nodes += 1
            if ns is None:
                continue
            kn = k(ns)
            if kn in seen:
                continue
            seen.add(kn)
            parent[kn] = (k(s), a)
            if is_goal(ns):
                plan = [a]
                cur = k(s)
                while cur in parent:
                    pk, pa = parent[cur]
                    plan.append(pa)
                    cur = pk
                return {"plan": plan[::-1], "nodes": nodes, "exhausted": False}
            q.append((ns, None))
    return {"plan": None, "nodes": nodes, "exhausted": not capped}


def distinguish(sims, starts, actions, max_depth=6, max_nodes=100000, time_limit=20.0):
    """Shortest action sequence whose PREDICTED observations differ between your
    candidate simulators. sims[i](state, action) -> (next_state, observation);
    starts[i] = that simulator's current state. Returns {'sequence': [...] or None,
    'groups': {obs_repr: [sim indices]} at the split, 'exhausted': bool}. Use it to
    pick ONE cheap real probe that tells your hypotheses apart; a None result with
    exhausted=True means the candidates are indistinguishable from here (so they
    should not change your plan)."""
    t_end = _b2time.time() + float(time_limit)
    start = tuple(starts)
    q = _b2c.deque([(start, [])])
    seen = {repr(start)}
    nodes = 0
    capped = False
    while q:
        tup, path = q.popleft()
        if len(path) >= max_depth:
            capped = True
            continue
        if nodes >= max_nodes or _b2time.time() > t_end:
            capped = True
            break
        for a in actions:
            nxt, obs = [], []
            for sim, s in zip(sims, tup):
                try:
                    ns, o = sim(s, a)
                except Exception as e:  # a crashing hypothesis is itself informative
                    ns, o = ("<error>", "<error:%s>" % type(e).__name__)
                nxt.append(ns)
                obs.append(repr(o))
            nodes += 1
            if len(set(obs)) > 1:
                groups = {}
                for i, o in enumerate(obs):
                    groups.setdefault(o[:120], []).append(i)
                return {"sequence": path + [a], "groups": groups, "exhausted": False}
            r = repr(tuple(nxt))
            if r not in seen:
                seen.add(r)
                q.append((tuple(nxt), path + [a]))
    return {"sequence": None, "groups": None, "exhausted": not capped}


def gf2_solve(A, b, cost=None):
    """Minimum-cost u in {0,1}^n with A u = b (mod 2): A[i][j] = 1 if pressing j
    toggles cell i; b[i] = 1 if cell i must change. cost[j] = actions to press j
    (include travel). Exact when nullity <= 20. Returns {'presses': [j...],
    'cost', 'exact', 'n_solutions'} or None if impossible (then the toggle model is wrong)."""
    rows = [[int(x) & 1 for x in r] + [int(b[i]) & 1] for i, r in enumerate(A)]
    n = len(A[0]) if A else 0
    piv = []
    r = 0
    for c in range(n):
        p = next((i for i in range(r, len(rows)) if rows[i][c]), None)
        if p is None:
            continue
        rows[r], rows[p] = rows[p], rows[r]
        for i in range(len(rows)):
            if i != r and rows[i][c]:
                rows[i] = [x ^ y for x, y in zip(rows[i], rows[r])]
        piv.append(c)
        r += 1
    if any(rows[i][n] for i in range(r, len(rows))):
        return None
    x0 = [0] * n
    for i, c in enumerate(piv):
        x0[c] = rows[i][n]
    free = [j for j in range(n) if j not in piv]
    basis = []
    for f in free:
        v = [0] * n
        v[f] = 1
        for i, c in enumerate(piv):
            v[c] = rows[i][f]
        basis.append(v)
    w = list(cost) if cost is not None else [1] * n
    def c_of(x):
        return sum(wi for wi, xi in zip(w, x) if xi)
    best = x0
    exact = len(basis) <= 20
    if exact:
        for mask in range(1, 2 ** len(basis)):
            x = x0[:]
            for j in range(len(basis)):
                if mask >> j & 1:
                    x = [a ^ c for a, c in zip(x, basis[j])]
            if c_of(x) < c_of(best):
                best = x
    else:
        improved = True
        while improved:
            improved = False
            for v in basis:
                cand = [a ^ c for a, c in zip(best, v)]
                if c_of(cand) < c_of(best):
                    best, improved = cand, True
    return {"presses": [j for j, v in enumerate(best) if v], "cost": c_of(best), "exact": exact,
            "n_solutions": 2 ** len(basis)}


def zm_solve(A, b, m, cost=None, max_states=2000000):
    """Minimum-cost u in {0..m-1}^n with A u = b (mod m), any m (prime or not), by
    exact enumeration (n small). u[j] = how many times to press j. Returns
    {'presses': u, 'cost'} or None if impossible."""
    n = len(A[0]) if A else 0
    if m ** n > max_states:
        raise ValueError("too many unknowns for exact search; reduce variables")
    w = list(cost) if cost is not None else [1] * n
    best = None
    for u in _b2it.product(range(m), repeat=n):
        if all(sum(a * x for a, x in zip(row, u)) % m == bi % m for row, bi in zip(A, b)):
            c = sum(wi * ui for wi, ui in zip(w, u))
            if best is None or c < best[1]:
                best = (list(u), c)
    return None if best is None else {"presses": best[0], "cost": best[1]}


B2_TOOL_NAMES = ("bfs_plan", "distinguish", "gf2_solve", "zm_solve")
