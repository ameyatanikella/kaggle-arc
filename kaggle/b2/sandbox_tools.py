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




def fit_boolean(rows, values, max_degree=2, max_terms=4):
    """Sparsest exact multilinear fit y = sum_S theta_S prod_{i in S} b_i over observed
    Boolean configurations `rows` (lists of 0/1). Smallest support first (MDL). Returns
    {'terms': {(i,..): coef}, 'alternatives': n_other_supports_same_size,
     'unidentified': [monomials not pinned by the data]} or None if no fit <= max_terms.
    Several alternatives = choose a configuration that separates them before trusting it."""
    from fractions import Fraction as _F
    n = len(rows[0]) if rows else 0
    monos = [()]
    for d in range(1, max_degree + 1):
        monos += list(_b2it.combinations(range(n), d))
    feat = [[int(all(r[i] for i in m)) for m in monos] for r in rows]
    y = [_F(v) for v in values]

    def solve(cols):
        A = [[_F(f[j]) for j in cols] + [yy] for f, yy in zip(feat, y)]
        piv, rr = [], 0
        for c in range(len(cols)):
            p = next((i for i in range(rr, len(A)) if A[i][c] != 0), None)
            if p is None:
                continue
            A[rr], A[p] = A[p], A[rr]
            A[rr] = [x / A[rr][c] for x in A[rr]]
            for i in range(len(A)):
                if i != rr and A[i][c] != 0:
                    A[i] = [a - A[i][c] * b for a, b in zip(A[i], A[rr])]
            piv.append(c)
            rr += 1
        if any(A[i][-1] != 0 for i in range(rr, len(A))):
            return None
        x = [_F(0)] * len(cols)
        for i, c in enumerate(piv):
            x[c] = A[i][-1]
        return x
    for k in range(0, max_terms + 1):
        sols = []
        for cols in _b2it.combinations(range(len(monos)), k):
            if k == 0:
                if all(v == 0 for v in y):
                    sols.append({})
                continue
            x = solve(cols)
            if x is not None and all(v != 0 for v in x):
                sols.append({monos[j]: (int(v) if v.denominator == 1 else float(v)) for j, v in zip(cols, x)})
        if sols:
            pinned = set()
            for j, m in enumerate(monos):
                e = [0] * len(monos)
                e[j] = 1
                if solve_rank_ok(feat, e):
                    pinned.add(m)
            return {"terms": sols[0], "alternatives": len(sols) - 1,
                    "unidentified": [m for m in monos if m not in pinned]}
    return None


def solve_rank_ok(feat, e):
    """True iff unit vector e lies in the row space of feat (exact)."""
    from fractions import Fraction as _F

    def rank(M):
        M = [[_F(v) for v in r] for r in M]
        rk = 0
        for c in range(len(M[0]) if M else 0):
            p = next((i for i in range(rk, len(M)) if M[i][c] != 0), None)
            if p is None:
                continue
            M[rk], M[p] = M[p], M[rk]
            for i in range(len(M)):
                if i != rk and M[i][c] != 0:
                    f = M[i][c] / M[rk][c]
                    M[i] = [a - f * b for a, b in zip(M[i], M[rk])]
            rk += 1
        return rk
    return bool(feat) and rank(feat + [e]) == rank(feat)


def fit_guarded(xs, ys):
    """1-D rule with a guard: y = a*x+b, or y = a*x+b below a threshold and a constant
    at/after it (wall, cap, saturation). Simplest exact rule first. Returns dict or None."""
    pts = sorted(zip(xs, ys))

    def aff(p):
        if len(p) < 2:
            return (0.0, p[0][1]) if p else None
        (x0, y0), (x1, y1) = p[0], p[-1]
        if x1 == x0:
            return None
        a = (y1 - y0) / (x1 - x0)
        b = y0 - a * x0
        return (a, b) if all(abs(a * x + b - y) < 1e-9 for x, y in p) else None
    f = aff(pts)
    if f:
        return {"kind": "affine", "a": f[0], "b": f[1]}
    for k in range(1, len(pts)):
        f = aff(pts[:k])
        if f and all(abs(y - pts[k][1]) < 1e-9 for _, y in pts[k:]):
            return {"kind": "guarded", "a": f[0], "b": f[1], "threshold": pts[k][0], "value_after": pts[k][1]}
    return None


B2_TOOL_NAMES = ("bfs_plan", "distinguish", "gf2_solve", "zm_solve", "fit_boolean", "fit_guarded")
