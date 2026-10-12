"""Compute-only helpers preloaded into the harness Python sandbox (B2_TOOLS).

Pure Python (stdlib only). Every helper has a wall-clock budget (default 8 s)
well below the sandbox's 30 s hard limit, so two calls fit in one snippet.
Every helper returns a dict whose keys are listed in TOOL_DOC. TOOL_DOC is the
single source of truth: b2_patches inserts it verbatim into the system prompt,
and tests/test_sandbox_tools.py checks that every documented key is returned.
None of these execute game actions.

Assumptions (see math_contract.md):
  * gf2_solve / zm_solve: linear, commuting presses; one press of j adds column j
    (mod 2 / mod m); costs are ADDITIVE per press. Order-dependent travel cost is
    not modelled.
  * fit_boolean: inputs are exactly 0/1; model class = multilinear polynomials of
    degree <= max_degree. Sparsest exact fit is an MDL-style preference, not proof.
  * fit_guarded: 1-D, one guard; the guard position is identified only up to an
    interval, which is reported.
  * bfs_plan / distinguish: deterministic simulators written by the model; results
    hold for the model, not for the real game. "exhausted" proves a negative only
    for the finite modelled state space.
"""
import collections as _b2c
import itertools as _b2it
import time as _b2time

TOOL_DOC = (
    "- Preloaded compute-only helpers (no game actions, no action cost; each stops after `time_limit` "
    "seconds, default 8). Each returns a dict:\n"
    "  - `bfs_plan(start, step, is_goal, actions)` -> {'plan': list|None, 'exhausted': bool, 'nodes': int}: "
    "shortest action list in YOUR simulator `step(state, action) -> next_state` (return None for a blocked "
    "move; states must be hashable or pass `key=`). 'exhausted' True with plan None = unreachable in your model.\n"
    "  - `distinguish(sims, starts, actions)` -> {'sequence': list|None, 'groups': list, 'errors': dict, "
    "'exhausted': bool}: shortest action sequence whose predicted observations differ between your candidate "
    "simulators `sim(state, action) -> (next_state, obs)` (starts[i] = sim i's current state); 'groups' "
    "lists which sims predict which observation. Use it to pick one probe that separates competing "
    "hypotheses; a simulator that raises is reported in 'errors', never counted as a difference.\n"
    "  - `gf2_solve(A, b, cost=None)` -> {'presses': list, 'cost', 'exact', 'n_solutions'} or None if no "
    "solution: cheapest set of presses for a toggle/XOR puzzle; A[i][j]=1 if press j flips cell i, "
    "b[i]=1 if cell i must change (target XOR current). `zm_solve(A, b, m, cost=None)` -> {'presses': list "
    "of press counts, 'cost'} or None: the same for counters that cycle mod m (b = (target-current) mod m).\n"
    "  - `fit_boolean(rows, values)` -> {'terms': {monomial: coef}, 'alternatives': int, 'unidentified': "
    "list, 'complete': bool} or None: sparsest exact polynomial rule value = sum coef*prod(switches) over "
    "0/1 switch states (`rows` = lists of 0/1, `values` = numbers; monomial () is the constant). "
    "'unidentified' lists monomials the observed states cannot pin down - observing a state that includes "
    "them would; 'alternatives' > 0 means other equally small rules also fit; 'complete' False = the time limit "
    "cut the search ('terms' may then be None). None = no rule with at most 4 terms of degree <= 2 fits "
    "(suspect hidden state or a missing variable).\n"
    "  - `fit_guarded(xs, ys)` -> {'kind': 'affine'|'cap_high'|'cap_low', 'a', 'b', 'cap', 'guard_between'} "
    "or None: y = a*x+b, possibly held at a constant 'cap' beyond a wall/threshold; the data only locate "
    "that wall between the two x values in 'guard_between'.\n"
)


def _deadline(time_limit):
    return _b2time.time() + float(time_limit)


# ----------------------------------------------------------------------------- search
def bfs_plan(start, step, is_goal, actions, max_nodes=200000, key=None, time_limit=8.0):
    """Shortest action list to a goal state in a deterministic model (see TOOL_DOC)."""
    t_end = _deadline(time_limit)
    k = key or (lambda s: s)
    if is_goal(start):
        return {"plan": [], "exhausted": False, "nodes": 0}
    seen = {k(start)}
    parent = {}
    q = _b2c.deque([start])
    nodes = 0
    capped = False
    while q:
        if nodes >= max_nodes or _b2time.time() > t_end:
            capped = True
            break
        s = q.popleft()
        ks = k(s)
        for a in actions:
            ns = step(s, a)
            nodes += 1
            if ns is None:
                continue
            kn = k(ns)
            if kn in seen:
                continue
            seen.add(kn)
            parent[kn] = (ks, a)
            if is_goal(ns):
                plan, cur = [], kn
                while cur in parent:
                    pk, pa = parent[cur]
                    plan.append(pa)
                    cur = pk
                return {"plan": plan[::-1], "exhausted": False, "nodes": nodes}
            q.append(ns)
    return {"plan": None, "exhausted": not capped, "nodes": nodes}


def distinguish(sims, starts, actions, max_depth=6, max_nodes=100000, time_limit=8.0):
    """Shortest action sequence on which the candidate simulators predict different
    observations (BFS over the product state; see TOOL_DOC). A simulator that raises
    on an action is excluded from comparison for that branch and reported in 'errors'."""
    t_end = _deadline(time_limit)
    start = tuple(starts)
    q = _b2c.deque([(start, [])])
    seen = {repr(start)}
    nodes = 0
    capped = False
    errors = {}
    while q:
        tup, path = q.popleft()
        if len(path) >= max_depth:
            capped = True
            continue
        if nodes >= max_nodes or _b2time.time() > t_end:
            capped = True
            break
        for a in actions:
            nxt, obs, alive = [], [], []
            for i, (sim, s) in enumerate(zip(sims, tup)):
                if s is _DEAD:
                    nxt.append(_DEAD)
                    continue
                try:
                    ns, o = sim(s, a)
                except Exception as e:
                    rec = errors.setdefault(i, {"count": 0, "first": "%s after %r" % (type(e).__name__, path + [a])})
                    rec["count"] += 1
                    nxt.append(_DEAD)
                    continue
                nxt.append(ns)
                obs.append((i, repr(o)))
                alive.append(i)
            nodes += 1
            distinct = {o for _, o in obs}
            if len(distinct) > 1:
                groups = {}
                for i, o in obs:
                    groups.setdefault(o, []).append(i)
                return {"sequence": path + [a],
                        "groups": [{"obs": o[:160], "sims": ix} for o, ix in groups.items()],
                        "errors": errors, "exhausted": False}
            if len(alive) < 2:
                continue  # fewer than two working hypotheses left on this branch
            r = repr(tuple(nxt))
            if r not in seen:
                seen.add(r)
                q.append((tuple(nxt), path + [a]))
    return {"sequence": None, "groups": [], "errors": errors, "exhausted": not capped}


class _Dead:
    def __repr__(self):
        return "<dead>"


_DEAD = _Dead()


# ----------------------------------------------------------------------------- modular systems
def gf2_solve(A, b, cost=None, time_limit=8.0):
    """Minimum additive-cost u in {0,1}^n with A u = b (mod 2) (see TOOL_DOC).
    Gaussian elimination on bitmasks, then Gray-code enumeration of the solution coset
    (exact if it finishes within time_limit, else best found with exact=False)."""
    m = len(A)
    n = len(A[0]) if m else 0
    if len(b) != m or any(len(r) != n for r in A):
        raise ValueError("A must be a list of equal-length rows and len(b) == len(A)")
    w = list(cost) if cost is not None else [1] * n
    if len(w) != n:
        raise ValueError("cost must have one entry per press (column of A)")
    rows = []
    for i, r in enumerate(A):
        mask = 0
        for j, x in enumerate(r):
            if int(x) & 1:
                mask |= 1 << j
        rows.append([mask, int(b[i]) & 1])
    piv_cols, r_i = [], 0
    for c in range(n):
        p = next((i for i in range(r_i, m) if rows[i][0] >> c & 1), None)
        if p is None:
            continue
        rows[r_i], rows[p] = rows[p], rows[r_i]
        for i in range(m):
            if i != r_i and rows[i][0] >> c & 1:
                rows[i][0] ^= rows[r_i][0]
                rows[i][1] ^= rows[r_i][1]
        piv_cols.append(c)
        r_i += 1
    if any(rows[i][1] for i in range(r_i, m)):
        return None
    x0 = 0
    for i, c in enumerate(piv_cols):
        if rows[i][1]:
            x0 |= 1 << c
    pivset = set(piv_cols)
    basis = []
    for f in range(n):
        if f in pivset:
            continue
        v = 1 << f
        for i, c in enumerate(piv_cols):
            if rows[i][0] >> f & 1:
                v |= 1 << c
        basis.append(v)

    def cost_of(x):
        return sum(w[j] for j in range(n) if x >> j & 1)
    best_x, best_c = x0, cost_of(x0)
    k = len(basis)
    t_end = _deadline(time_limit)
    exact = True
    x, cx = x0, best_c
    for i in range(1, 1 << k):
        if (i & 0xFFF) == 0 and _b2time.time() > t_end:
            exact = False
            break
        j = (i & -i).bit_length() - 1        # Gray code: flip one basis vector per step
        v = basis[j]
        bits = v
        while bits:
            low = bits & -bits
            jj = low.bit_length() - 1
            cx += -w[jj] if x >> jj & 1 else w[jj]
            bits ^= low
        x ^= v
        if cx < best_c:
            best_x, best_c = x, cx
    return {"presses": [j for j in range(n) if best_x >> j & 1], "cost": best_c, "exact": exact,
            "n_solutions": 1 << k}


def zm_solve(A, b, m, cost=None, max_states=1000000):
    """Minimum additive-cost press counts u in {0..m-1}^n with A u = b (mod m), any m
    (prime or not), by exact enumeration (see TOOL_DOC)."""
    n = len(A[0]) if A else 0
    if m < 2 or len(b) != len(A) or any(len(r) != n for r in A):
        raise ValueError("need m >= 2, equal-length rows, len(b) == len(A)")
    if m ** n > max_states:
        raise ValueError("m**n = %d candidate press vectors exceeds max_states=%d; fix some presses first"
                         % (m ** n, max_states))
    w = list(cost) if cost is not None else [1] * n
    best = None
    for u in _b2it.product(range(m), repeat=n):
        c = sum(wi * ui for wi, ui in zip(w, u))
        if best is not None and c >= best[1]:
            continue
        if all(sum(a * x for a, x in zip(row, u)) % m == bi % m for row, bi in zip(A, b)):
            best = (list(u), c)
    return None if best is None else {"presses": best[0], "cost": best[1]}


# ----------------------------------------------------------------------------- fitting
def _rank_frac(M):
    from fractions import Fraction as _F
    M = [[_F(v) for v in r] for r in M]
    rk = 0
    ncol = len(M[0]) if M else 0
    for c in range(ncol):
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


def fit_boolean(rows, values, max_degree=2, max_terms=4, time_limit=8.0):
    """Sparsest exact multilinear fit over observed 0/1 configurations (see TOOL_DOC).
    Exact rational arithmetic. 'unidentified' is computed for the full degree<=max_degree
    model: monomial S is pinned iff its unit vector lies in the row space of the
    observed design (exact rank test). 'complete' is False if the time budget cut the
    support search short (then a sparser rule may exist beyond what was searched)."""
    from fractions import Fraction as _F
    if not rows:
        return None
    n = len(rows[0])
    if len(values) != len(rows) or any(len(r) != n for r in rows):
        raise ValueError("rows must have equal length and len(values) == len(rows)")
    if any(x not in (0, 1, True, False) for r in rows for x in r):
        raise ValueError("rows must contain only 0/1 switch states (multilinear model)")
    t_end = _deadline(time_limit)
    monos = [()]
    for d in range(1, max_degree + 1):
        monos += list(_b2it.combinations(range(n), d))
    feat = [[int(all(r[i] for i in mono)) for mono in monos] for r in rows]
    y = [_F(v) for v in values]
    # pinned monomials (identifiability in the full model)
    base_rank = _rank_frac(feat)
    pinned = []
    for j in range(len(monos)):
        e = [0] * len(monos)
        e[j] = 1
        pinned.append(_rank_frac(feat + [e]) == base_rank)
    unidentified = [monos[j] for j in range(len(monos)) if not pinned[j]]

    def solve(cols):
        A_ = [[_F(f[j]) for j in cols] + [yy] for f, yy in zip(feat, y)]
        rr = 0
        piv = []
        for c in range(len(cols)):
            p = next((i for i in range(rr, len(A_)) if A_[i][c] != 0), None)
            if p is None:
                return None          # dependent columns: a smaller support would fit (MDL)
            A_[rr], A_[p] = A_[p], A_[rr]
            A_[rr] = [x / A_[rr][c] for x in A_[rr]]
            for i in range(len(A_)):
                if i != rr and A_[i][c] != 0:
                    A_[i] = [a - A_[i][c] * b for a, b in zip(A_[i], A_[rr])]
            piv.append(c)
            rr += 1
        if any(A_[i][-1] != 0 for i in range(rr, len(A_))):
            return None
        return [A_[i][-1] for i in range(len(cols))]
    complete = True
    for k in range(0, max_terms + 1):
        sols = []
        if k == 0:
            if all(v == 0 for v in y):
                sols.append({})
        else:
            for cols in _b2it.combinations(range(len(monos)), k):
                if _b2time.time() > t_end:
                    complete = False
                    break
                x = solve(cols)
                if x is not None and all(v != 0 for v in x):
                    sols.append({monos[j]: (int(v) if v.denominator == 1 else float(v))
                                 for j, v in zip(cols, x)})
        if sols:
            return {"terms": sols[0], "alternatives": len(sols) - 1, "unidentified": unidentified,
                    "complete": complete}
        if not complete:
            break
    return None if complete else {"terms": None, "alternatives": 0, "unidentified": unidentified,
                                  "complete": False}


def fit_guarded(xs, ys, tol=1e-9):
    """Simplest exact 1-D rule (see TOOL_DOC): affine, or affine with the value held at a
    constant beyond a wall at the high end ('cap_high': y=a*x+b for x<guard, y=cap for
    x>guard) or at the low end ('cap_low'). The guard is identified only up to the open
    interval 'guard_between' = (last x that must be on the line, first x that must be capped)."""
    pts = sorted(zip(xs, ys))
    if len(pts) < 2:
        return None

    def aff(p):
        if not p:
            return None
        if len(p) == 1 or p[0][0] == p[-1][0]:
            if all(abs(q[1] - p[0][1]) <= tol for q in p):
                return (0.0, float(p[0][1]))
            return None
        (x0, y0), (x1, y1) = p[0], p[-1]
        a = (y1 - y0) / (x1 - x0)
        b = y0 - a * x0
        return (a, b) if all(abs(a * x + b - y) <= tol for x, y in p) else None
    f = aff(pts)
    if f:
        return {"kind": "affine", "a": f[0], "b": f[1], "cap": None, "guard_between": None}
    best = None
    for kind in ("cap_high", "cap_low"):
        seq = pts if kind == "cap_high" else pts[::-1]
        for k in range(1, len(seq)):
            lin, capped = seq[:k], seq[k:]
            cap = capped[0][1]
            if not all(abs(y - cap) <= tol for _, y in capped):
                continue
            # The slope is identified only by points that cannot be on the capped side
            # (y != cap); a point with y == cap could belong to either piece.
            if len({x for x, y in lin if abs(y - cap) > tol}) < 2:
                continue
            g = aff(sorted(lin))
            if g is None:
                continue
            # guard interval: from the last point that is on the line but NOT equal to cap
            # (it must be on the linear side) to the first point that is NOT on the line
            # (it must be on the capped side)
            on_line = lambda x, y: abs(g[0] * x + g[1] - y) <= tol
            must_lin = [x for x, y in lin if abs(y - cap) > tol]
            must_cap = [x for x, y in capped if not on_line(x, y)]
            if kind == "cap_high":
                lo = max(must_lin) if must_lin else float("-inf")
                hi = min(must_cap) if must_cap else float("inf")
            else:
                lo = max(must_cap) if must_cap else float("-inf")
                hi = min(must_lin) if must_lin else float("inf")
            cand = {"kind": kind, "a": g[0], "b": g[1], "cap": cap, "guard_between": (lo, hi)}
            if best is None:
                best = cand
            break
    return best


B2_TOOL_NAMES = ("bfs_plan", "distinguish", "gf2_solve", "zm_solve", "fit_boolean", "fit_guarded")
