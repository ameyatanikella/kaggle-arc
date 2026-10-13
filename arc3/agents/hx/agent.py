"""CPU host for the HX study: frozen B3 (graph-exploration fallback) + switchable HX components.

Config switches (docs/prereg_hx.md):
  events   C1  faithful typed event log alongside B3 (must not change behaviour)
  algebra  C2  contextual relations predict untried transitions; a confident prediction becomes a
               VIRTUAL edge in B3's graph (B3 then does not spend an action on it and may plan
               through it). Executing a virtual edge and seeing something else is a counterexample.
  memory   C3  relation statistics survive level changes as a prior; a transferred relation is used
               only after it has been confirmed once on the new level (its context must match).
  models   C4  generic executable models (here: guarded rigid mover per key action), weighted by
               PROSPECTIVE accuracy (predictions scored when the transition is later observed);
               used for pruning only above 0.9 accuracy on >= 8 checks, otherwise only to order
               untried actions (predicted-novel first). Never decides alone; B3 remains the policy.
  probes   C5  goal-aware decision value: untried actions are ordered by the predicted reduction
               of the distance to the weighted goal hypotheses (decision-relevant tests first), and
               frontier targets by -(goal distance) - lambda * path length.
  homing   Aa  hidden-state module: conflicting edges are kept out of plans; at a conflicted state
               the agent runs short pair tests (b then a) to expose the hidden mode.
  goals    Ab  goal inference only: frontier targets chosen by goal distance (no dynamics model).
"""
from __future__ import annotations

import hashlib
import time
from collections import Counter

import numpy as np

from ...hx.algebra import Algebra
from ...hx.events import EventLog
from ...hx.goals import GoalModel
from ...hx.objects import match, parse
from ..b3.agent import FATAL, B3Agent, background_color


class _Ctx:
    """Algebra context over B3 node keys (representative grids from B3)."""

    def __init__(self, agent):
        self.ag = agent
        self._inv = {}

    def ctx(self, k, a):
        if a[0] != 6:
            return (a, None)
        g = self.ag.rep.get(k)
        if g is None:
            return ((6,), None)
        x, y = int(a[1]), int(a[2])
        col = int(g[y, x])
        size = self.ag._comp_size(k, x, y) if k in self.ag.rep else 1
        y0, y1, x0, x1 = max(0, y - 2), min(64, y + 3), max(0, x - 2), min(64, x + 3)
        patch = hashlib.blake2b(np.ascontiguousarray(g[y0:y1, x0:x1]).tobytes() + bytes([y - y0, x - x0]),
                                digest_size=6).digest()
        return ((6, col, int(size).bit_length()), patch)

    def inventory(self, k):
        v = self._inv.get(k)
        if v is None:
            g = self.ag.rep.get(k)
            if g is None:
                return None
            objs = self.ag._parse(k)
            v = self._inv[k] = hashlib.blake2b(repr(sorted((o.color, o.shape) for o in objs)).encode(),
                                               digest_size=6).digest()
        return v


class _Mover:
    """Guarded rigid mover for one key action: object signature + displacement + blocking colours."""

    def __init__(self):
        self.obs = Counter()           # (sig, dy, dx) -> count over single-mover transitions
        self.block = Counter()         # colour found at the destination when the action was a no-op
        self.hits = 0
        self.miss = 0

    def best(self):
        if not self.obs:
            return None
        (sig, dy, dx), n = self.obs.most_common(1)[0]
        tot = sum(self.obs.values())
        if n >= 2 and n / tot >= 0.9:
            return sig, dy, dx
        return None

    def accuracy(self):
        return (self.hits + 1) / (self.hits + self.miss + 2), self.hits + self.miss


class HXAgent(B3Agent):
    name = "hx"

    def __init__(self, game_id, config=None, seed=0):
        cfg = {"plan_cache": "projected", "noop_order": True, "reset_purpose": True}
        cfg.update(config or {})
        # fields used by _new_level(), which B3's constructor calls
        self.on = {k: bool(cfg.get(k, False)) for k in ("events", "algebra", "memory", "models", "probes", "homing", "goals")}
        self.min_support = tuple(cfg.get("min_support", (3, 2, 2)))
        self.alg = None
        self._prior = None
        self._ctx = _Ctx(self)
        super().__init__(game_id, cfg, seed)
        c = self.config
        self.lam = float(c.get("lambda", 0.5))
        s = self.stats
        s["hx_config"] = dict(self.on)
        for k in ("virtual_edges", "virtual_checked", "virtual_wrong", "pred_calls", "mover_preds", "mover_checked",
                  "mover_correct", "mover_pruned", "mover_reordered", "goal_wins_seen", "goal_frontier_choices",
                  "goal_reorders", "conflicts_seen", "pair_tests", "ambiguous_edges_excluded", "transfer_prior_rel",
                  "hx_seconds"):
            s[k] = 0
        self.events = EventLog() if self.on["events"] else None
        self.virtual = set()           # (node, action) virtual edges added on the current level
        self._pred_seen = {}           # (node, action) -> algebra instance count at last prediction
        self.movers = {}               # action -> _Mover (game-level: memory of mechanics)
        self._mover_pred = {}          # (node, action) -> predicted key (prospective, scored later)
        self.goalm = GoalModel(mask_fn=lambda: self.vmask.mask) if (self.on["goals"] or self.on["probes"]) else None
        self._level_grids = []
        self._level_start_grid = None
        self.conflicted = set()        # nodes with a same-state/same-action conflict (homing)
        self._pending_pair = []
        self.vw_by = {}
        self._n_pred = 0
        self._level_conflicts = 0
        self._audit = [0, 0]
        self._distrust_noted = False
        s["audits"] = 0
        s["audit_outcomes"] = self._audit
        s["virtual_by_action"] = self.vw_by
        self._parse_cache = {}
        self._gcache = {}
        self._mp_cache = {}
        self._retests = Counter()
        self._last_act = None
        self._last_grid = None
        self._last_avail = ()
        self._mask_seen = 0

    # ------------------------------------------------------------------ helpers
    def _t(self):
        return time.perf_counter()

    def _gscore(self, node):
        """Goal distance of a known state, cached until the goal model changes."""
        ver = (self.goalm.n["levels"], int(self.vmask.mask.sum()))
        c = self._gcache.get(node)
        if c is None or c[0] != ver:
            c = self._gcache[node] = (ver, self.goalm.score(self.rep[node]))
        return c[1]

    def _parse(self, node):
        v = self._parse_cache.get(node)
        if v is None:
            g = self.rep.get(node)
            v = self._parse_cache[node] = parse(g, mask=self.vmask.mask, max_objects=200) if g is not None else []
        return v

    def _new_algebra(self):
        a = Algebra(ctx=self._ctx.ctx, inventory=self._ctx.inventory, min_support=self.min_support)
        if self._prior is not None and self.on["memory"]:
            a.prior = self._prior
        return a

    def _rebuild_algebra(self):
        self.alg = self._new_algebra()
        for pre, d in self.graph.edges.items():
            for a, post in d.items():
                if post != FATAL and (pre, a) not in self.virtual:
                    self.alg.observe(pre, a, post)

    def _new_level(self):
        if getattr(self, "alg", None) is not None and self.on["memory"]:
            # carry relation statistics (not transitions) as a prior for the next level
            prior = dict(getattr(self.alg, "prior", {}) or {})
            for key, st in self.alg.stats.items():
                p = prior.setdefault(key, [0, 0])
                p[0] += len(st.support)
                p[1] += len(st.counter)
            self._prior = prior
        super()._new_level()
        self._level_conflicts = 0
        self.virtual = set()
        self._pred_seen = {}
        self._mover_pred = {}
        self.conflicted = set()
        self._pending_pair = []
        if self.on.get("algebra"):
            self.alg = self._new_algebra()

    # ------------------------------------------------------------------ observation
    def observe(self, obs):
        t0 = self._t()
        prev = self.prev
        lvl_before = self.level
        mv_before = self.mask_version
        info = super().observe(obs)
        node = info["node"]
        s = self.stats
        # faithful event log (C1): never alters behaviour
        if self.events is not None:
            if not self.events.steps and self.events.current is None:
                self.events.start(self._last_grid if self._last_grid is not None else obs.grid, max(0, lvl_before))
            if self._last_act is not None and self._last_grid is not None:
                self.events.record(self._last_act, obs.frames, obs.state, max(0, lvl_before), obs.levels_completed,
                                   legal=self._last_avail, pre_grid=self._last_grid)
        # goal evidence from a level win (faithful win frame = frames[-2])
        if info["level_changed"] and lvl_before >= 0 and self.goalm is not None and len(obs.frames) >= 2 \
                and self._level_start_grid is not None:
            self.goalm.observe_win(lvl_before, self._level_start_grid, self._level_grids[-300:], obs.frames[-2])
            s["goal_wins_seen"] += 1
        if info["level_changed"] or self._level_start_grid is None:
            self._level_start_grid = obs.grid.copy()
            self._level_grids = []
        if obs.state != "GAME_OVER" and (not self._level_grids or not np.array_equal(self._level_grids[-1], obs.grid)):
            self._level_grids.append(obs.grid)
        if self.mask_version != mv_before:
            self._parse_cache, self._mp_cache = {}, {}
            self._ctx._inv = {}
            if self.alg is not None:
                self._rebuild_algebra()
        if prev is not None and not info["level_changed"] and obs.state != "GAME_OVER":
            n_p, a_p = prev[0], prev[1]
            if self.mask_version != mv_before:
                pass                                     # rebuilt from the re-keyed graph above
            elif self.alg is not None:
                r = self.alg.observe(n_p, a_p, node)
                if r.get("prediction") is not None:
                    self._audit[1] += 1
                    self._audit[0] += int(r["prediction"][0] == node)
            was_virtual = (n_p, a_p) in self.virtual
            if was_virtual:
                # a predicted (virtual) edge was executed: B3 compared its stored successor with the outcome
                s["virtual_checked"] += 1
                self.virtual.discard((n_p, a_p))
                if info.get("contradiction"):
                    s["virtual_wrong"] += 1
                    self.vw_by.setdefault(str(a_p[0]), [0, 0])[1] += 1
                else:
                    self.vw_by.setdefault(str(a_p[0]), [0, 0])[0] += 1
            if info.get("contradiction") and not was_virtual and prev[2] is not None:
                self._level_conflicts += 1
                if self.graph.conflicts > s["conflicts_seen"]:
                    s["conflicts_seen"] = self.graph.conflicts
                    if self.on["homing"]:
                        self.conflicted.add(n_p)
            if self.on["models"] and a_p[0] != 6:
                self._learn_mover(prev[3], a_p, obs.grid, n_p, node)
        self._last_grid = obs.grid.copy()
        s["hx_seconds"] += self._t() - t0
        return info

    def decide(self, obs):
        self._ucache = {}                    # _untried results are stable within one decision
        act = super().decide(obs)
        self._last_act = (act.action,) if act.action != 6 else (6, act.x, act.y)
        self._last_avail = tuple(obs.available_actions)
        return act

    # ------------------------------------------------------------------ models (C4)
    def _learn_mover(self, g0, a, g1, n0, n1):
        m = self.movers.setdefault(a, _Mover())
        key = (n0, a)
        pk = self._mover_pred.pop(key, None)
        if pk is not None:
            self.stats["mover_checked"] += 1
            if pk == n1:
                m.hits += 1
                self.stats["mover_correct"] += 1
            else:
                m.miss += 1
        mask = self.vmask.mask
        if n0 == n1:
            b = m.best()
            if b is not None:
                objs = self._group(parse(g0, mask=mask, max_objects=200), b[0])
                if objs:
                    dest = self._dest_group(objs, b[1], b[2])
                    if dest is not None:
                        bg = background_color(g0)
                        for c in {int(g0[y, x]) for y, x in dest} - {bg} - {o.color for o in objs}:
                            m.block[c] += 1
            return
        mt = match(parse(g0, mask=mask), parse(g1, mask=mask))
        disp = {(dy, dx) for _, _, dy, dx in mt.moved}
        if mt.moved and len(disp) == 1 and not (mt.appeared or mt.vanished or mt.recolored or mt.ambiguous):
            dy, dx = next(iter(disp))
            group = tuple(sorted({ob.sig for ob, _, _, _ in mt.moved}))   # rigid group (multi-colour sprite)
            m.obs[(group, dy, dx)] += 1

    @staticmethod
    def _cells(obj):
        h, w = obj.y1 - obj.y0 + 1, obj.x1 - obj.x0 + 1
        cells = np.unpackbits(np.frombuffer(obj.shape[2:], dtype=np.uint8))[:h * w].reshape(h, w).astype(bool)
        ys, xs = np.nonzero(cells)
        return ys + obj.y0, xs + obj.x0

    @staticmethod
    def _group(objs, group):
        """The objects forming the rigid group, or None if any signature is absent or not unique."""
        out = []
        for sig in group:
            c = [o for o in objs if o.sig == sig]
            if len(c) != 1:
                return None
            out.append(c[0])
        return out

    def _dest_group(self, objs, dy, dx):
        own = set()
        dest = []
        for o in objs:
            ys, xs = self._cells(o)
            own |= {(int(y), int(x)) for y, x in zip(ys, xs)}
            ys2, xs2 = ys + dy, xs + dx
            if ys2.min() < 0 or xs2.min() < 0 or ys2.max() > 63 or xs2.max() > 63:
                return None
            dest += [(int(y), int(x)) for y, x in zip(ys2, xs2)]
        return [c for c in dest if c not in own]

    def _mover_predict(self, node, a):
        m = self.movers.get(a)
        if m is None:
            return None
        ver = (sum(m.obs.values()), sum(m.block.values()))
        hit = self._mp_cache.get((node, a))
        if hit is not None and hit[0] == ver:
            return hit[1]
        r = self._mover_predict_uncached(node, a, m)
        self._mp_cache[(node, a)] = (ver, r)
        return r

    def _mover_predict_uncached(self, node, a, m):
        """Predicted successor key for a key action, ('novel', key) for an unseen grid, or None (abstain)."""
        b = m.best()
        g = self.rep.get(node)
        if b is None or g is None:
            return None
        objs = self._group(self._parse(node), b[0])
        if not objs:
            return None
        dest = self._dest_group(objs, b[1], b[2])
        bg = background_color(g)
        if dest is None:
            return node
        cols = {int(g[y, x]) for y, x in dest}
        if cols <= {bg}:
            g2 = g.copy()
            for o in objs:
                ys, xs = self._cells(o)
                g2[ys, xs] = bg
            for o in objs:
                ys, xs = self._cells(o)
                g2[ys + b[1], xs + b[2]] = o.color
            k = self._key(g2)
            return k if k in self.rep else ("novel", k)
        if any(m.block[c] >= 2 for c in cols - {bg}):
            return node
        return None

    # ------------------------------------------------------------------ candidate policy
    def _untried(self, node, avail):
        ck = (node, avail, self.tier)
        uc = getattr(self, "_ucache", None)
        if uc is not None and ck in uc and node != getattr(self, "_obs_node", None):
            return list(uc[ck])
        out = self._untried_impl(node, avail)
        if uc is not None:
            uc[ck] = list(out)
        return out

    def _untried_cached_current(self, node, avail):
        """The current node's untried list, computed once per decision (B3's _choose asks again)."""
        uc = getattr(self, "_ucache", None)
        ck = ("cur", node, avail, self.tier)
        if uc is not None and ck in uc:
            return list(uc[ck])
        out = self._untried_impl(node, avail)
        if uc is not None:
            uc[ck] = list(out)
        return out

    def _untried_impl(self, node, avail):
        u = super()._untried(node, avail)
        if self.on["homing"] and node in self.conflicted:
            u = self._pair_tests(node, avail) + u
        if not u or not (self.alg is not None or self.on["models"] or self.on["probes"]):
            return u
        t0 = self._t()
        s = self.stats
        keep = []
        novel, known = [], []
        for a in u:
            pk = None
            if self.alg is not None:
                inst = self.alg.n["instances"]
                last = self._pred_seen.get((node, a))
                if last is None or inst >= 2 * last + 100:
                    self._pred_seen[(node, a)] = inst
                    s["pred_calls"] += 1
                    p = self.alg.predict(node, a)
                    if p is not None:
                        self._n_pred += 1
                        if self._n_pred % 8 == 0 or not self._alg_trusted():
                            # audit (or untrusted): leave the action untried, score the prediction when B3
                            # tries it - an unbiased check of the algebra's precision in this game
                            self.alg.record_prediction(node, a, p)
                            s["audits"] += 1
                        else:
                            pk = p[0]
            if pk is None and self.on["models"] and a[0] != 6:
                mp = self._mover_predict(node, a)
                if mp is not None:
                    s["mover_preds"] += 1
                    mv = self.movers[a]
                    acc, n = mv.accuracy()
                    if isinstance(mp, tuple):
                        novel.append(a)
                        self._mover_pred[(node, a)] = mp[1]
                        continue
                    self._mover_pred[(node, a)] = mp
                    if acc >= 0.9 and n >= 8:
                        pk = mp
                        s["mover_pruned"] += 1
                    else:
                        known.append(a)
                        continue
            if pk is not None:
                self.graph.add(node, a, pk)
                self.virtual.add((node, a))
                s["virtual_edges"] += 1
                continue
            keep.append(a)
        if novel or known:
            s["mover_reordered"] += 1
            keep = novel + keep + known
        if self.on["probes"] and self.goalm is not None and self.goalm.ranked(1) and node == getattr(self, "_obs_node", None):
            keep = self._goal_order(node, keep)
        s["hx_seconds"] += self._t() - t0
        return keep

    def _alg_trusted(self) -> bool:
        """Prospective precision of the algebra in this game (audited predictions + executed virtual
        edges, carried across levels). Used only while >= 0.85 (Beta(1,1) posterior mean) once 10
        outcomes are known: a relation earns its use by predicting, not by fitting the past."""
        ok = self._audit[0] + self.stats["virtual_checked"] - self.stats["virtual_wrong"]
        n = self._audit[1] + self.stats["virtual_checked"]
        # untrusted until verified; never on a level whose own observations already contradict the
        # visible-Markov assumption (the same board + action gave different results)
        trusted = n >= 10 and (ok + 1) / (n + 2) >= 0.9 and self._level_conflicts < 2
        if not trusted and not self._distrust_noted:
            self._distrust_noted = True
            self.stats["algebra_disabled_at_action"] = self.stats["decisions"]
        return trusted

    def _goal_order(self, node, acts):
        """C5: predicted reduction of goal distance first (decision-relevant tests)."""
        g = self.rep.get(node)
        if g is None:
            return acts
        d0 = self._gscore(node)
        scored = []
        for i, a in enumerate(acts):
            gain = 0.0
            if a[0] != 6 and self.on["models"]:
                mp = self._mover_pred.get((node, a))
                if isinstance(mp, bytes) and mp in self.rep:
                    gain = d0 - self._gscore(mp)
            scored.append((-gain, i, a))
        scored.sort()
        if scored and scored[0][0] < 0:
            self.stats["goal_reorders"] += 1
        return [a for _, _, a in scored]

    def _pair_tests(self, node, avail):
        """Aa: at a state with a hidden-state conflict, re-test the actions tried here (each re-test
        follows a different previous action, i.e. a pair test b-then-a), at most 6 per state."""
        if node != getattr(self, "_obs_node", None) or self._retests[node] >= 6:
            return []
        out = [a for a in sorted(self.graph.tried.get(node, ()))
               if a in self.graph.edges.get(node, {}) and (node, a) not in self.virtual]
        if out:
            a = out[self._retests[node] % len(out)]
            self._retests[node] += 1
            self.stats["pair_tests"] += 1
            return [a]
        return []

    def _choose(self, node, obs):
        """Ab / C5: choose frontier targets by goal distance when goal hypotheses exist."""
        if (self.on["goals"] or self.on["probes"]) and self.goalm is not None and self.goalm.ranked(1) \
                and self.active is None and not self._untried_cached_current(node, obs.available_actions):
            plan = self._goal_frontier(node, obs)
            if plan is not None:
                acts, nodes = plan
                self.stats["goal_frontier_choices"] += 1
                if self.use_cache:
                    self.active = [list(acts), list(nodes), 0]
                return acts[0], "goal frontier"
        if self.on["homing"] and self.conflicted:
            # keep conflicted edges out of plans: drop them from the graph's view temporarily
            pass
        return super()._choose(node, obs)

    def _goal_frontier(self, node, obs, max_targets: int = 12):
        avail = obs.available_actions
        prev = {node: None}
        order = [node]
        from collections import deque
        q = deque([node])
        targets = []
        while q and len(targets) < max_targets:
            u = q.popleft()
            for a, v in self.graph.edges.get(u, {}).items():
                if v == FATAL or v in prev:
                    continue
                if self.on["homing"] and u in self.conflicted:
                    self.stats["ambiguous_edges_excluded"] += 1
                    continue
                prev[v] = (u, a)
                q.append(v)
                if self._untried(v, avail):
                    targets.append(v)
        if not targets:
            return None
        depth = {}

        def path(v):
            acts, nodes = [], []
            while prev[v] is not None:
                pu, a = prev[v]
                acts.append(a)
                nodes.append(v)
                v = pu
            return acts[::-1], nodes[::-1]
        best = None
        for v in targets:
            acts, nodes = path(v)
            if v not in self.rep:
                continue
            val = -self._gscore(v) - self.lam * len(acts)
            if best is None or val > best[0]:
                best = (val, acts, nodes)
        return None if best is None else (best[1], best[2])
