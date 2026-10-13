"""B3 (frozen candidate): B1's exploration policy + checked plan reuse with an ordinary cache.

Accepted by docs/b3_report.md. Rejected components (history refinement, goal reuse, temporal
experiments, dedup, object alternatives) are NOT in this execution path. They live in
arc3/research/b3_ablation for reproducibility.

Policy per decision:
  1. continue the active plan while each step's observed node equals the expected node;
  2. else try an untried candidate here;
  3. else reuse a cached plan from this node (validated by re-simulating it on the current
     graph) or search (BFS) to the nearest frontier, and cache the result. "No frontier
     reachable" is cached too, keyed by the graph's structural-change stamp;
  4. else reset the level if the start can reach a frontier;
  5. else the parent expands click candidates (component centres -> 4-px grid -> component
     pixels at 2-px stride) or keeps acting. A local allowance (per level) returns control to
     the parent and never ends the game; only WIN or the runner's budget ends it.

Timing is EXCLUSIVE: perception, inference, search, cache and policy_other partition each
decision, and every category is recorded in wall time and CPU time (process_time).
"""
from __future__ import annotations

import random
import time
import zlib
from collections import deque
from typing import Optional

import numpy as np

from ...agent_api import Act, Agent
from ...perception import VolatileMask, background_color, components

FATAL = ("FATAL",)
CATS = ("perception", "inference", "search", "cache", "policy_other")


class _Clock:
    def __init__(self, stats):
        self.s = stats
        for c in CATS:
            stats[f"wall_{c}"] = 0.0
            stats[f"cpu_{c}"] = 0.0

    def start(self):
        return time.perf_counter(), time.process_time()

    def add(self, cat, t0):
        w, c = time.perf_counter() - t0[0], time.process_time() - t0[1]
        self.s[f"wall_{cat}"] += w
        self.s[f"cpu_{cat}"] += c
        return w, c


class Graph:
    """Frame-keyed transition graph (latest observation wins, as B1)."""

    def __init__(self):
        self.edges: dict = {}          # node -> {action: node | FATAL}
        self.tried: dict = {}
        self.struct = 0                # structural-change stamp
        self.conflicts = 0

    def add(self, pre, action, post):
        d = self.edges.setdefault(pre, {})
        old = d.get(action)
        if old != post:
            self.struct += 1
            if old is not None:
                self.conflicts += 1
        d[action] = post
        self.tried.setdefault(pre, set()).add(action)
        return old

    def succ(self, node, action):
        v = self.edges.get(node, {}).get(action)
        return None if v is None or v == FATAL else v

    def bfs(self, src, is_goal):
        prev = {src: None}
        q = deque([src])
        exp = 0
        while q:
            u = q.popleft()
            exp += 1
            if u != src and is_goal(u):
                acts, nodes = [], []
                while prev[u] is not None:
                    pu, a = prev[u]
                    acts.append(a)
                    nodes.append(u)
                    u = pu
                return acts[::-1], nodes[::-1], exp
            for a, v in self.edges.get(u, {}).items():
                if v != FATAL and v not in prev:
                    prev[v] = (u, a)
                    q.append(v)
        return None, None, exp


class PlanCache:
    """Ordinary projected cache: plans keyed by start node, validated lazily by re-simulation;
    negative results keyed by start node and valid only while the structural stamp, candidate
    tier and mask version are unchanged."""

    def __init__(self, enabled: bool, stats: dict):
        self.on = enabled
        self.s = stats
        self.plans: dict = {}
        self.neg: dict = {}
        for k in ("cache_built", "cache_hits", "cache_rejected", "cache_validation_steps", "neg_built", "neg_hits",
                  "neg_stale"):
            stats.setdefault(k, 0)

    def lookup(self, node, graph, is_goal):
        if not self.on:
            return None
        best = None
        for e in list(self.plans.get(node, ())):
            acts, nodes, goal = e
            ok, cur = is_goal(goal), node
            for a, n in zip(acts, nodes):
                self.s["cache_validation_steps"] += 1
                if not ok or graph.succ(cur, a) != n:
                    ok = False
                    break
                cur = n
            if not ok:
                self.plans[node].remove(e)
                self.s["cache_rejected"] += 1
            elif best is None or len(acts) < len(best[0]):
                best = e
        if best:
            self.s["cache_hits"] += 1
        return best

    def store(self, node, acts, nodes):
        if self.on:
            self.plans.setdefault(node, []).append((acts, nodes, nodes[-1]))
            self.s["cache_built"] += 1

    def neg_hit(self, node, stamp) -> bool:
        if not self.on or node not in self.neg:
            return False
        if self.neg[node] != stamp:
            del self.neg[node]
            self.s["neg_stale"] += 1
            return False
        self.s["neg_hits"] += 1
        return True

    def neg_store(self, node, stamp):
        if self.on:
            self.neg[node] = stamp
            self.s["neg_built"] += 1


class B3Agent(Agent):
    name = "b3"

    def __init__(self, game_id, config=None, seed=0):
        super().__init__(game_id, config, seed)
        c = self.config
        self.use_cache = c.get("plan_cache", "projected") != "none"
        self.max_clicks = int(c.get("max_clicks", 48))
        self.level_allowance = int(c.get("level_allowance", 600))
        self.noop_order = bool(c.get("noop_order", False))
        self.reset_purpose = bool(c.get("reset_purpose", False))
        self.cls_stats: dict = {}            # action class -> [states where it was a no-op, effective count]
        self.rng = random.Random(seed)
        self.vmask = VolatileMask()
        self.mask_version = 0
        self.tier = 0
        self.level = -1
        s = self.stats = {}
        self.clock = _Clock(s)
        for k in ("decisions", "bfs_calls", "bfs_expansions", "plans_started", "plan_steps_followed", "plan_breaks",
                  "allowance_returns", "predictions", "pred_correct", "pred_change_tp", "pred_change_fp",
                  "pred_change_fn", "resets", "tier_expansions", "fallback_actions", "mask_remaps", "levels_seen",
                  "graph_nodes", "graph_conflicts"):
            s[k] = 0
        s["config"] = {"plan_cache": "projected" if self.use_cache else "none", "noop_order": self.noop_order,
                       "reset_purpose": self.reset_purpose}
        s["resets_skipped_no_purpose"] = 0
        s["noop_deferred"] = 0
        s["choice_reasons"] = {}
        self.effect_bbox: dict = {}
        self._new_level()

    def _new_level(self):
        self.graph = Graph()
        self.cache = PlanCache(self.use_cache, self.stats)
        self.rep: dict = {}                  # node -> representative grid
        self.cands: dict = {}
        self._size_cache: dict = {}
        self.prev = None                     # (node, action, predicted, grid)
        self.active = None                   # [acts, nodes, i]
        self.level_steps = 0
        self.start_node = None

    def _key(self, grid) -> bytes:
        return self.vmask.key(grid)

    def _candidates(self, node, avail):
        k = (node, self.tier, avail)
        if k in self.cands:
            return self.cands[k]
        acts = [(a,) for a in avail if a not in (0, 6)]
        if 6 in avail:
            g = self.rep[node]
            bg = background_color(g)
            comps = [c for c in components(g, skip_colors=(bg,)) if not self.vmask.mask[c.cy, c.cx]]
            comps.sort(key=lambda c: (c.size, c.y0, c.x0))
            seen = set()
            for c in comps:
                if (c.cx, c.cy) not in seen and len(seen) < self.max_clicks:
                    seen.add((c.cx, c.cy))
                    acts.append((6, c.cx, c.cy))
            if self.tier >= 1:
                for y in range(2, 62, 4):
                    for x in range(2, 62, 4):
                        if (x, y) not in seen and not self.vmask.mask[y, x]:
                            seen.add((x, y))
                            acts.append((6, x, y))
            if self.tier >= 2:
                for c in comps:
                    ys, xs = np.nonzero(g[c.y0:c.y1 + 1, c.x0:c.x1 + 1] == c.color)
                    for yy, xx in zip(ys[::2] + c.y0, xs[::2] + c.x0):
                        if (int(xx), int(yy)) not in seen:
                            seen.add((int(xx), int(yy)))
                            acts.append((6, int(xx), int(yy)))
        head = [a for a in acts if a[0] != 6]
        tail = [a for a in acts if a[0] == 6]
        r = random.Random(zlib.crc32(node) * 1000003 + 7919 * self.tier + self.seed)   # order-independent
        r.shuffle(head)
        r.shuffle(tail)
        self.cands[k] = head + tail
        return self.cands[k]

    def _cls(self, node, a):
        if a[0] != 6:
            return ("key", a[0])
        g = self.rep.get(node)
        if g is None:
            return ("click", None)
        x, y = a[1], a[2]
        col = int(g[y, x])
        return ("click", col, int(self._comp_size(node, x, y)).bit_length())

    def _comp_size(self, node, x, y):
        cache = self._size_cache.setdefault(node, {})
        if (x, y) not in cache:
            g = self.rep[node]
            from scipy import ndimage
            lab, _ = ndimage.label(g == g[y, x])
            cache[(x, y)] = int((lab == lab[y, x]).sum())
        return cache[(x, y)]

    def _hist_unchanged(self, c) -> bool:
        st = self.cls_stats.get(c)
        return bool(st) and st[1] == 0 and len(st[0]) >= 3

    def _untried(self, node, avail):
        tried = self.graph.tried.get(node, ())
        u = [a for a in self._candidates(node, avail) if a not in tried]
        if self.noop_order and u:
            good = [a for a in u if not self._hist_unchanged(self._cls(node, a))]
            if len(good) < len(u):
                self.stats["noop_deferred"] += 1
                u = good + [a for a in u if self._hist_unchanged(self._cls(node, a))]
        return u

    def _remap(self):
        """HUD mask grew: re-key every known frame (keys only merge) and rebuild the graph."""
        new_of = {n: self._key(g) for n, g in self.rep.items()}
        g2 = Graph()
        for pre, d in self.graph.edges.items():
            for a, post in d.items():
                g2.add(new_of.get(pre, pre), a, post if post == FATAL else new_of.get(post, post))
        rep2 = {}
        for n, g in self.rep.items():
            rep2.setdefault(new_of[n], g)
        self.graph, self.rep, self.cands = g2, rep2, {}
        if self.start_node is not None:                 # bug fix 2026-10-13: the start key must be re-keyed too
            self.start_node = new_of.get(self.start_node, self._key(self.rep.get(self.start_node, next(iter(rep2.values())))))
        self.cache = PlanCache(self.use_cache, self.stats)
        self.active = None
        self.mask_version += 1
        self.stats["mask_remaps"] += 1
        return new_of

    # ------------------------------------------------------------------
    def act(self, obs, deadline):
        s = self.stats
        s["decisions"] += 1
        t = self.clock.start()
        if obs.levels_completed != self.level:
            if self.level >= 0:
                s["levels_seen"] += 1
                self._snapshot()
            self.level = obs.levels_completed
            self._new_level()
        grid = obs.grid
        fatal = obs.state == "GAME_OVER"
        remap = None
        if self.prev is not None and not fatal:
            old = self.vmask.mask.copy()
            self.vmask.update(self.prev[3], grid)
            if not np.array_equal(old, self.vmask.mask):
                remap = True
        if remap:
            m = self._remap()
            n_p, a_p, _, g_p = self.prev
            self.prev = (m.get(n_p, n_p), a_p, None, g_p)
        node = self._key(grid)
        self.rep.setdefault(node, grid)
        if self.start_node is None:
            self.start_node = node
        self.clock.add("perception", t)

        t = self.clock.start()
        if self.prev is not None:
            n_p, a_p, pred, g_p = self.prev
            post = FATAL if fatal else node
            self.graph.add(n_p, a_p, post)
            changed = (not fatal) and node != n_p
            if self.noop_order:
                c = self._cls(n_p, a_p)
                st = self.cls_stats.setdefault(c, [set(), 0])
                if changed:
                    st[1] += 1
                elif not fatal:
                    st[0].add(n_p)
            if pred is not None:
                s["predictions"] += 1
                s["pred_correct"] += int(pred == post)
                pc = pred != FATAL and pred != n_p
                s["pred_change_tp"] += int(pc and changed)
                s["pred_change_fp"] += int(pc and not changed)
                s["pred_change_fn"] += int((not pc) and changed)
            if changed:
                d = np.nonzero(g_p != grid)
                if len(d[0]):
                    k = str(a_p[0])
                    y0, y1, x0, x1 = int(d[0].min()), int(d[0].max()), int(d[1].min()), int(d[1].max())
                    b = self.effect_bbox.get(k)
                    self.effect_bbox[k] = (min(b[0], y0), max(b[1], y1), min(b[2], x0), max(b[3], x1)) if b else (y0, y1, x0, x1)
            if self.active is not None:
                acts, nodes, i = self.active
                if post == nodes[i]:
                    s["plan_steps_followed"] += 1
                    self.active = None if i + 1 >= len(acts) else [acts, nodes, i + 1]
                else:
                    s["plan_breaks"] += 1
                    self.active = None
        self.clock.add("inference", t)

        t = self.clock.start()
        self._nested = [0.0, 0.0]
        if fatal:
            self.prev, self.active = None, None
            s["resets"] += 1
            self._end_policy(t)
            return Act(0, why="reset after game over")
        self.level_steps += 1
        if self.level_steps > self.level_allowance:
            s["allowance_returns"] += 1         # control returns to the parent; the game goes on
            self.level_steps = 0
            self.active = None
            if self.tier < 2 and 6 in obs.available_actions:
                self.tier += 1
                s["tier_expansions"] += 1
        choice, why = self._choose(node, obs)
        s["choice_reasons"][why] = s["choice_reasons"].get(why, 0) + 1
        if choice == "RESET":
            self.prev, self.active = None, None
            s["resets"] += 1
            self._end_policy(t)
            return Act(0, why=why)
        pred = self.graph.edges.get(node, {}).get(choice)
        self.prev = (node, choice, pred, grid)
        self._end_policy(t)
        if choice[0] == 6:
            return Act(6, choice[1], choice[2], why=why)
        return Act(choice[0], why=why)

    def _end_policy(self, t):
        w = time.perf_counter() - t[0] - self._nested[0]
        c = time.process_time() - t[1] - self._nested[1]
        self.stats["wall_policy_other"] += w
        self.stats["cpu_policy_other"] += c

    def _timed(self, cat, fn):
        t = self.clock.start()
        r = fn()
        w, c = self.clock.add(cat, t)
        self._nested[0] += w
        self._nested[1] += c
        return r

    def _choose(self, node, obs):
        s = self.stats
        avail = obs.available_actions
        if self.active is not None:
            acts, nodes, i = self.active
            if self._untried(nodes[-1], avail):
                return acts[i], "active plan"
            self.active = None
        untried = self._untried(node, avail)
        if untried:
            return untried[0], "untried"
        is_goal = lambda n: bool(self._untried(n, avail))
        stamp = (self.graph.struct, self.tier, self.mask_version)
        if not self._timed("cache", lambda: self.cache.neg_hit(node, stamp)):
            plan = self._timed("cache", lambda: self.cache.lookup(node, self.graph, is_goal))
            if plan is None:
                acts, nodes, exp = self._timed("search", lambda: self.graph.bfs(node, is_goal))
                s["bfs_calls"] += 1
                s["bfs_expansions"] += exp
                if acts:
                    plan = (acts, nodes, nodes[-1])
                    self._timed("cache", lambda: self.cache.store(node, acts, nodes))
                else:
                    self._timed("cache", lambda: self.cache.neg_store(node, stamp))
            if plan is not None:
                acts, nodes, _ = plan
                s["plans_started"] += 1
                if self.use_cache:
                    self.active = [list(acts), list(nodes), 0]
                return acts[0], "plan"
        if obs.level_actions_used > 0 and node != self.start_node:
            if not self.reset_purpose:
                return "RESET", "reset to reach frontier"
            # reset only with a concrete purpose: the RECORDED graph shows the start reaching an
            # unfinished frontier (or the start itself has untried candidates)
            st = self.start_node
            purpose = bool(self._untried(st, avail)) or self._timed(
                "search", lambda: self.graph.bfs(st, is_goal)[0]) is not None
            if purpose:
                return "RESET", "reset to reach frontier"
            s["resets_skipped_no_purpose"] += 1
        if self.tier < 2 and 6 in avail:
            self.tier += 1
            s["tier_expansions"] += 1
            u = self._untried(node, avail)
            if u:
                return u[0], "expanded candidates"
        s["fallback_actions"] += 1
        c = self._candidates(node, avail)
        return (self.rng.choice(c) if c else (avail[0],)), "fallback"

    def _snapshot(self):
        s = self.stats
        s["graph_nodes"] += len(self.graph.edges)
        s["graph_conflicts"] += self.graph.conflicts
        s["hud_mask_hex"] = np.packbits(self.vmask.mask).tobytes().hex()
        s["effect_bbox"] = dict(self.effect_bbox)

    def finalize(self):
        self._snapshot()
        self.stats["wall_total"] = sum(self.stats[f"wall_{c}"] for c in CATS)
        self.stats["cpu_total"] = sum(self.stats[f"cpu_{c}"] for c in CATS)
        return self.stats
