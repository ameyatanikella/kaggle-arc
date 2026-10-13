"""B3: model-based explorer with checked plan reuse and temporal state (docs/prereg_b3.md).

Switches (config):
  plan_cache   none | projected | dependency
  temporal     False | True       history-refined state abstraction (graph.py)
  experiments  off | shared | separate   temporal repeat-until-change experiments
Shared base (all variants):
  * B1's exploration policy (untried candidate here, else path to nearest frontier, else a
    level reset if the start can reach a frontier);
  * a local allowance (per level, per candidate tier) that returns control to the parent;
  * a parent that expands click candidates (component centres -> 4-px grid -> component
    pixels at 2-px stride) and otherwise keeps acting. B3 never stops the game itself;
    only WIN or the runner's budget ends it.
Observations only: frames and public metadata. No engine internals, no game sources.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ...agent_api import Act, Agent
from ...perception import VolatileMask, background_color, components
from .abstraction import dependency_region
from .graph import FATAL, L_MAX, TemporalGraph
from .plans import PlanCache

TIMERS = ("t_perception", "t_inference", "t_search", "t_validation", "t_experiments", "t_policy")


@dataclass
class Experiment:
    start: tuple            # precondition node (masked frame, history)
    mk: int
    action: tuple
    allowance: int          # R repetitions beyond the first known no-change application
    fresh: bool             # precondition "no trailing unchanged actions" verified at start
    k: int = 0
    outcome: Optional[str] = None


class B3Agent(Agent):
    name = "b3"

    def __init__(self, game_id, config=None, seed=0):
        super().__init__(game_id, config, seed)
        c = self.config
        self.mode = c.get("plan_cache", "none")
        self.temporal = bool(c.get("temporal", False))
        self.exp_mode = c.get("experiments", "off")
        self.max_clicks = int(c.get("max_clicks", 48))
        self.level_allowance = int(c.get("level_allowance", 600))
        self.exp_R = int(c.get("exp_repeats", 5))
        self.goal_reuse = bool(c.get("goal_reuse", False))
        self.goal_plans: list = []          # [{'region','start_proj','actions','exp_proj','uses','ok','fail'}]
        self.goal_active = None             # [plan, i]
        self.shuffle = bool(c.get("shuffle", True))
        self.rng = random.Random(seed)
        self.vmask = VolatileMask()
        self.tier = 0
        self.level = -1
        s = self.stats = {k: 0.0 for k in TIMERS}
        for k in ("decisions", "bfs_calls", "bfs_expansions", "plan_steps_followed", "plan_breaks", "plans_started",
                  "allowance_returns", "predictions", "pred_correct", "pred_change_tp", "pred_change_fp",
                  "pred_change_fn", "resets", "tier_expansions", "fallback_actions", "mask_remaps",
                  "exp_started", "exp_valid_verified", "exp_valid_assumed", "exp_distinguishing",
                  "exp_hypotheses_eliminated", "exp_novel_obs", "exp_fatal", "exp_overhead_actions",
                  "rules_learned", "rules_used_ok", "rules_used_fail", "levels_seen",
                  "goal_plans_stored", "goal_reuse_started", "goal_reuse_steps_ok", "goal_reuse_completed",
                  "goal_reuse_invalidated"):
            s[k] = 0
        s["config"] = {"plan_cache": self.mode, "temporal": self.temporal, "experiments": self.exp_mode,
                       "goal_reuse": self.goal_reuse}
        self.effect_bbox: dict = {}
        self._new_level()

    # ------------------------------------------------------------------ per level
    def _new_level(self):
        self.graph = TemporalGraph(self.temporal)
        self.cache = PlanCache(self.mode, self.stats)
        self.raw: dict = {}                 # raw grid bytes -> grid
        self.rep_grid: dict = {}            # mk id -> representative raw grid
        self.cands_cache: dict = {}         # (mk, tier) -> candidates
        self.hist: tuple = ()
        self.prev = None                    # (mk, hist, action, predicted_post, grid)
        self.active = None                  # [actions, nodes, i, goal, steps_taken, allowance]
        self.exp: Optional[Experiment] = None
        self.exp_done: set = set()          # (mk, action)
        self.exp_edges: set = set()         # edges first observed during experiments
        self.seen_mk: set = set()
        self.level_steps = 0
        self.start_mk = None
        self.attempt_trace: list = []       # (action, grid_after) since the last (re)start
        self.attempt_start = None

    # ------------------------------------------------------------------ perception
    def _mk(self, grid) -> int:
        b = grid.tobytes()
        if b not in self.raw:
            self.raw[b] = grid.copy()
        mk = self.graph.mk_id(self.vmask.key(grid))
        self.rep_grid.setdefault(mk, grid)
        return mk

    def _remap_after_mask_change(self):
        """The HUD mask grew: re-key every known masked frame (keys can only merge) and rebuild."""
        new_ids_by_key: dict = {}
        mapping = {}
        rep = {}
        for old_mk, g in list(self.rep_grid.items()):
            k = self.vmask.key(g)
            nid = new_ids_by_key.setdefault(k, len(new_ids_by_key))
            mapping[old_mk] = nid
            rep.setdefault(nid, g)
        self.graph.mk_ids = new_ids_by_key
        self.graph.remap_frames(mapping)
        self.rep_grid = rep
        self.cands_cache = {}
        self.exp_done = {(mapping.get(m, m), a) for m, a in self.exp_done}
        self.seen_mk = {mapping.get(m, m) for m in self.seen_mk}
        self.cache.on_version(self.graph.version)
        self.active = None
        self.stats["mask_remaps"] += 1
        return mapping

    # ------------------------------------------------------------------ candidates
    def _candidates(self, mk: int, avail: tuple) -> list:
        key = (mk, self.tier, avail)
        if key in self.cands_cache:
            return self.cands_cache[key]
        acts = [(a,) for a in avail if a not in (0, 6)]
        if 6 in avail:
            g = self.rep_grid[mk]
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
        if self.shuffle:
            head = [a for a in acts if a[0] != 6]
            tail = [a for a in acts if a[0] == 6]
            self.rng.shuffle(head)
            self.rng.shuffle(tail)
            acts = head + tail
        self.cands_cache[key] = acts
        return acts

    def _untried(self, node, avail) -> list:
        tried = self.graph.tried.get(node, ())
        return [a for a in self._candidates(node[0], avail) if a not in tried]

    # ------------------------------------------------------------------ main loop
    def act(self, obs, deadline):
        s = self.stats
        s["decisions"] += 1
        t0 = time.perf_counter()
        if obs.levels_completed != self.level:
            if self.level >= 0:
                s["levels_seen"] += 1
                self._snapshot()
                if self.goal_reuse and self.attempt_start is not None and self.prev is not None:
                    self._store_goal_plan()
                if self.goal_active is not None:
                    s["goal_reuse_completed"] += 1
                self.goal_active = None
            self.level = obs.levels_completed
            self._new_level()
        grid = obs.grid
        if self.attempt_start is None and obs.state != "GAME_OVER":
            self.attempt_start = grid.copy()
            self.attempt_trace = []
            if self.goal_reuse and self.goal_active is None:
                self._try_goal_plan(grid)
        if self.prev is not None and obs.state != "GAME_OVER":
            old = self.vmask.mask.copy()
            self.vmask.update(self.prev[4], grid)
            if not np.array_equal(old, self.vmask.mask):
                s["_mask_changed"] = True
        mask_changed = s.pop("_mask_changed", False)
        if mask_changed:
            mapping = self._remap_after_mask_change()
            if self.prev is not None:
                mk_p, h_p, a_p, pred_p, g_p = self.prev
                self.prev = (mapping.get(mk_p, mk_p), h_p, a_p, None, g_p)
        mk = self._mk(grid)
        if self.start_mk is None:
            self.start_mk = mk
        s["t_perception"] += time.perf_counter() - t0

        # ---- inference: record the transition that led here
        t1 = time.perf_counter()
        fatal = obs.state == "GAME_OVER"
        if self.prev is not None:
            mk_p, h_p, a_p, pred_p, g_p = self.prev
            res = self.graph.add(mk_p, h_p, a_p, None if fatal else mk, fatal)
            post = res["post"]
            changed = (not fatal) and mk != mk_p
            if pred_p is not None:
                s["predictions"] += 1
                s["pred_correct"] += int(pred_p == post)
                pred_changed = pred_p != FATAL and pred_p[0] != mk_p
                s["pred_change_tp"] += int(pred_changed and changed)
                s["pred_change_fp"] += int(pred_changed and not changed)
                s["pred_change_fn"] += int((not pred_changed) and changed)
            if res["conflict"]:
                self.cache.on_contradiction((res["pre"] if not res["refined"] else self.graph.node(mk_p, h_p), a_p))
            if res["refined"]:
                self.cache.on_refine(mk_p, self.graph.version)
            if res.get("struct"):
                self.cache.on_new_edge(res["pre"])
            if changed:
                d = np.nonzero(g_p != grid)
                if len(d[0]):
                    k = a_p[0]
                    y0, y1, x0, x1 = int(d[0].min()), int(d[0].max()), int(d[1].min()), int(d[1].max())
                    b = self.effect_bbox.get(k)
                    self.effect_bbox[k] = (min(b[0], y0), max(b[1], y1), min(b[2], x0), max(b[3], x1)) if b else (y0, y1, x0, x1)
            # active-plan validation (unchanged frames are fine if the expected node says so)
            tv = time.perf_counter()
            if self.active is not None:
                acts, nodes, i, goal, steps, allow = self.active
                if post == nodes[i]:
                    s["plan_steps_followed"] += 1
                    if (res["pre"], a_p) in self.exp_edges:
                        s["rules_used_ok"] += 1
                    i += 1
                    self.active = None if i >= len(acts) else [acts, nodes, i, goal, steps + 1, allow]
                else:
                    if (res["pre"], a_p) in self.exp_edges:
                        s["rules_used_fail"] += 1
                    s["plan_breaks"] += 1
                    self.cache.on_contradiction((res["pre"], a_p))
                    self.active = None
            s["t_validation"] += time.perf_counter() - tv
            # experiment bookkeeping
            if self.exp is not None:
                self._exp_observe(res, changed, fatal, mk)
            if not fatal:
                self.hist = (h_p + ((a_p, changed),))[-(L_MAX + 1):]
                self.attempt_trace.append((a_p, grid.copy()))
                if self.goal_active is not None:
                    self._validate_goal_step(grid)
            pre_node = res["pre"]
            if not self._untried(pre_node, obs.available_actions):
                self.cache.on_goal_exhausted(pre_node)
        if mk not in self.seen_mk:
            self.seen_mk.add(mk)
        s["t_inference"] += time.perf_counter() - t1

        # ---- policy
        tp = time.perf_counter()
        if fatal:
            self.prev, self.hist, self.active = None, (), None
            self.attempt_start, self.goal_active = None, None
            if self.exp is not None:
                self._exp_finish("fatal")
            s["resets"] += 1
            s["t_policy"] += time.perf_counter() - tp
            return Act(0, why="reset after game over")
        self.level_steps += 1
        if self.level_steps > self.level_allowance:
            # local allowance exhausted: control returns to the parent (never ends the game)
            s["allowance_returns"] += 1
            self.level_steps = 0
            self.active = None
            if self.tier < 2 and 6 in obs.available_actions:
                self.tier += 1
                s["tier_expansions"] += 1
                self.cache.on_candidates_changed()
        node = self.graph.node(mk, self.hist)
        avail = obs.available_actions
        choice, why = self._choose(node, mk, avail, obs)
        s["t_policy"] += time.perf_counter() - tp
        if choice == "RESET":
            self.prev, self.hist, self.active = None, (), None
            self.attempt_start, self.goal_active = None, None
            s["resets"] += 1
            return Act(0, why=why)
        pred = self.graph.det_successor(node, choice)
        if pred is None and (self.graph.edges.get(node, {}).get(choice) or {}).get(FATAL):
            pred = FATAL
        self.prev = (mk, self.hist, choice, pred, grid)
        if choice[0] == 6:
            return Act(6, choice[1], choice[2], why=why)
        return Act(choice[0], why=why)

    # ------------------------------------------------------------------ policy core
    def _is_frontier(self, avail):
        return lambda n: bool(self._untried(n, avail))

    def _choose(self, node, mk, avail, obs):
        s = self.stats
        # 0. a reused goal plan continues while every step validates on its dependency region
        if self.goal_active is not None:
            plan, i = self.goal_active
            if i < len(plan["actions"]):
                self.goal_active = [plan, i + 1]
                return plan["actions"][i], "goal plan reuse"
            self._invalidate_goal(plan)                     # ran out without completing the level
        # 1. an experiment in progress continues through its expected no-change prefix
        if self.exp is not None:
            if self.exp.k < self.exp.allowance:
                return self.exp.action, "experiment"
            self._exp_finish("no_change")
        # 2. an active plan continues (its next expected node was validated on arrival)
        if self.active is not None:
            acts, nodes, i, goal, steps, allow = self.active
            if steps < allow and self._untried(goal, avail):
                return acts[i], "active plan"
            s["allowance_returns"] += int(steps >= allow)
            self.active = None
        # 3. untried candidate here
        untried = self._untried(node, avail)
        if untried:
            return untried[0], "untried"
        # 4. cached plan or fresh search to the nearest frontier
        is_goal = self._is_frontier(avail)
        stamp = (self.graph.struct_changes, self.graph.version, self.tier)
        plan = None
        if not self.cache.neg_lookup(node, stamp):
            plan = self.cache.lookup(node, self.graph, is_goal)
            if plan is None:
                t = time.perf_counter()
                acts, nodes, exp = self.graph.bfs(node, is_goal)
                s["t_search"] += time.perf_counter() - t
                s["bfs_calls"] += 1
                s["bfs_expansions"] += exp
                if acts:
                    plan = (acts, nodes, nodes[-1])
                    self.cache.store(node, acts, nodes, nodes[-1], self.graph.version)
                else:
                    self.cache.neg_store(node, self.graph.last_visited, stamp)
        if plan is not None:
            acts, nodes, goal = plan
            s["plans_started"] += 1
            if self.mode != "none":
                self.active = [list(acts), list(nodes), 0, goal, 0, len(acts) + 2]
                self.active[2] = 0
                return acts[0], "plan"
            return acts[0], "bfs step"
        # 5. no reachable frontier: temporal experiments (if enabled)
        if self.exp_mode != "off":
            e = self._exp_next(node, mk, avail)
            if e is not None:
                return e
        # 6. level reset if the start can reach a frontier
        if obs.level_actions_used > 0 and mk != self.start_mk:
            return "RESET", "reset to reach unexplored frontier"
        # 7. parent: expand candidates, else keep acting (never stop)
        if self.tier < 2 and 6 in avail:
            self.tier += 1
            s["tier_expansions"] += 1
            self.cache.on_candidates_changed()
            u = self._untried(node, avail)
            if u:
                return u[0], "expanded candidates"
        s["fallback_actions"] += 1
        cands = self._candidates(mk, avail)
        return self.rng.choice(cands) if cands else (avail[0],), "fallback"

    # ------------------------------------------------------------------ temporal experiments
    def _exp_candidates(self, node, mk, avail):
        out = []
        for a in self._candidates(mk, avail):
            post = self.graph.det_successor(node, a)
            if post is not None and post[0] == mk and (mk, a) not in self.exp_done:
                out.append(a)
        return out

    def _exp_next(self, node, mk, avail):
        s = self.stats
        cands = self._exp_candidates(node, mk, avail)
        if not cands:
            return None
        fresh = not self.hist or self.hist[-1][1]          # no trailing unchanged actions
        if self.exp_mode == "separate" and not fresh:
            # restore the fresh precondition before the next test (reset + navigation charged)
            s["exp_overhead_actions"] += 1
            return "RESET", "fresh start for separate experiment"
        a = cands[0]
        t = time.perf_counter()
        self.exp = Experiment(start=node, mk=mk, action=a, allowance=self.exp_R, fresh=fresh)
        self.exp_done.add((mk, a))
        s["exp_started"] += 1
        s["t_experiments"] += time.perf_counter() - t
        return a, "experiment start"

    def _exp_observe(self, res, changed, fatal, mk):
        e = self.exp
        e.k += 1
        if (res["pre"], e.action) not in self.graph.latest or res.get("conflict"):
            self.exp_edges.add((res["pre"], e.action))
        if fatal:
            self._exp_finish("fatal")
        elif changed:
            if mk not in self.seen_mk:
                self.stats["exp_novel_obs"] += 1
            self._exp_finish("change")

    def _exp_finish(self, outcome):
        s = self.stats
        e = self.exp
        if e is None:
            return
        e.outcome = outcome
        s["exp_valid_verified" if e.fresh else "exp_valid_assumed"] += 1
        # branches: H0 = no change within allowance, Hk = first change at repetition k (k=1..R)
        s["exp_hypotheses_eliminated"] += e.allowance        # every outcome leaves exactly one branch
        if outcome in ("change", "fatal"):
            s["exp_distinguishing"] += 1
            s["rules_learned"] += 1
            s["exp_fatal"] += int(outcome == "fatal")
        self.exp = None

    # ------------------------------------------------------------------ goal-plan reuse
    def _store_goal_plan(self):
        """The level just completed. Store the SHORTEST known path from the attempt's start to the
        completing step (graph BFS), so exploratory detours do not enter its dependency region.
        Falls back to the executed trace if the graph has no such path."""
        mk_p, h_p, a_last = self.prev[0], self.prev[1], self.prev[2]
        pre_node = self.graph.node(mk_p, h_p)
        start_mk = self.graph.mk_ids.get(self.vmask.key(self.attempt_start))
        acts, nodes = None, None
        if start_mk is not None:
            start = self.graph.node(start_mk, ())
            if start == pre_node:
                acts, nodes = [], []
            else:
                acts, nodes, _ = self.graph.bfs(start, lambda n: n == pre_node)
        if acts is not None:
            frames = [self.attempt_start] + [self.rep_grid[n[0]] for n in nodes]
            acts = list(acts) + [a_last]
            self.stats["goal_plans_from_graph"] = self.stats.get("goal_plans_from_graph", 0) + 1
        else:
            acts = [a for a, _ in self.attempt_trace] + [a_last]
            frames = [self.attempt_start] + [g for _, g in self.attempt_trace]
        region = dependency_region(frames) & ~self.vmask.mask
        if not region.any():
            region = ~self.vmask.mask
        self.goal_plans.append({"region": region, "start_proj": frames[0][region].tobytes(),
                                "actions": acts, "exp_proj": [f[region].tobytes() for f in frames[1:]],
                                "uses": 0, "ok": 0, "fail": 0})
        self.stats["goal_plans_stored"] += 1

    def _try_goal_plan(self, grid):
        for p in self.goal_plans:
            if grid[p["region"]].tobytes() == p["start_proj"]:
                p["uses"] += 1
                self.goal_active = [p, 0]
                self.stats["goal_reuse_started"] += 1
                return

    def _validate_goal_step(self, grid):
        plan, i = self.goal_active
        if i - 1 < len(plan["exp_proj"]) and grid[plan["region"]].tobytes() == plan["exp_proj"][i - 1]:
            self.stats["goal_reuse_steps_ok"] += 1
            return
        self._invalidate_goal(plan)

    def _invalidate_goal(self, plan):
        """Counterexample (e.g. coupled state outside the region): drop the claim, keep exploring."""
        plan["fail"] += 1
        if plan in self.goal_plans:
            self.goal_plans.remove(plan)
        self.goal_active = None
        self.stats["goal_reuse_invalidated"] += 1

    # ------------------------------------------------------------------ records
    def _snapshot(self):
        g = self.graph
        s = self.stats
        s["graph_nodes"] = s.get("graph_nodes", 0) + len(g.edges)
        s["graph_conflicts"] = s.get("graph_conflicts", 0) + g.conflicts
        s["graph_refinements"] = s.get("graph_refinements", 0) + g.refinements
        s["graph_unresolved"] = s.get("graph_unresolved", 0) + g.unresolved
        s["graph_rebuilds"] = s.get("graph_rebuilds", 0) + g.rebuilds
        s["hud_mask_hex"] = np.packbits(self.vmask.mask).tobytes().hex()
        s["effect_bbox"] = {str(k): v for k, v in self.effect_bbox.items()}

    def finalize(self):
        self._snapshot()
        return self.stats
