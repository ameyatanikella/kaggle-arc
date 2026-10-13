"""Local plans and the three plan-cache modes compared in docs/prereg_b3.md.

PlanEntry is a local controller:
  * start (precondition): the abstract node it applies from;
  * actions and expected nodes: one expected node per step, which may be visibly unchanged
    (same frame, different history), so unchanged pixels are not a plan failure;
  * deps: the edges (node, action) whose deterministic outcomes the plan relies on;
  * model version: the graph version (HUD-mask / abstraction state) it was built under;
  * goal: the target node, whose completion condition is "still has untried candidates";
  * allowance: maximum steps before control returns to the parent.

Modes:
  none        no cache; global BFS whenever a plan is needed (B1 behaviour).
  projected   ordinary cache keyed by (start node, goal kind). Entries are validated lazily
              by re-simulating every step on the current graph before use.
  dependency  entries indexed by every node they pass through (suffix reuse) and by every edge
              they depend on. A contradicted edge invalidates exactly the dependent entries,
              and a goal that stops being a frontier invalidates entries targeting it. Unrelated
              entries survive. A model-version change invalidates entries built under the old version.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class PlanEntry:
    pid: int
    start: tuple
    actions: list
    nodes: list                      # expected node after each action
    goal: tuple
    version: int
    deps: set = field(default_factory=set)
    allowance: int = 0
    uses: int = 0


class PlanCache:
    def __init__(self, mode: str, stats: dict):
        self.mode = mode
        self.stats = stats
        self.next_id = 0
        self.by_start: dict = {}     # node -> {pid}
        self.by_node: dict = {}      # node -> {(pid, index)} (dependency mode: suffix reuse)
        self.by_edge: dict = {}      # (node, action) -> {pid}
        self.by_goal: dict = {}      # goal -> {pid}
        self.entries: dict = {}
        self._words: set = set()
        self.neg: dict = {}          # node -> stamp (projected) | frozenset reachable set (dependency)
        self.neg_by_node: dict = {}  # dependency: node in reachable set -> {source nodes}
        for k in ("neg_built", "neg_hits", "neg_invalidated",
                  "cache_built", "cache_hits", "cache_misses", "cache_invalidated", "cache_validation_steps",
                  "cache_rejected_on_validation"):
            stats.setdefault(k, 0)
        stats.setdefault("t_cache", 0.0)

    # ---------------------------------------------------------------- store
    def store(self, start, actions, nodes, goal, version, slack=2):
        if self.mode == "none" or not actions:
            return None
        t0 = time.perf_counter()
        w = tuple(actions)                              # dedup profile: identical action words
        if w in self._words:
            self.stats["cache_dup_words"] = self.stats.get("cache_dup_words", 0) + 1
        self._words.add(w)
        e = PlanEntry(self.next_id, start, list(actions), list(nodes), goal, version,
                      allowance=len(actions) + slack)
        self.next_id += 1
        seq = [start] + list(nodes)
        e.deps = {(seq[i], actions[i]) for i in range(len(actions))}
        self.entries[e.pid] = e
        self.by_start.setdefault(start, set()).add(e.pid)
        if self.mode == "dependency":
            for i, n in enumerate(seq[:-1]):
                self.by_node.setdefault(n, set()).add((e.pid, i))
            for d in e.deps:
                self.by_edge.setdefault(d, set()).add(e.pid)
            self.by_goal.setdefault(goal, set()).add(e.pid)
        self.stats["cache_built"] += 1
        self.stats["t_cache"] += time.perf_counter() - t0
        return e

    def _drop(self, pid):
        e = self.entries.pop(pid, None)
        if e is None:
            return
        self.by_start.get(e.start, set()).discard(pid)
        if self.mode == "dependency":
            seq = [e.start] + e.nodes
            for i, n in enumerate(seq[:-1]):
                self.by_node.get(n, set()).discard((pid, i))
            for d in e.deps:
                self.by_edge.get(d, set()).discard(pid)
            self.by_goal.get(e.goal, set()).discard(pid)
        self.stats["cache_invalidated"] += 1

    # ---------------------------------------------------------------- negative results
    # "No frontier is reachable from node" is a reusable model computation too.
    def neg_store(self, node, reachable, stamp):
        if self.mode == "none":
            return
        t0 = time.perf_counter()
        if self.mode == "projected":
            self.neg[node] = stamp                       # ordinary: valid while nothing changed
        else:
            R = frozenset(reachable)
            self.neg[node] = R
            for n in R:
                self.neg_by_node.setdefault(n, set()).add(node)
        self.stats["neg_built"] += 1
        self.stats["t_cache"] += time.perf_counter() - t0

    def neg_lookup(self, node, stamp) -> bool:
        if self.mode == "none" or node not in self.neg:
            return False
        if self.mode == "projected" and self.neg[node] != stamp:
            del self.neg[node]
            self.stats["neg_invalidated"] += 1
            return False
        self.stats["neg_hits"] += 1
        return True

    def _neg_drop(self, src):
        R = self.neg.pop(src, None)
        if R is None:
            return
        if self.mode == "dependency":
            for n in R:
                self.neg_by_node.get(n, set()).discard(src)
        self.stats["neg_invalidated"] += 1

    def on_new_edge(self, pre):
        """A new transition out of `pre`: only negative results whose reachable set contains it
        can have changed (dependency mode). Plans stay valid (they may just not be shortest)."""
        if self.mode == "dependency":
            for src in list(self.neg_by_node.get(pre, ())):
                self._neg_drop(src)

    def on_candidates_changed(self):
        """Candidate tier expanded: every node may have new untried actions."""
        for src in list(self.neg):
            self._neg_drop(src)

    # ---------------------------------------------------------------- invalidation
    def on_contradiction(self, edge):
        """A real observation contradicted the deterministic outcome of `edge`."""
        if self.mode == "dependency":
            t0 = time.perf_counter()
            for pid in list(self.by_edge.get(edge, ())):
                self._drop(pid)
            self.stats["t_cache"] += time.perf_counter() - t0

    def on_goal_exhausted(self, goal):
        if self.mode == "dependency":
            for pid in list(self.by_goal.get(goal, ())):
                self._drop(pid)

    def on_version(self, version):
        """HUD-mask change: every node identity changed, so every entry is stale."""
        for src in list(self.neg):
            self._neg_drop(src)
        if self.mode == "dependency":
            for pid in [p for p, e in self.entries.items() if e.version != version]:
                self._drop(pid)

    def on_refine(self, mk, version):
        """Temporal refinement split the nodes of ONE masked frame `mk`. Entries passing through
        it are dropped; all others keep valid node identities and are re-stamped (preserved)."""
        if self.mode == "dependency":
            t0 = time.perf_counter()
            for src in [n for n, R in self.neg.items() if n[0] == mk or any(x[0] == mk for x in R)]:
                self._neg_drop(src)
            for pid, e in list(self.entries.items()):
                if e.start[0] == mk or any(n[0] == mk for n in e.nodes):
                    self._drop(pid)
                else:
                    e.version = version
                    self.stats["cache_preserved_on_refine"] = self.stats.get("cache_preserved_on_refine", 0) + 1
            self.stats["t_cache"] += time.perf_counter() - t0

    # ---------------------------------------------------------------- lookup
    def lookup(self, node, graph, is_goal):
        """A cached plan usable from `node`: (actions, nodes, goal) or None."""
        if self.mode == "none":
            return None
        t0 = time.perf_counter()
        best = None
        if self.mode == "projected":
            for pid in list(self.by_start.get(node, ())):
                e = self.entries[pid]
                # lazy validation: re-simulate every step on the current graph
                ok = is_goal(e.goal)
                cur = node
                for a, n in zip(e.actions, e.nodes):
                    self.stats["cache_validation_steps"] += 1
                    if not ok or graph.det_successor(cur, a) != n:
                        ok = False
                        break
                    cur = n
                if not ok:
                    self.stats["cache_rejected_on_validation"] += 1
                    self._drop(pid)
                    continue
                if best is None or len(e.actions) < len(best[0]):
                    best = (e.actions, e.nodes, e.goal)
        else:
            for pid, i in list(self.by_node.get(node, ())):
                e = self.entries.get(pid)
                if e is None or e.version != graph.version or not is_goal(e.goal):
                    continue
                acts, nodes = e.actions[i:], e.nodes[i:]
                if best is None or len(acts) < len(best[0]):
                    best = (acts, nodes, e.goal)
                    e.uses += 1
        self.stats["cache_hits" if best else "cache_misses"] += 1
        self.stats["t_cache"] += time.perf_counter() - t0
        return best
