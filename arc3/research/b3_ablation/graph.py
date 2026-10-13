"""Temporal state graph for B3.

A node is (masked-frame id, history suffix). The history is the sequence of
(action, visibly_changed) pairs since the level (re)started. Every masked frame has its
own suffix length L (default 0, which is B1's frame-keyed abstraction). A transition
conflict is the same node and action leading to different successors. It proves the
current abstraction merges states that behave differently (Markov conflict). With
temporal refinement on, the suffix length of that frame is increased (up to L_MAX) and
the graph is rebuilt from the transition log: the abstraction is provisional and is
refined on counterexamples. Conflicts that survive L_MAX are kept as alternatives
(non-deterministic edges, which planning does not rely on), never silently overwritten.
With refinement off, the latest observation wins (B1 semantics).
Equal visible frames are therefore never taken as proof of equivalent states once a
counterexample exists.
"""
from __future__ import annotations

from collections import Counter, deque
from typing import Optional

FATAL = ("FATAL",)
L_MAX = 6


class TemporalGraph:
    def __init__(self, temporal: bool):
        self.temporal = temporal
        self.mk_ids: dict = {}            # masked-frame bytes -> small int
        self.L: dict = {}                 # mk id -> history suffix length
        self.log: list = []               # (mk_pre, hist_pre, action, mk_post, changed, fatal)
        self.edges: dict = {}             # node -> {action: Counter(post node)}
        self.latest: dict = {}            # (node, action) -> post node (latest observation)
        self.tried: dict = {}             # node -> set(actions)
        self.version = 0                  # bumps on every rebuild (model version)
        self.conflicts = 0
        self.unresolved = 0
        self.refinements = 0
        self.rebuilds = 0
        self.struct_changes = 0           # new (node, action) edges or new successors

    # ---------------------------------------------------------------- keys
    def mk_id(self, mk: bytes) -> int:
        i = self.mk_ids.get(mk)
        if i is None:
            i = self.mk_ids[mk] = len(self.mk_ids)
        return i

    def node(self, mk: int, hist: tuple) -> tuple:
        L = self.L.get(mk, 0) if self.temporal else 0
        return (mk, hist[-L:] if L else ())

    # ---------------------------------------------------------------- updates
    def add(self, mk_pre: int, hist_pre: tuple, action, mk_post: Optional[int], fatal: bool) -> dict:
        """Record one real transition; returns {'conflict': bool, 'refined': bool, 'pre', 'post'}."""
        changed = (mk_post is not None and mk_post != mk_pre)
        self.log.append((mk_pre, hist_pre, action, mk_post, changed, fatal))
        out = {"conflict": False, "refined": False}
        pre = self.node(mk_pre, hist_pre)
        post = FATAL if fatal else self.node(mk_post, hist_pre + ((action, changed),))
        known = self.edges.get(pre, {}).get(action)
        if known and post not in known:
            out["conflict"] = True
            self.conflicts += 1
            if self.temporal and self.L.get(mk_pre, 0) < L_MAX and len(hist_pre) > self.L.get(mk_pre, 0):
                # refine on the counterexample, then keep refining while the LOGGED evidence still
                # shows conflicts out of this frame (no need to wait for new observations)
                while True:
                    self.L[mk_pre] = self.L.get(mk_pre, 0) + 1
                    self.refinements += 1
                    self.rebuild()
                    if self.L[mk_pre] >= L_MAX or not self._conflicted(mk_pre):
                        break
                out["refined"] = True
                pre = self.node(mk_pre, hist_pre)
                post = FATAL if fatal else self.node(mk_post, hist_pre + ((action, changed),))
                out.update(pre=pre, post=post, struct=True)
                return out
            if self.temporal:
                self.unresolved += 1
        self._insert(pre, action, post)
        out.update(pre=pre, post=post, struct=self._last_struct)
        return out

    def _insert(self, pre, action, post):
        d = self.edges.setdefault(pre, {})
        c = d.get(action)
        if c is None:
            c = d[action] = Counter()
        struct = post not in c
        if not self.temporal:
            c.clear()                      # B1 semantics: latest observation wins
        c[post] += 1
        if struct:
            self.struct_changes += 1
        self._last_struct = struct
        self.latest[(pre, action)] = post
        self.tried.setdefault(pre, set()).add(action)

    def _conflicted(self, mk) -> bool:
        return any(len(c) > 1 for n, d in self.edges.items() if n[0] == mk for c in d.values())

    def rebuild(self):
        self.edges, self.latest, self.tried = {}, {}, {}
        for mk_pre, hist_pre, action, mk_post, changed, fatal in self.log:
            pre = self.node(mk_pre, hist_pre)
            post = FATAL if fatal else self.node(mk_post, hist_pre + ((action, changed),))
            self._insert(pre, action, post)
        self.version += 1
        self.rebuilds += 1

    def remap_frames(self, new_ids: dict):
        """The HUD mask changed: masked-frame ids are re-interned and the graph rebuilt."""
        self.log = [(new_ids[a], h, act, (new_ids[b] if b is not None else None), ch, f)
                    for a, h, act, b, ch, f in self.log]
        self.L = {new_ids.get(k, k): v for k, v in self.L.items()}
        self.rebuild()

    # ---------------------------------------------------------------- queries
    def det_successor(self, node, action):
        """The successor if the edge is deterministic so far, else None (alternatives kept)."""
        c = self.edges.get(node, {}).get(action)
        if not c or len(c) != 1:
            return None
        post = next(iter(c))
        return None if post == FATAL else post

    def bfs(self, src, is_goal, max_nodes=200000):
        """Shortest path over deterministic, non-fatal edges to a node with is_goal(node).
        Returns (actions, expected_nodes, expansions) or (None, None, expansions)."""
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
            if exp > max_nodes:
                break
            for a, c in self.edges.get(u, {}).items():
                if len(c) != 1:
                    continue
                v = next(iter(c))
                if v != FATAL and v not in prev:
                    prev[v] = (u, a)
                    q.append(v)
        self.last_visited = prev.keys()
        return None, None, exp
