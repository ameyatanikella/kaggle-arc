"""Experiment selection: matroid rank gain + model separation + decision relevance.

Valid guarantees (see math_contract.md):
  * `greedy_min_cost_basis`: Rado-Edmonds. For a *linear* matroid (rows of a fixed
    matrix) with fixed additive element costs, sorting by cost and keeping rows
    that raise the exact rank yields a minimum-cost basis. Nothing more.
  * `partition_entropy`: for deterministic hypotheses with the given weights, the
    entropy of the predicted-outcome partition equals the expected information gain.
    If the weights are heuristic, so is the number.
Everything else here (combined scores, regret thresholds) is a labelled heuristic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Optional, Sequence

from .linalg import IncrementalRank


# ------------------------------------------------------------ matroid greedy
def greedy_min_cost_basis(rows: Sequence[Sequence], costs: Sequence[float],
                          already: Sequence[Sequence] = ()) -> list[int]:
    """Indices of a minimum-cost set of rows completing `already` to a basis of the
    span of rows+already (linear matroid, additive costs). Ties broken by index."""
    ncol = len(rows[0]) if rows else (len(already[0]) if already else 0)
    inc = IncrementalRank(ncol)
    for r in already:
        inc.add(r)
    chosen = []
    for i in sorted(range(len(rows)), key=lambda i: (costs[i], i)):
        if not math.isfinite(costs[i]):
            continue            # unreachable experiments are not elements
        if inc.add(rows[i]):
            chosen.append(i)
    return chosen


# ------------------------------------------------------------ hypotheses
@dataclass
class Hypothesis:
    name: str
    weight: float
    predict: Callable[[Any], Hashable]          # experiment -> observable outcome
    best_action: Optional[Hashable] = None      # recommendation under this hypothesis
    meta: dict = field(default_factory=dict)


def outcome_partition(hyps: Sequence[Hypothesis], experiment) -> dict:
    groups: dict = {}
    for h in hyps:
        groups.setdefault(h.predict(experiment), []).append(h)
    return groups


def partition_entropy(hyps: Sequence[Hypothesis], experiment) -> float:
    """Expected information gain (bits) of a deterministic experiment under the weights."""
    tot = sum(h.weight for h in hyps)
    if tot <= 0:
        return 0.0
    H = 0.0
    for g in outcome_partition(hyps, experiment).values():
        p = sum(h.weight for h in g) / tot
        if p > 0:
            H -= p * math.log2(p)
    return H


def pair_separation(hyps: Sequence[Hypothesis], experiment, decision_weight: float = 4.0) -> float:
    """Weighted mass of hypothesis pairs the experiment separates. Pairs that
    recommend different actions get `decision_weight` (heuristic weighting)."""
    out = 0.0
    preds = [h.predict(experiment) for h in hyps]
    for i in range(len(hyps)):
        for j in range(i + 1, len(hyps)):
            if preds[i] != preds[j]:
                w = hyps[i].weight * hyps[j].weight
                if hyps[i].best_action != hyps[j].best_action:
                    w *= decision_weight
                out += w
    return out


def decision_relevant_pairs(hyps: Sequence[Hypothesis]) -> int:
    return sum(1 for i in range(len(hyps)) for j in range(i + 1, len(hyps))
               if hyps[i].best_action != hyps[j].best_action)


# ------------------------------------------------------------ selectors
@dataclass
class Candidate:
    name: str
    cost: float                       # all constituent real actions incl. travel/setup; inf = unreachable
    experiment: Any = None
    jac_row: Optional[Sequence] = None  # parameter Jacobian / design row of the response (if modelled)
    progress: float = 0.0             # optional: expected progress toward a plausible solution


def select_rank_only(cands: Sequence[Candidate], chosen_rows: Sequence[Sequence]) -> Optional[Candidate]:
    """Cheapest reachable candidate whose row raises the rank; None = 'rank saturated'."""
    rows = [c.jac_row for c in cands if c.jac_row is not None]
    if not rows:
        return None
    inc = IncrementalRank(len(rows[0]))
    for r in chosen_rows:
        inc.add(r)
    best = None
    for c in cands:
        if c.jac_row is None or not math.isfinite(c.cost):
            continue
        if inc.gain(c.jac_row) and (best is None or c.cost < best.cost):
            best = c
    return best


def select_separation_only(cands: Sequence[Candidate], hyps: Sequence[Hypothesis]) -> Optional[Candidate]:
    best, best_v = None, 0.0
    for c in cands:
        if not math.isfinite(c.cost):
            continue
        v = pair_separation(hyps, c.experiment) / max(c.cost, 1e-9)
        if v > best_v:
            best, best_v = c, v
    return best


def select_combined(cands: Sequence[Candidate], hyps: Sequence[Hypothesis],
                    chosen_rows: Sequence[Sequence] = (), rank_bonus: float = 1.0,
                    decision_weight: float = 4.0) -> Optional[Candidate]:
    """Heuristic: value = (decision-weighted pair separation + rank_bonus*rank_gain
    + progress) / cost. Returns None when no reachable candidate has positive value
    (nothing left that can change the decision or the identified subspace)."""
    rows = [c.jac_row for c in cands if c.jac_row is not None]
    inc = None
    if rows or chosen_rows:
        ncol = len(rows[0]) if rows else len(chosen_rows[0])
        inc = IncrementalRank(ncol)
        for r in chosen_rows:
            inc.add(r)
    relevant = decision_relevant_pairs(hyps) > 0
    best, best_v = None, 0.0
    for c in cands:
        if not math.isfinite(c.cost):
            continue
        sep = pair_separation(hyps, c.experiment, decision_weight) if relevant else 0.0
        rg = inc.gain(c.jac_row) if (inc is not None and c.jac_row is not None) else 0
        v = (sep + rank_bonus * rg + c.progress) / max(c.cost, 1e-9)
        if v > best_v:
            best, best_v = c, v
    return best


# ------------------------------------------------------------ decision-relevant stopping
def minimax_regret(policies: Sequence[Hashable], models: Sequence[Hashable],
                   cost: Callable[[Hashable, Hashable], float], fail_cost: float = 1e6):
    """Return (best_policy, its max regret, table). cost(pi, M) may be inf (= failure).
    Exact only if the true model is in `models` and costs are exact."""
    def c(pi, m):
        v = cost(pi, m)
        return fail_cost if (v is None or not math.isfinite(v)) else v
    best_by_m = {m: min(c(pi, m) for pi in policies) for m in models}
    table = {pi: max(c(pi, m) - best_by_m[m] for m in models) for pi in policies}
    pi_star = min(policies, key=lambda pi: (table[pi], str(pi)))
    return pi_star, table[pi_star], table


def should_act(policies, models, cost, regret_tolerance: float, fail_cost: float = 1e6):
    pi, reg, _ = minimax_regret(policies, models, cost, fail_cost)
    return (reg <= regret_tolerance), pi, reg
