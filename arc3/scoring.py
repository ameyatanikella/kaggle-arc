"""RHAE replica of arc_agi 0.9.9 `EnvironmentScoreCalculator` (for analysis only).

The official numbers always come from `arcade.close_scorecard()`; this replica is
used for counterfactuals ("what if level 3 had taken 40 actions") and is pinned
to the official calculator by tests/test_scoring.py.
"""
from __future__ import annotations


def level_score(baseline: int, actions: int) -> float:
    if actions <= 0:
        return 0.0
    return min(115.0, 100.0 * (baseline / actions) ** 2)


def game_score(baselines: list[int], completed_level_actions: list[int]) -> float:
    """baselines: per-level human baseline; completed_level_actions: actions on each
    completed level, in order (len <= len(baselines))."""
    n = len(baselines)
    tot_w = n * (n + 1) / 2
    num = 0.0
    max_w = 0.0
    for i, a in enumerate(completed_level_actions[:n]):
        s = level_score(baselines[i], a)
        num += (i + 1) * s
        if s > 0:
            max_w += i + 1
    if tot_w == 0:
        return 0.0
    return min(num / tot_w, 100.0 * max_w / tot_w)


def max_score_if_completed_through(n_levels: int, k: int) -> float:
    """Upper bound on game score when exactly the first k levels are completed."""
    tot = n_levels * (n_levels + 1) / 2
    return 100.0 * (k * (k + 1) / 2) / tot
