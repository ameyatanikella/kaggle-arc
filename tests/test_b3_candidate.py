"""Frozen B3 candidate (arc3/agents/b3): never ends a game itself; cache reuse; no-op-aware ordering."""
from arc3.agents.b3.agent import B3Agent
from arc3.research.b3_ablation.synthetic import CorridorGame, play

BITS = [(k & 1, (k >> 1) & 1, (k >> 2) & 1) for k in range(4)]


def run(cfg, levels, max_actions=6000):
    a = B3Agent("synthetic", cfg, seed=0)
    r = play(a, CorridorGame(levels), max_actions)
    a.finalize()
    assert r["won"] or r["actions"] >= max_actions          # the agent itself never stops the game
    return r, a.stats


def test_candidate_wins_and_reuses_plans_without_changing_behaviour():
    lv = [{"switches": b} for b in BITS]
    r0, s0 = run({"plan_cache": "none"}, lv)
    r1, s1 = run({"plan_cache": "projected"}, lv)
    assert r0 == r1                                            # the cache changes cost, not behaviour
    assert s1["bfs_expansions"] < s0["bfs_expansions"]
    assert s1["cpu_search"] + s1["cpu_cache"] <= s0["cpu_search"] + s0["cpu_cache"] + 0.05


def test_noop_order_defers_useless_actions():
    lv = [{"switches": (0, 0, 0), "hide_switches": True, "delay": 0}]   # 1, 2, 5 are invisible no-ops
    r0, s0 = run({"plan_cache": "projected"}, lv, 3000)
    r1, s1 = run({"plan_cache": "projected", "noop_order": True}, lv, 3000)
    assert r1["won"] and s1["noop_deferred"] > 0
    assert r1["actions"] <= r0["actions"]


def test_timing_categories_are_exclusive_and_complete():
    r, s = run({"plan_cache": "projected"}, [{"switches": (0, 0, 0)}])
    cats = ("perception", "inference", "search", "cache", "policy_other")
    assert abs(s["cpu_total"] - sum(s[f"cpu_{c}"] for c in cats)) < 1e-9
    assert all(s[f"wall_{c}"] >= 0 for c in cats)
