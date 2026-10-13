"""Targeted synthetic controls for candidate B3 (docs/prereg_b3.md). No ARC data involved."""
from collections import deque
from itertools import product

from arc3.agents.b3.abstraction import bisimulation_quotient, check_quotient
from arc3.agents.b3.agent import B3Agent
from arc3.agents.b3.synthetic import CorridorGame, play

BASE = {"plan_cache": "dependency", "temporal": True, "shuffle": False}
BITS = [(k & 1, (k >> 1) & 1, (k >> 2) & 1) for k in range(8)]


def run(levels, cfg, max_actions=12000):
    a = B3Agent("synthetic", cfg, seed=0)
    r = play(a, CorridorGame(levels), max_actions)
    a.finalize()
    assert r["won"] or r["actions"] >= max_actions          # never stops the game by itself
    return r, a.stats


# ---------------------------------------------------------------- complete diagnostic model
def test_complete_model_quotient_omits_irrelevant_state():
    states = list(product(range(7), range(32)))             # position x five nuisance switches

    def step(s, a):
        p, b = s
        return (min(6, p + 1), b) if a == 0 else (p, b ^ (1 << (a - 1)))
    blocks = bisimulation_quotient(states, range(6), step, lambda s: s[0] == 6)
    assert len(set(blocks.values())) == 7                   # 224 -> 7 behaviour classes
    assert check_quotient(states, range(6), step, blocks) == 0


def test_complete_model_keeps_hidden_counter_and_frame_key_fails():
    def hs(s, a):                                           # 4 invisible waits, then 'open' works
        if s == 5:
            return 5
        return min(4, s + 1) if a == "wait" else (5 if s == 4 else s)
    blocks = bisimulation_quotient(list(range(6)), ["wait", "open"], hs, lambda s: s == 5)
    assert len(set(blocks.values())) == 6                   # equal frames, distinct behaviour
    q, seen, solved = deque([0]), {False}, False            # search keyed by visible frame only
    while q:
        s = q.popleft()
        for a in ("wait", "open"):
            n = hs(s, a)
            solved |= n == 5
            if (n == 5) not in seen:
                seen.add(n == 5)
                q.append(n)
    assert not solved


# ---------------------------------------------------------------- learned model (the agent)
def test_c1_irrelevant_state_projected_reuse():
    levels = [{"switches": b} for b in BITS]
    r0, s0 = run(levels, BASE)
    r1, s1 = run(levels, {**BASE, "goal_reuse": True})
    assert r0["won"] and r1["won"]
    assert r1["actions"] < 0.5 * r0["actions"]
    assert s1["goal_reuse_started"] >= 5 and s1["goal_reuse_invalidated"] == 0


def test_c2_context_change_rejects_cached_plan():
    r, s = run([{"switches": (0, 0, 0)}, {"switches": (0, 0, 0), "blocker": True}],
               {**BASE, "goal_reuse": True}, 3000)
    assert s["goal_reuse_started"] == 0 and r["levels"] == 1 and not r["won"]


def test_c3_coupled_object_invalidates_and_recovers():
    r, s = run([{"switches": (0, 0, 0)}, {"switches": (1, 0, 0), "coupled": True}],
               {**BASE, "goal_reuse": True})
    assert s["goal_reuse_started"] == 1 and s["goal_reuse_invalidated"] == 1 and r["won"]


def test_c4_delayed_effect_after_unchanged_steps():
    lv = [{"delay": 4, "hide_switches": True}]
    r0, s0 = run(lv, {"plan_cache": "dependency", "temporal": False, "shuffle": False}, 3000)
    r1, s1 = run(lv, {**BASE, "experiments": "shared"}, 3000)
    assert r1["won"] and s1["exp_distinguishing"] >= 1
    assert r1["actions"] * 5 < r0["actions"]                # base only gets there by random fallback


def test_c5_destructive_effect_needs_fresh_start():
    lv = [{"delay": 4, "jam": True, "hide_switches": True}]
    rs, ss = run(lv, {**BASE, "experiments": "shared"}, 3000)
    rf, sf = run(lv, {**BASE, "experiments": "separate"}, 3000)
    assert ss["exp_distinguishing"] == 0 and ss["exp_valid_assumed"] > 0   # shared run was fooled
    assert sf["exp_distinguishing"] >= 1 and sf["exp_valid_assumed"] == 0 and rf["won"]
    assert rf["actions"] * 5 < rs["actions"]


def test_dependency_cache_preserves_unrelated_entries():
    from arc3.agents.b3.plans import PlanCache
    st = {}
    c = PlanCache("dependency", st)
    a = (1, ()); b = (2, ()); d = (3, ()); e = (4, ())
    c.store(a, [(4,)], [b], b, version=0)
    c.store(d, [(4,)], [e], e, version=0)
    c.on_contradiction((a, (4,)))
    assert st["cache_invalidated"] == 1 and len(c.entries) == 1
    c.on_refine(3, version=1)                                # refinement of frame 3 only
    assert len(c.entries) == 0
    c.store(a, [(4,)], [b], b, version=1)
    c.on_refine(9, version=2)                                # unrelated frame: entry preserved
    assert len(c.entries) == 1 and st.get("cache_preserved_on_refine", 0) >= 1
