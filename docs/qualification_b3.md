# B3 frozen candidate: qualification protocol (registered before the runs)

**Candidate (frozen).** `arc3/agents/b3/agent.py` (lean): B1's exploration policy plus the B3 base (local allowances never end the game; candidate tiers) plus the **ordinary projected plan cache** with cached negative results. The rejected components are not in the execution path; they are preserved in `arc3/research/b3_ablation` (see docs/b3_report.md). Candidate order is a deterministic per-node shuffle seeded by `crc32(node)` and the seed, so it does not depend on the order in which nodes are examined.

**Comparators (same harness, same budgets).**
- B1, `explorer` (frozen CPU baseline).
- B3-nocache: the same lean code with `plan_cache = none`. This isolates the cache.

**Protocol** (`docs/plan.md`, "Evaluation protocol"; broadened to 5 seeds):
- 25 public development games × seeds 0–4; 3,000 actions; 600 s per game; official in-process scorecard (RHAE).
- **Paired by game:** per-game means over seeds; paired deltas with a 90% game-level bootstrap CI (10,000 resamples).
- **Reported:** levels, RHAE, wins, actions, error classes, refusals, runtime overruns (`time_cap` stops, listed per run), and exclusive agent timing (perception / inference / search / cache / policy_other, wall and CPU), with runner-level `agent_seconds` and `engine_seconds`.
- **Capped runs are counted explicitly.** Per-decision costs use each run's own decision count, so a time-capped run is not compared as if it had used its whole action budget.

**Promotion rule (from docs/plan.md, unchanged).** The candidate qualifies over a comparator only if all of these hold:
- paired level delta ≥ 0;
- no new error class;
- no runtime overrun (no `time_cap` stop the comparator did not also have);
- RHAE not worse than −1 point.

The cache comparison (candidate vs B3-nocache) must additionally show the agent CPU time per decision for search plus cache lower, and no level loss.

**Known base-policy failure mode, recorded before the runs, not fixed in this freeze.** With no reachable frontier, the base resets whenever it is away from the start, without checking that the start can still reach a frontier (B1 checked this). This produces reset / fallback-action loops. It is a diagnosis target, not a qualification criterion.

## Results (2026-10-13; `runs/b3/qual/evaluation.json`, `failure_diag.json`)
**Qualification.** B3c: 53 levels, RHAE 0.253. B1: 45 levels, 0.115. B3c qualifies over B1: levels +0.064 per game (90% CI [−0.04, +0.18]), RHAE +0.14, no new errors, no overruns. B3c vs B3-nocache: outcomes are identical game by game, and search + cache CPU fell 22.9 → 2.7 µs per decision.

**Timing reconciled (exclusive, CPU).**
- The cache saves about 7.6 s of agent CPU over 125 runs: 6% of agent CPU and about 1.5% of wall time. The game engine dominates wall time (283 s, vs 202 s for the agent).
- Agent time is mostly candidate generation and untried checks (policy_other, 82 s CPU) and perception (41 s).
- The earlier "58 s" was a sum of wall time under 10-worker contention.
- **No time-capped runs occurred with or without the cache,** so the earlier time-cap avoidance did not replicate.

**Failure classification** (78 failed level-1 game-seeds; reference = B0's winning path, loop-free in B3c's masked-key space and replayed on the engine):
| Category | Game-seeds |
|---|---|
| Exploration: budget ran out with frontier remaining | **61** |
| Exploration: exhausted, then reset/fallback loops | 8 |
| Reached final state, completing action untried | 4 |
| No reference solution (sk48) | 5 |
| Perception / hidden state (loop-free reference does not win) | 0 |
| Action set (reference action not among candidates) | 0 |
| Search failure | 0 |

In the 61 frontier-at-cap runs, **56% of all actions changed nothing visible** (86–91% in ar25, dc22, m0r0, ft09, cd82; about 0% in re86 and tr87, which are breadth explosions instead).

## Change 1 (registered before running): no-op-aware candidate ordering
**Change.** `noop_order: true`. An action class (key id; or a click on a component of a given colour and size bucket) that is **historically unchanged** — it has produced no visible change at ≥ 3 distinct states, and never a change, is moved to the back of the candidate order at new states. It is never removed, so exploration stays complete. Counts persist across levels.

**Comparison.** Frozen B3c vs B3c + change 1, same protocol (25 games × seeds 0–4).

**Criteria.**
- The promotion rule above: paired level delta ≥ 0, no new error class, no runtime overrun, RHAE not worse than −1.
- **Mechanism check:** the no-op share of actions in the previously frontier-at-cap runs must fall.

Reported regardless: per-category failure counts after the change.

## Change 1 result (2026-10-13; `runs/b3/qual/B3c_noop`, `failure_diag_B3c_noop.json`)
| | B3c (frozen) | B3c + change 1 |
|---|---|---|
| Levels (25 × 5 seeds) | 53 | **61** |
| Paired levels per game | | **+0.064, 90% CI [+0.016, +0.120]** |
| RHAE (mean) | 0.253 | +0.021 [−0.032, +0.082] |
| Errors / time caps | 0 / 0 | 0 / 0 (max wall 84 s per game) |
| No-op share, previously frontier-at-cap runs | 0.564 | **0.458** |
| Failed level-1 game-seeds | 78 | 73 (frontier-at-cap 61 → 56; others unchanged) |

- **Per game:** gains in lp85 (+0.6 levels per game), lf52 (+0.4), s5i5 (+0.4), cd82 (+0.2) and ft09 (+0.2); a loss in tn36 (−0.2).
- **Decision:** **qualifies** under the registered rule, and the mechanism check passes.
- **New frozen candidate:** `--agent b3 --config '{"plan_cache":"projected","noop_order":true}'` at the commit that records this section.
- **Next largest category:** still exploration at the cap (56 game-seeds). The exhausted-looping base flaw (8) is the clearest single fix after that.


## Tightened conclusions (review of 2026-10-13)
1. **Levels vs score.** 61 vs 53 levels supports no-op-aware ordering on this development set, and the paired level CI excludes zero. The **score gain is uncertain** (RHAE +0.021, CI [−0.032, +0.082] includes zero). Finding more solutions and finding efficient solutions are separate achievements.
2. **Replay success is narrow evidence.** A winning loop-free reference path that survives B3's masked keys shows only that **those particular paths** are representable and executable. It does not establish the absence of hidden-state or perception failures: a frame-keyed model can execute a winning path while merging other histories. The earlier hidden-state evidence (g50t, sc25 conflicts; raw-key Markov conflicts) stands. The table's "0" therefore means "no failure in which the reference path itself breaks", not "no hidden-state problems".
3. **Deferral is not effective completeness.** Deferred ("historically unchanged") actions are never removed, but under a finite budget they can stay untried forever. An unchanged end frame can also hide a useful transition, e.g. information shown only in animation frames or a hidden counter. The tn36 regression (−0.2 levels per game) is examined below.

## Change 2 (registered before running): purposeful resets
**Change.** `reset_purpose: true`. With no reachable frontier from the current node, B3 resets only if the **recorded exploration graph** shows the start node reaching an unfinished frontier, or the start has untried candidates. Otherwise it does not reset: it expands candidates or keeps acting in place. No true game graph is assumed.

**Comparison.** Change 1 (`noop_order`) vs change 1 + change 2, same protocol (25 × seeds 0–4).

**Criteria.**
- The promotion rule (paired level delta ≥ 0, no new error class, no runtime overrun, RHAE not worse than −1).
- **Mechanism:** reset/fallback-loop actions in the 8 previously looping runs fall.

**Bug found by change 2's first run (6 KeyErrors in vc33 and lp85).** `_remap()` re-keyed every stored frame after a HUD-mask change, but not `start_node`.
- In the frozen candidate, the stale start key only fed the `node != start` reset test. After a mask remap that test was wrong, silently.
- Change 2 also looks the start up in the graph, which raised the KeyError.
- **Fixed** (the start key is re-keyed with everything else).
- Because the fix also changes change 1's behaviour, change 1 is **re-run with the fix** (`B3c_noop_fix`) as change 2's comparator. The erroring run (`B3c_noop_reset`, 61 levels with 6 errors) is kept and reported but not used.
