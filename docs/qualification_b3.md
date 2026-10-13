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
