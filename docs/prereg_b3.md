# Registration: candidate B3 (checked plan reuse, temporal state, temporal experiments, restricted dedup, object ambiguity)

Registered 2026-10-13 before any B3 code exists and before any new game start. The commit containing this file is authoritative for the time. Amendments go at the end, dated and with reasons. Criteria are never relaxed after results are seen.

**Motivation.** `/Users/ameya/Documents/Codex/2026-10-11/do/outputs/math_transfer_report.md` reports synthetic probes. They motivate experiments; they are not ARC evidence. No paper algorithm is imported wholesale.

**Objective (unchanged).** Solve unseen ARC games efficiently without training on their solutions.
- 48/48 is a target, not a promised consequence.
- The protected Kaggle hidden set is untouched. No game source code or solution is inspected; agents see only `Observation`s (frames and public metadata).
- The development suite is the 25 public games (`data/environment_files`). They are already contaminated by inspection, so no unseen-game claim follows from them.

## Frozen baselines (preserved, not modified)
- **Competition baseline B0:** Kaggle `axizcc/arc3-b0-franzen-m2-repro` v1, public LB 26.92 (submission 57112928). It is an LLM harness on GPU and is not part of the CPU development comparisons.
- **CPU development baseline B1:** `arc3/agents/explorer.py`, run `runs/b1_explorer_v1`.
  - Commit 48cb59b (dirty tree), config `{}`, `max_actions` 3000, `game_seconds` 600, seed 0, arc_agi 0.9.9, arcengine 0.9.3.
  - Result: mean score 0.1153 and 9 levels. 22 of 25 games were ended by the agent's own 600-action per-level cap.
  - B1's code stays as is. B1 is re-run under the protocol below for comparability.

## Protocol (identical for every version)
- `python -m arc3.runner --agent <spec> --games all --seeds 0,1,2 --max-actions 3000 --game-seconds 600 --workers 10`.
  - Official in-process scorecard scoring; one `make` per game; level-reset semantics.
  - B3 variants are configs of one agent class (`arc3/agents/b3/agent.py`). Every component is a switch, so effects can be attributed.
- **Seeds:** B3 shuffles candidate order with the seed. B1 is deterministic.
- **Per version and game, recorded:**
  - levels completed and SDK score;
  - actions per completed level, and which completions contributed score;
  - decision-time split (perception, inference/model update, search, validation, cache operations);
  - cache construction, lookup, reuse and invalidation counts and time;
  - prediction accuracy of known edges and change precision;
  - HUD masks, effect regions and object masks (saved per game).
- **Order of evidence:**
  1. differential checks on recorded states and traces;
  2. synthetic controls (tests);
  3. the registered development comparisons below.

## Base agent B3-base (V1), shared by all variants
- **State graph** over masked-frame keys, as in B1, maintained incrementally.
- **Never terminates the game because a local allowance ran out.** A per-level exploration allowance returns control to the parent, which resets the level and continues with expanded candidates. The game ends only on WIN or the runner's budget.
- **Candidate expansion tiers** when exploration is exhausted: component centres, then a 4-px interior grid, then a 2-px grid.
- **V1 replans** a fresh BFS to the nearest frontier on every non-local step. This preserves B1's repeated search.

## Ablations (one switch each)
| Id | Change vs previous | Switch |
|---|---|---|
| V1 | B3-base | `plan_cache=none, temporal=off, experiments=off` |
| V2 | ordinary projected plan cache: active plans kept, entries keyed by (abstract start, goal), validated lazily by re-simulating on the current graph before use | `plan_cache=projected` |
| V3 | **dependency-aware plan cache:** entries carry edge dependencies, model version, expected intermediate observations (including unchanged steps), completion and invalidation conditions, and an allowance; a contradiction invalidates only the entries that depend on the contradicted edge | `plan_cache=dependency` |
| V4 | V3 + **temporal state:** keys refined by an action/change-history suffix where a transition conflict proves hidden state (provisional, refined on counterexamples, alternatives retained) | `temporal=on` |
| V5 | V4 + **temporal experiments, shared scheduling:** repeat-until-change experiments with preconditions, predicted branches and an allowance; compatible experiments run back-to-back from one start | `experiments=shared` |
| V5s | V4 + temporal experiments, **separate** scheduling (fresh start via reset/navigation before each) | `experiments=separate` |
| D | restricted algebraic dedup: profiled first; exact noncommutative signature implemented and verified offline | profile + offline benchmark |
| O | object-correspondence ambiguity: bipartite matching with retained alternatives, on recorded traces (B3 does not consume object identities) | trace-level comparison |

## Success and regression criteria (fixed now)
Unit: sums over 25 games × 3 seeds unless noted. "Levels" means levels completed.

1. **Plan reuse (V3 vs V1).** Success requires both:
   - search expansions per decision −50% or better, and total decision time lower;
   - **no regression:** levels ≥ V1 − 1 and mean score ≥ V1 − 0.5.

   **Unique advantage over the ordinary cache (V3 vs V2):** claimed only if V3 has lower total cache plus search cost, or fewer invalidation recomputations, with equal or better levels. Otherwise **retain the ordinary projected cache** as the simpler component.
2. **Temporal state (V4 vs V3).** Success requires:
   - the synthetic controls pass (irrelevant state omitted, delayed effect after unchanged steps, context change invalidates plan, coupled objects refine);
   - and either levels ≥ V3 + 1, or witnessed hidden-state games (g50t, sc25) gain a level;
   - and no regression (levels ≥ V3 − 1).
3. **Temporal experiments (V5, V5s vs V4).**
   - **Reported separately:** valid distinguishing experiments, hypotheses eliminated, novel observations, learned rules later used successfully, levels and score.
   - **Retained** only if levels > V4, or score > V4 + 0.5, without regression.
   - **Shared beats separate** only with ≥ as many valid distinguishing experiments per action, and levels ≥ separate.
   - The destructive-effect and fresh-start controls must pass.
4. **Dedup (D).** Retained only if profiling shows duplicate algebraic hypotheses taking ≥ 5% of decision time, and the tool reduces total cost or memory with zero incorrect merges. Verification controls must pass first: distributive identity, `AB−BA`, the truncation false zero, and `A²=I` vs the free algebra.
5. **Objects (O).** Retained only if, on recorded traces, keeping alternatives raises the count of usable, model-consistent observations over commit-to-nearest, without new contradictions.
   - **Matroid feasibility** is used only for constraints verified to be matroids. If B3 has none, it is reported as not applicable.
6. **Global regression and failure rules.** These fail a variant:
   - any crash, refusal storm or harness error;
   - losing a level that V1 completes on every seed (listed per game).

## Freeze
After the development comparison, the chosen configuration is frozen by commit hash before any closed evaluation that the existing protocol permits. No closed evaluation is planned in this cycle.

## Amendments
**1. 2026-10-13, before any registered development run.** Only smoke tests on ≤ 5 games and synthetic controls had been run.
- (a) **Negative search results are cached.** Both caches reuse "no frontier reachable from node". The ordinary/projected cache invalidates on any structural graph change, tier change or rebuild. The dependency cache invalidates only when a new edge leaves the cached reachable set, or on a tier change or refinement of a member frame. Motivation: smoke runs showed that failed searches, repeated while the agent acts in an exhausted region, dominate repeated search. The brief asks to cache reusable model computations, not only plans.
- (b) **Refinement is iterative on logged evidence.** A conflict raises a frame's history length repeatedly until its logged edges are deterministic or L_MAX = 6 is reached.
- (c) **Irrelevant-state compression is implemented as goal-plan reuse projected on dependency regions.** The plan is the shortest known path to the completing step, keyed on the start frame restricted to the plan's dependency region, validated step by step on that region and invalidated on a counterexample. It is a **separate switch** (`goal_reuse`) and a separate version, **V4b = V4 + goal_reuse**. V5/V5s build on V4, not V4b, so effects stay attributable. Its criterion is the same as V4's, applied V4b vs V4.
- (d) **Temporal experiments** start only when no frontier is reachable. They use R = 5 repetitions beyond the known no-change application. "Separate" restores freshness with a level reset (charged as overhead).
- (e) **Components O and D were evaluated on traces and offline before the development runs, as registered.**
  - O: commit vs retained correspondence on B0's level-1 traces.
  - D: the offline benchmark plus the verification controls.
  - D's in-agent profile is the duplicate-action-word count in plan caches during the development runs.
- (f) Versions run: B1, V1, V2, V3, V4, V4b, V5, V5s. Criteria unchanged.
