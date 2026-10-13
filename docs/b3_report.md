# Candidate B3: checked plan reuse, temporal state, experiments, dedup, objects. Development report

**Outcome summary**
1. **Dependency-aware plan reuse passed its registered criterion.**
   - Search expansions per decision −97% (3.37 → 0.10).
   - Decision time 146 s → 88 s.
   - Levels and SDK score unchanged (32 levels; 0.197). Without the cache, 6 of 75 runs hit the 600 s time cap.
2. **It beat the ordinary projected cache only marginally:** cache + search 0.57 s vs 0.62 s, same levels. In practice the simpler cache is about equivalent, as the toy probe found.
3. **Temporal state (history refinement) failed:**
   - 27 levels vs 32 (su15 lost on every seed);
   - conflicts doubled rather than resolving;
   - prediction accuracy fell from 0.989 to 0.972.

   Goal-plan reuse passed all synthetic controls but triggered once in development, correctly invalidated.
4. **Temporal experiments:** separate (fresh-start) scheduling beat shared scheduling (29 vs 28 levels; all 349 preconditions verified vs 21 of 368).
   - Neither produced novel observations or later-used rules.
   - Both sit on top of the rejected temporal state and stay below V3 (32).
5. **Restricted algebraic dedup and object-ambiguity tracking did not earn a place.**
   - Dedup: B3 has no algebraic hypotheses; duplicate plans are identical action words at 0.1% of decision time.
   - Object ambiguity: retaining alternatives gave fewer usable, model-consistent observations than nearest-match commitment (811 vs 963, both 0 contradictions).

## Candidate, baselines, protocol
- **Registration:** `docs/prereg_b3.md` (commit a907b9f, before any B3 code), plus amendment 1 (before any registered run).
- **Code:** `arc3/agents/b3/`, frozen for the comparison at commit **ad6f1b8**.
  - `graph.py`: temporal graph;
  - `plans.py`: none / projected / dependency caches;
  - `abstraction.py`: quotient checks, dependency regions;
  - `agent.py`;
  - `synthetic.py`: controls;
  - `correspond.py`: objects.
  - Also `arc3/algebra/ncsig.py`: exact noncommutative signature.
- **Frozen baselines (unchanged):**
  - B0 (Kaggle LLM harness, LB 26.92), not part of this CPU comparison.
  - B1 (`explorer`, commit 48cb59b, `{}`, 3000 actions / 600 s). Re-run here: 27 levels, 9 per seed, deterministic.
- **Protocol:** 25 public development games × seeds 0, 1, 2; 3,000 actions; 600 s per game; official in-process SDK scorecard. Agents see observations only; no game sources or solutions. The protected hidden set was not touched, and no unseen-game claim is made.

## Development comparison (`runs/b3/dev/`, `evaluation.json`)
| Version | Levels (seeds) | SDK score | Median actions per completed level | Search exp./decision | Decision time (perception / inference / search / policy) | Cache built / hits / invalidated | Negative-cache hits | Prediction acc. / change precision |
|---|---|---|---|---|---|---|---|---|
| B1 | 27 (9,9,9) | 0.115 | 242 | n/a | n/a | — | — | — |
| V1 base | 32 (9,9,14) | 0.197 | 320 | 3.37 | 146 s (41 / 19 / 4.6 / 82) | — | — | 0.988 / 0.992 |
| V2 ordinary cache | 32 (9,9,14) | 0.197 | 320 | 0.10 | 103 s (34 / 15 / 0.40 / 53) | 2288 / 1264 / 932 | 18,425 | 0.989 / 0.993 |
| **V3 dependency cache** | **32 (9,9,14)** | **0.197** | 320 | **0.10** | **88 s (28 / 13 / 0.47 / 47)** | 2236 / 1326 / 1981 | 18,511 | 0.989 / 0.992 |
| V4 + temporal state | 27 (8,8,11) | 0.196 | 223 | 0.13 | 90 s | 2972 / 1961 / 2688 | 5,278 | 0.972 / 0.992 |
| V4b + goal reuse | 27 (8,8,11) | 0.196 | 223 | 0.13 | 84 s | (as V4) | 5,278 | 0.972 / 0.992 |
| V5 + experiments, shared | 28 (8,9,11) | 0.196 | 273 | 0.13 | 80 s | 3024 / 2019 / 2725 | 4,584 | 0.973 / 0.991 |
| V5s + experiments, separate | 29 (9,9,11) | 0.196 | 322 | 0.13 | 91 s | 2997 / 1974 / 2675 | 5,814 | 0.975 / 0.992 |

**Which completions contributed to score** (V1 to V3 are identical):
- The mean is dominated by cheap level-1 completions: r11l (115 / 110 / 26 per seed), vc33 seed 2 (49), sp80 (19 / 5 / 3) and tn36 seed 2 (6.9).
- All other completions score under 3 points per level: the explorer completes them with far more actions than the human baseline.
- B1 → V1 (+5 levels, 0.115 → 0.197) comes from no longer ending games on a local cap. tu93 gained 2 and 4 levels on seeds 1 and 2; lf52 lost a level on 2 seeds.

**Records.** Per run, `agent_stats` holds the HUD mask (`hud_mask_hex`, packed bits), the effect bounding box per action (`effect_bbox`), graph counters and timers. Per-object masks are not recorded: B3 does not segment objects for planning.

## Component decisions against the registered criteria
1. **Plan reuse, V3 vs V1: PASS.**
   - Expansions −97%; decision time −40%; levels 32 = 32; score equal; no regressions.
   - **Unique advantage over the ordinary cache (V3 vs V2):** the registered criterion is met (cache + search 0.567 s vs 0.617 s; expansions 21,529 vs 22,093; equal levels), but by a small margin.
   - V3 invalidates eagerly (1,981 entries) where V2 rejects lazily on re-simulation (932 rejections after 2,872 validation steps).
   - In development no refinement occurs under V3, so the dependency cache's unique feature — preserving unrelated entries on refinement — is shown only in tests.
2. **Temporal state, V4 vs V3: FAIL.** −5 levels: su15 on every seed, lp85 seed 2, tu93 seed 2.
   - **Why:** 581 refinements, but conflicts *rose* from 220 to 437, with 21 unresolved. The history suffix did not capture the hidden variable in these games. It multiplied nodes, each with fresh untried candidates, which slowed exploration.
   - **On recorded traces** (`scripts/b3_differential.py`): conflicts are real (g50t 4, sk48 18), but refinement did not improve one-step prediction (0.768 vs 0.771) because revisits are rare.
   - **Goal reuse (V4b): FAIL** for development value. It started once and was invalidated once (a correct rejection) on development games, despite passing every synthetic control.
3. **Temporal experiments.** Against V4, both pass the registered retain rule (V5 28 levels, V5s 29), but both stay below V3's 32 because they sit on the rejected V4.
   - **Separate:** 349 of 350 experiments ran from verified fresh preconditions; 1 was distinguishing; 338 reset actions of overhead.
   - **Shared:** 346 of 368 ran under *assumed* preconditions; 13 were distinguishing.
   - Hypotheses eliminated: 1,745 (separate) and 1,835 (shared). Novel observations: 0. Learned rules later used successfully: 0.
   - **Shared does not beat separate** (levels 28 < 29). The destructive-effect control shows shared scheduling can be fooled.
   - **Not adopted:** there is no registered evidence on the retained base (V3).
4. **Restricted algebraic dedup: NOT RETAINED.**
   - **Verified first** (`tests/test_ncsig.py`, 8 tests):
     - the O(d) recurrence equals the triangular matrix;
     - the distributive identity is exactly zero;
     - `AB−BA` is nonzero although scalar substitution misses it;
     - truncation to d = 8 gives a false zero on A⁸, and d = 9 detects it;
     - free `A²−I` ≠ 0 while toggle semantics give A² = I;
     - 120 random formulas agree with independent expansion;
     - exact agreement with the report's probe implementation.
   - **Offline** (`runs/b3/dedup_bench.json`):
     - the random-matrix identity test was fastest (< 1 ms, < 8 KiB);
     - the exact signature beat expansion only on memory at (A+B)¹⁶ (5.1 MB vs 36.7 MB);
     - it was 380× slower on 60 random formula pairs.
   - **In the agent:** no algebraic hypotheses exist. About 53% of cached plans duplicate an existing action word, caught by string equality; all cache work was 0.1% of decision time.
5. **Object ambiguity: NOT RETAINED** (`runs/b3/objects_eval.json`).
   - The candidate games had up to 29 identical shapes (tu93).
   - Retained, motion-gated alternatives gave 811 usable model-consistent observations, against 963 for nearest commitment. Neither produced a contradiction.
   - Plain enumeration needed motion gating to be tractable. No counting algorithm was warranted.
   - **Matroid feasibility: not applicable.** B3 has no portfolio or independence constraint, so nothing was labelled a matroid.

## Synthetic controls (`tests/test_b3_controls.py`, 8 pass) and their limits
| Control | Result |
|---|---|
| Irrelevant state, complete model | 224 states → 7 behaviour classes; 0 inconsistent edges |
| Hidden counter, complete model | 6 states kept distinct; frame-keyed search fails |
| C1 irrelevant switches (learned) | Projected goal reuse: 1,096 vs 4,072 actions, 0 wrong reuses |
| C2 context change | Stale plan rejected at its precondition |
| C3 coupled object | Plan invalidated mid-execution; level still solved |
| C4 delayed effect | Experiments: 136 actions vs ~2,600 for the base (which got there by random fallback) |
| C5 destructive effect | Shared experiments fooled; fresh-start scheduling solves it in 134 actions |

Certificates from the learned model are conditional on that model; none is claimed to be unconditionally correct.

## Remaining failure modes
- Exploration completeness, not search, limits levels:
  - 225,000 decisions produced 32 completions;
  - ft09's toggle space is explored with about 48 mostly useless clicks per state;
  - the fallback took 13,239 actions in V3.
- History-suffix refinement is the wrong representation for hidden variables in the development games. A model of counters and modes, plus targeted experiments, would be needed.
- Scores are dominated by a few cheap completions. Most completions cost far more than the human baselines.

## Reproduce
```
.venv/bin/python -m pytest -q tests/test_b3_controls.py tests/test_ncsig.py
.venv/bin/python scripts/b3_differential.py
.venv/bin/python scripts/b3_objects_eval.py
.venv/bin/python scripts/b3_dedup_bench.py
bash scripts/b3_dev_runs.sh
.venv/bin/python scripts/b3_evaluate.py
```

**Frozen retained configuration:** commit ad6f1b8, run with `python -m arc3.runner --agent b3 --config '{"plan_cache":"dependency"}'`. No closed evaluation was run.

## Decision
**Retain only specified components:** the B3 base (local allowances return control to the parent and never end the game; candidate tiers) with **dependency-aware plan and negative-result caching (V3)**. Reject the rest for now: temporal history refinement, projected goal reuse, temporal experiments, restricted algebraic dedup and object-ambiguity tracking. If future runs show no cost difference between the caches, the ordinary projected cache is an acceptable equivalent.
