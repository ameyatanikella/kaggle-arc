# Pre-registration: algebraic world-model diagnostics (candidate "DIAG")

Registered 2026-10-12, about 22:00 UTC. This was written **before** any code for the module existed and before any shadow-mode run. Changes after this point are allowed only as dated amendments at the bottom, each with a reason. Criteria are never relaxed after results are seen.

## Purpose and status
DIAG is a separate experimental candidate. It is **not** part of any submission notebook. It checks and diagnoses an agent's world model using the algebraic structure of each model component.

Its matroid layer may only:
- organize independent constraints;
- propose candidate dependent sets.

Actual polynomial relations decide consistency. DIAG never selects exploration by rank gain.

The motivation is synthetic results supplied by the user (no advantage for rank-first exploration; consistency checks matched a robust baseline on single faults; weak on multiple faults; caching helped only while assumptions held). These are motivation, not evidence for ARC.

## Arms (matched model, observations, budgets)
- **A** Existing agent, no checker. In shadow mode this is the recorded B0 trace itself.
- **B** A plus a strong ordinary checker:
  - exact-arithmetic RANSAC consensus fits over minimal subsets;
  - a consensus-ensemble prediction check, with "uncertain" when consensus fits disagree;
  - leave-one-out residuals;
  - the same applicability filtering as C.
- **C** B plus algebraic dependency diagnostics:
  - rank from the parameter-to-observation map;
  - circuits and their exact polynomial relations;
  - minimal-hitting-set diagnoses over violated relations, never assuming a single fault;
  - contradiction vs ambiguity vs unverifiable classification;
  - discrete-ambiguity (sign/symmetry) checks at full rank;
  - targeted reobservation or distinguishing-context suggestions.
- **C-nocache** C with structural caching disabled. This arm exists only so that caching costs and benefits are attributable separately from the diagnostics.

## Stage 1: deterministic fixtures (gate for stage 2)
Every fixture must pass for C before any shadow run. B's outputs are recorded, not asserted.

| Fixture | C must |
|---|---|
| F1 correct nonlinear polynomial model | report consistent, 0 false alarms |
| F2 one incorrect observation | name exactly that observation as the unique minimal diagnosis; the repaired model predicts held-out observations exactly |
| F3 multiple incorrect observations (2 and 3) | include the true fault set among its minimal diagnoses; never report a unique single-observation diagnosis when none exists; suggest a discriminating reobservation when diagnoses are ambiguous |
| F4 full rank with unresolved sign ambiguity | report local identifiability **and** the unresolved sign; propose a context that resolves it; never claim unique identification |
| F5 changed context invalidates a cached relation | never combine observations across the changed context; invalidate the dependent cache entries (counted); no false alarm |
| F6 incomplete hypothesis family | blame the hypothesis (family inadequate), not a small set of observations |
| F7 misleading perception/tracking | identity-uncertain observations are excluded from relations and reported as unverifiable; with untagged swaps, the contradiction is localized to the post-swap observations, not reported as a single fault |

## Stage 2: shadow mode on development traces (public games only)
**Data.**
- B0's recorded level-1 attempts on the 25 public games (`data/franzen_v3_output/transcripts`), replayed on the offline engine to regenerate observations.
- These games are a contaminated development set. No protected or hidden game is used.
- No generalization claim follows from stage 2.

**Components** (proposed online from observations seen so far; no engine internals):
- (i) mover kinematics, linear over Q: displacement per action is constant under a stated applicability condition;
- (ii) click toggles, linear over GF(2): a cell is the parity of presses of the buttons that flip it.

Anything else is labelled unsupported and left to ordinary prediction checking.

**Behaviour.** No behavioural effect: checkers only observe.

**Metrics** (all reported, losing results included):
- **M1 Encounter.** Fraction of eligible steps with ≥ 1 checkable relation (C); violated-relation events.
- **M2 Injected-fault localization** on real traces. Corrupt k ∈ {1, 2, 3} observations at random, 30 trials per trace and component, fixed seeds. For each arm:
  - exact localization rate;
  - correctly-ambiguous rate (true set among the reported diagnoses and ambiguity flagged);
  - **confident-wrong rate** (a unique fault set reported and it is wrong);
  - precision and recall.
- **M3 False alarms** on clean segments (no injection, in which the component predicts every eligible observation correctly).
- **M4 Repair outcome.** After repair (drop the diagnosed observations, refit), accuracy of predictions for the next eligible observations.
- **M5 Cost.** Per-step compute (median and p95), structural setup, cache memory, invalidations, and recompute time, for C and C-nocache.

## Success and failure criteria (fixed now)
C "beats" B in shadow mode only if **all** of these hold:
- **S1 (real decisions).** ≥ 1 checkable relation on ≥ 30% of eligible steps in ≥ 5 development games.
- **S2 (multi-fault).** Pooled over k ∈ {2, 3}:
  - C's confident-wrong rate ≤ B's − 10 percentage points;
  - and C's (exact + correctly-ambiguous) rate ≥ B's.

  For k = 1, C's exact localization ≥ B's − 2 percentage points.
- **S3 (no extra false alarms).** C's false alarms ≤ B's on clean segments.
- **S4 (repair, no regression).** C-repaired next-step accuracy ≥ B-repaired − 2 percentage points.
- **S5 (cost).** C's added compute per step: median ≤ 50 ms, p95 ≤ 500 ms on this machine; cache memory ≤ 50 MB per game.

Further rules:
- A reduction in rank computations, repeated actions or contradiction counts is **not** sufficient by itself.
- If S2 and S4 show no advantage beyond the thresholds, **retain B** (the simpler implementation).

**Stage 3** (behavioural A/B/C with matched planner, proposer, model backend, observations and total budgets, scored with the SDK) runs only if stage 2 passes. It also needs an agent whose planner consumes explicit world-model components. The current submission agent (B0, an LLM harness) does not, so stage 3 needs that integration plus GPU budget, and it will be pre-registered separately.

## Out of scope / prohibited behaviours (checked by tests)
The module never:
- bans zero-rank or repeated actions;
- interrupts a plan for rank;
- claims unique identification from full rank;
- ends a game on its own budget.

When it is unsupported, out of budget or raises an exception, the safe fallback is ordinary prediction checking (B).

## Amendments
(none)
