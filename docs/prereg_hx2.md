# Pre-registration: HX2 (per-game hypothesis ledger, predictive-history state, certified algebra)

Registered 2026-10-14, before any HX2 ablation run. Design source: the user's research note "Learning an ARC-AGI-3 world model while exploring" (13 Oct). The development measurements below (shadow mode on public B0 trajectories) were made while building and are not evidence of score.

## Question
Does a per-game hypothesis ledger raise level completion on games it was not tuned on? The ledger:
- forecasts prospectively, scored before outcomes are used;
- refutes goals live, keeping prerequisites apart from sufficient goals;
- splits predictive-history state on conflicts;
- matches objects relationally, including camera shifts;
- prunes only when certified, otherwise uses predictions as soft priors.

**Falsifiers, from the design:**
- the ledger becomes more accurate but clears no extra later levels at the same budget;
- active probes spend actions without changing plans or raising completion.

## Components (code at the registration commit)
**Library (`arc3/hx/`):**
- `goal_ledger.py`: fact / AND / THEN hypotheses, live non-win refutation, statuses `sufficient?` / `prerequisite`;
- `history.py`: a fixed feature family, written before any run:
    - prev_same, prev_effective, count mod 2/3/4, prev_action;
    - parity of each other key action;
    - three competing UNDO models (stack, single-level, all-actions stack);
- `scene.py`: camera shift, plus Weisfeiler-Leman-refined matching;
- `ledger.py`: prospective forecasts, effect accuracy, change precision, failure causes, certification.

**Description-length bits:** fixed in `MODEL_BITS`.

**Certification rule:** every forecasting model agrees, each has at least 0.9 prospective accuracy on at least 5 checks of the action class, and the action is not history-dependent. UNDO is always history-dependent.

**LLM host:** `B2_HX_LEDGER=1` (`kaggle/b2/hx_patch.py`):
- the `hx_ledger()` sandbox call;
- goal-refutation, camera and enriched hidden-state reports, shown only when triggered.

**CPU host:** `arc3/agents/hx` switches `ledger`, `history`, `certify`.

## Development facts (shadow mode, 100 Franzen-v3 B0 runs, public games; `scripts/hx2_shadow.py`)
These were measured during the build; they are not evidence of score.

- **Replay fidelity:** 100 of 100 runs.
- **Graph model:** effect accuracy 0.950, covering 9.5% of actions. Algebra: effect accuracy 0.652, covering 1.4%.
- **History features:** they explain 2 of 68 real hidden-state conflicts.
- **UNDO models:**
    - stack 0.673; all-actions stack 0.682; single-level 0.439;
    - per game: su15 about 0.88, sk48 at most 0.19.
- **Goal ledger:** 879 refutations of sufficiency across 16,072 distinct states; 9 camera shifts (bp35).

## CPU-host arms (cumulative; same family split as `docs/prereg_hx.md`)
| Arm | Config |
|---|---|
| A | `{"events":true}` (= HX C1 = frozen B3) |
| B | A + `ledger` (forecasts + goal ledger driving the goal-frontier choice) |
| C | B + `history` (splits, UNDO models, camera; history-dependent actions never pruned; plans dropped when the history model predicts divergence) |
| D | C + `algebra` (trust gate) + `memory` + `models` + `probes` + `certify` (certified prune, else soft prior) |

- **Splits and seeds:**
    - Tuning: families B + D, 8 games, seeds 0–4.
    - Holdout: families A + C + E, 17 games, seeds 0–9.
- **Caveat:** the holdout games were already used once, for HX v1. They are held out from HX2 tuning, but the developer has seen HX v1's holdout outcomes on them, so they are not fresh. No other public games exist. The sealed HardGrid set is not used.
- **Budget:** 3,000 actions and 600 s per game, 8 workers, the host kept awake (`caffeinate -i`).
    - Runs that hit the time cap are reported.
    - The primary analysis is repeated on runs where both arms used all 3,000 actions.
- **Tuning:** may change only `max_pairs`, the history feature support threshold and the certification thresholds. Then the code is frozen and the holdout runs once.

## Measures
- **Primary:** holdout levels completed (paired by game and seed).
- **Secondary:**
    - per-model effect accuracy and change precision;
    - goal refutations;
    - invalid hard-prunes: certified edges that were wrong when executed;
    - goal-frontier choices;
    - plans dropped by the history model;
    - camera shifts;
    - decision time;
    - time-capped runs.

## Adoption rule (each rung against the previous one; holdout)
- **RETAIN:** the mean paired Δ levels per game is > 0; the 90% game-bootstrap CI lower bound is ≥ 0; no new error class; median decision time ≤ 2× the previous rung's.
- **REJECT:** the CI upper bound is < 0.
- **Otherwise NEUTRAL.**

**Final-notebook rule:** the LLM-host ledger (`B2_HX_LEDGER`) goes to Kaggle only after:
- the scripted-model integration gate passes (done 14 Oct: 161 `hx_ledger()` calls, 0 errors);
- one counterbalanced paired session, B0 + HX vs B0 + HX + LEDGER, when quota allows, under the existing rule (pooled Δ ≥ −2 points and triggers fired; differences under about 8 points are not effects).

A leaderboard claim needs an actual leaderboard draw.

## Amendments
**Amendment 1** (2026-10-14, after the tuning pass, before any holdout run).

**Change.** Goal-ledger facts now ignore the 4-cell border, the harness's own HUD convention.

**Why.**
- In the Kaggle B0 + HX transcripts (sb26), the win frame carried a level-complete indicator in the top row: a bar recoloured from 5 to 0.
- HX reported it as the goal fact `absent(colour 5)` (distance 260).
- Win-time UI drawn in the border is not goal evidence.

**Scope.**
- First tuning pass: A 28, B 30, C 30, D 34 levels, kept in `runs/hx2/tune_v1`.
- B, C and D are re-run on tuning with the fix. A is unaffected.
