# Pre-registration: HX (decision-relevant automata learning + contextual action algebra)

Registered 2026-10-13, before any HX code or evaluation run. Amendments are appended below with a timestamp, and always before the runs they affect.

## Question
Does a module that keeps a few executable world models, learns contextual action relations, and chooses experiments by decision value raise level completion on games it was not tuned on? Two alternatives are tested under the same compute budget:
- (a) a predictive-state / homing module for hidden-state ambiguity;
- (b) a goal-inference module that uses level transitions and object changes without a full dynamics model.

Competition score has priority over the novelty of any component. A component is kept only if it earns its place below.

## What is already measured (baseline, step 1; `scripts/hx_baseline.py` → `runs/hx/baseline.json`)
- **B0** is Franzen's M2 notebook, unchanged (Apache-2.0).
  - Franzen v3 public log, 4 passes × 25 games: mean 46.49 per run; 61.89 if each game's best pass is taken.
  - Our own B0 arm, 2 passes: 45.16.
  - Leaderboard: 26.92 (one draw). Unedited copies: 25.77 ± 3.93 per draw (forum, n > 1,200).
- **Failure modes of unfinished runs** (Franzen v3):
  - 34 stalled while deliberating (≥ 30K tokens on the level, > 1,200 tokens per action);
  - 31 stalled while acting;
  - 2 starved of compute;
  - 33 won.

## Two hosts (the LLM cannot be ablated at the required power)
1. **CPU host.** The frozen B3 graph explorer (`{"plan_cache":"projected","noop_order":true,"reset_purpose":true}`) acts as the graph-exploration fallback. HX components are added as config switches.
   - Cheap enough for the full ladder, 10 seeds and whole-family holdouts.
   - Evidence from this host is about the **decision value of the components for a model-based explorer**. It is not evidence about the LLM agent.
2. **LLM host.** B0, plus the HX tool and harness triggers (`B2_HX`).
   - The only evidence about score comes from Kaggle: paired public-game runs (contaminated, about 8-point resolution) and leaderboard draws on hidden games (SD about 4 per draw).

## Game families and split (CPU host; fixed now)
Families come from control modality and the level-1 mechanic classes in `docs/algebra_on_games.md`:
- **B (click, inverse/cyclic):** lp85, vc33
- **D (keys only):** g50t, ls20, re86, tr87, tu93, wa30
- **A (click toggles / commuting clicks):** ft09, cn04, bp35, s5i5, cd82, dc22
- **C (order-dependent clicks):** r11l, su15, sb26, lf52
- **E (mixed controls, no effective level-1 component click):** ar25, ka59, m0r0, sc25, sk48, sp80, tn36

**Tuning split:** families B and D (8 games), seeds 0–4. Any threshold or parameter may be chosen here.

**Holdout split:** families A, C and E (17 games), seeds 0–9. Run once per arm after tuning is frozen. No parameter changes after the first holdout run; a bug fix requires an amendment and a full re-run of every holdout arm.

**Caveat (stated now).** The developer has seen all 25 games in earlier work (`docs/algebra_on_games.md`). "Holdout" means held out from HX tuning, not unseen by the developer. Only the leaderboard measures unseen games.

## Budget (identical for all CPU arms)
- 3,000 actions and 600 s wall per game.
- 10 workers on the same machine.
- Official in-process scorecard.

## CPU-host arms
Arms are cumulative unless marked otherwise. The ladder order differs from the brief in one place: memory carries relations and model parameters, so it is tested after they exist.

| Arm | Content |
|---|---|
| C0 | Frozen B3 |
| C1 | C0 + faithful event log (typed events; transitions only between consecutive decision states; win frame retained; object parse with abstention). **Expected behaviour: identical to C0** (same levels and actions per seed). Overhead is measured |
| C2 | C1 + contextual action algebra: relations with context and support; refuted at the first counterexample; used for predicted edges and pruning only within a level |
| C3 | C2 + cross-level memory: relations and model parameters transfer across levels only where their context preconditions hold; they are re-verified on first use |
| C4 | C3 + candidate models (≤ 5 generic families, weighted by prospective prediction accuracy, replay fit and simplicity; return to the explorer when none is useful) |
| C5 | C4 + decision-valued probes, with goal hypotheses taken from the goal module |
| Aa | C1 + homing / predictive-state module only |
| Ab | C1 + goal-inference module only (goal-directed frontier choice from level-transition evidence, no dynamics model beyond the observed graph) |
| LOO | Full stack minus one component, each in turn (attribution) |

## CPU-host measures
- **Primary:** levels completed on the holdout (summed over seeds).
- **Secondary:**
  - official score;
  - actions per completed level;
  - decision wall/CPU time (exclusive categories);
  - predicted-edge precision;
  - relations formed / refuted / used;
  - probes taken and their outcomes;
  - goal-hypothesis accuracy at the next level;
  - errors.

## CPU-host adoption rule (per component, against the previous rung, holdout only)
- **RETAIN:**
  - the mean paired Δ levels per game (averaged over seeds) is > 0, and the 90% bootstrap CI over games has lower bound ≥ 0;
  - no new error class;
  - median decision time ≤ 2× the previous rung's.
- **REJECT:** the CI upper bound is < 0.
- **Otherwise NEUTRAL:** not retained (no evidence of value).
- **Alternatives (Aa, Ab):** compared with C2 (algebra alone) on the same holdout at the same budget, by the same statistic. The better of {C2, Aa, Ab} on holdout levels is reported as the preferred CPU module, even if it is the least novel.

## LLM host
**Integration gate (before any Kaggle push), with a mock LLM and the local harness:**
1. Every trigger fires on its scripted condition.
2. The `hx` tool's documentation is generated from the code, and every documented call works.
3. No harness exception.
4. With HX off, behaviour equals B0 (same prompts, byte for byte).
5. Token overhead of the triggered reports and per-action CPU overhead are measured.

**Kaggle paired runs.** B0 vs B0 + HX, counterbalanced (session 1 AB, session 2 BA), 2 passes per arm, using the quota that expires at 2026-10-17 00:00 UTC.
- **Reported:** paired Δ score and levels with game-bootstrap CIs, trigger counts, tool calls, report tokens, throughput, and transcripts of the largest gains and regressions.
- **Decision:** HX goes to the leaderboard only if the pooled paired Δ is ≥ −2 points (no sign of harm beyond noise) and its triggers fired. Differences under about 8 points are not interpreted as effects.

**Leaderboard draws (hidden games).** Each HX submission is one draw, compared with the B0 reference distribution (25.77 ± 3.93).
- The posterior mean and its standard error are reported after each draw.
- No claim of improvement is made unless the mean of the HX draws exceeds 25.77 by more than 2 SE.

## Final notebook rule
A feature ships only if it improves held-out performance (CPU holdout for CPU-side behaviour; leaderboard draws for the LLM agent) without a larger loss from reduced inference throughput. Otherwise the final notebook is B0, plus at most the retained switches.

## Amendments
(none yet)
