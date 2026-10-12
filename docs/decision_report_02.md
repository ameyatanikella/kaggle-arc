# Decision report 02 (2026-10-12, about 10:20 UTC)

## What changed since report 01
- **B2 switches.** Five are now implemented and integration-tested in the real harness with mock LLMs. All are OFF by default, and OFF = B0.
  - `B2_BAR_FIX`: deletes the line that tells the model to ignore the edge bar. That bar is the action budget, and B0's own game-over text says so.
  - `B2_PROMPT_FIXES`: corrects the tool-schema claim that snippets are "ephemeral" even though function retention is on, and splits two fused bullets.
  - `B2_SOLVED_MEMORY`: port of rellik13's M85 (Apache-2.0). Pins solved-level facts and the model's stated rule.
  - `B2_TOOLS`: compute-only sandbox helpers `bfs_plan`, `distinguish`, `gf2_solve`, `zm_solve`. Spliced into the real sandbox subprocess; retention works.
  - `B2_STUCK_NUDGE`: one factual line on token-heavy, action-light levels.
- **Kaggle tooling.**
  - Paired A/B commit notebook: B0 vs B2 on the public 25 at a competition-scaled budget, each arm in its own subprocess with a timeout.
  - B2 notebook builder, readiness check script, runbook.
  - Builder regression tests proving that B0's rerun path is byte-identical to Franzen's.
- **Analysis tools.**
  - Transcript statistics (per-level turns, tokens, minutes, helper use).
  - Censoring-aware scheduler replay simulator. Its equal-split baseline reproduces Tufa's reported mean (1.55 vs 1.60).
  - Dependency/hash manifest.

## Evidence added
- **Under-exploration on failed levels.** On the level where B0 runs fail, they had used a median of only 0.73× the human-baseline actions. In a real Flash-Next transcript set (rellik13), g50t, sk48 and tr87 sat on L1 for about 20 min and 25–35K reasoning tokens with 0.18–0.31× baseline actions. This motivates `B2_STUCK_NUDGE`. It is not proof that a nudge helps.
- **Pass-to-pass variance is huge.** Matching each game's best pass would raise B0's public mean from 46.5 to 61.5.
- **Inefficiency trade-off.** Uniformly scaling actions ×1.25 costs 3.2 points. One more level on an unfinished run is worth +13.5 on that run.

## Statistical reality
A paired A/B with 2 passes × 25 games has a delta SD of about 5 points, and LB draws have an SD of about 4. Within the remaining quota, effects of a few points cannot be established statistically. Decisions will rest on three things: mechanism checks (did the switch fire and change behaviour as intended), gross-regression gates, and priors from public evidence. This is stated up front so results are not over-read.

## Next action (unchanged)
**Kaggle access** (token at `.kaggle/access_token`, and a yes/no on using the daily submission) is still the single blocker for B0's first leaderboard draw and for every model-dependent measurement.
