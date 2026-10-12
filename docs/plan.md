# Implementation plan (compressed to the time remaining)

Started 2026-10-12 09:15 UTC. 21.6 days remain to the final deadline (2026-11-02 23:59 UTC).
The entry and team-merger deadline is 2026-10-26 23:59 UTC.

## Candidates
| ID | What | Status |
|---|---|---|
| **B0** | Franzen Milestone-2 notebook (Duck harness + Qwen3.8-Flash-Next W4A16 on RTX PRO 6000), Apache-2.0. Kept verbatim on the rerun path; the commit run is a 20-min smoke test (`kaggle/b0/build_b0.py`) | Built locally. **Blocked on Kaggle access** to push and submit |
| **B1** | CPU-only state-graph explorer: HUD masking, object-centred clicks, fatal-edge avoidance, costed resets (`arc3/agents/explorer.py`) | Runs. 0.115 on public-25. It is a fallback and infrastructure test, not a competitive agent |
| **B2** | B0 plus individually switchable harness changes aimed at the measured bottleneck (below) | Design stage. Needs GPU evaluation |

## Measured bottleneck (Franzen's own 100-run public-game log, parsed in `scripts/analyze_franzen_log.py`)
- Mean 46.49. The **loss from levels never completed is 50.9 points**, and the loss from action inefficiency on completed levels is only 2.6.
- On completed levels the agent uses a median of 1.55× fewer actions than the human baseline, and 80% of completed levels are at or under baseline.
- **Implication.** Spending extra real actions on probes is nearly free in score until a level's human/agent ratio drops below about 1.07, which is where `min(1.15, ratio²)` starts to bind. Thinking tokens are the scarce resource. Action-minimizing intervention design (rank / min-cost-basis selection) therefore targets the wrong cost. The relevant cost is **tokens to a correct world model**. Experiment design should trade cheap actions for expensive deliberation.

**Exchange rate** (counterfactual on the same log): scaling every completed level's actions by 1.10/1.25/1.5 costs 1.0/3.2/6.8 points. One more completed level on an unfinished run is worth +13.5 on that run (+9.1 averaged over all runs). Extra actions are therefore only worth it when they are *targeted at stuck levels*. They are not free in general.

**External evidence** (rellik13's M2 write-up, Apache-2.0 repo, single noisy runs):
- Adding advice text to prompts hurt, including at level starts.
- Removing misleading text helped.
- Pinning solved-level rules helped (+2.8).
- Longer retained history helped the most (14.5 → 22.5).
- Pruning experts (REAP) and fresh-context restarts did not help.

## B2 candidate changes, in priority order (each behind its own switch; all OFF reproduces B0 exactly)
1. **Corrections, no new advice.** B0's system prompt says both "do not get distracted by [the edge bar]" and "count remaining budget into your plans". The bar is usually the action budget, so delete the misleading line. A second candidate: the "Optimize for as few in-game actions as possible" line conflicts with the measured bottleneck. Test it *only* as a separate arm, because it may act as advice.
2. **Solved-level memory pinning.** When a level is cleared, pin facts from that level in the system prompt: the action count, the winning action tail, and the rule the model wrote ("Rule for level N:"). Port of rellik13's M85 (Apache-2.0), reported at +2.8 in their harness. In B0, history already survives level changes until the 118K→59K trim, so the expected gain is smaller. Measure "rules captured" and the number of levels reached.
3. **Systematic probe tools in the sandbox.** `probe_effects()`, `distinguish(models, ...)` and `gf2_min_cost`. They cost about 5 lines of system-prompt documentation. Measure actual tool usage. If they go unused, they are removed (pitfall: a component that controls nothing).
4. **Stuck nudge (`B2_STUCK_NUDGE`).** One short factual line when a level has consumed ≥45K generated tokens with ≤40 actions. It re-fires at most once per 45K tokens.
   - Evidence it targets a real pattern: on the level where B0's runs fail, they had used a median of only 0.73× the human-baseline actions (Franzen v3 log). In rellik13's real Flash-Next run, g50t/sk48/tr87 sat on L1 for about 20 min and 25–35K reasoning tokens with 0.18–0.31× baseline actions.
   - Pass-to-pass variance is large (m0r0 6/5/2/0, tn36 7/7/1/2). Best-pass-per-game would score 61.5 vs 46.5. Unlucky stalls are common.
   - This is advice-type text (risk per rellik13). It is lower priority than corrections and must win its own arm.
5. **Throughput.** Move to the NVFP4 + built-in MTP model variant, whose reported peak decode is 1,135 vs 946 tok/s, or tune streams and context. Highest expected value per rellik13 and Franzen ("more tokens = more levels"), but it needs GPU-side engineering and validation.

## Evaluation protocol (fixed in advance)
- **Local dev split:** the 25 public games. All are contaminated: inspected by me, and used by Franzen and Tufa during development. No public-game split is a true holdout. The hidden LB is the only generalization evidence.
- **Primary metric:** official arc_agi score (RHAE). Also reported: levels completed, wins, actions, tokens, wall time, errors, and tool usage.
- **Paired comparison:** B2 arm vs B0 arm on the same games, ≥2 passes each, using fixed seeds where the server allows it. Report game-level paired bootstrap CIs.
- **Promotion rule:** promote a change only if its paired public-game delta in levels completed is ≥ 0, with no new error class, no runtime overrun, and RHAE not worse than −1 point. Single LB draws (SD ≈ 4) are never used to accept or reject a change.
- **Immutable fallback:** B0 stays submit-ready throughout.

## Submission schedule (1/day; final score = better of the 2 selected submissions' private scores)
- **Day 0 (token arrives):**
  - Run `kaggle_ready.sh`.
  - Push B0 (20-min smoke commit).
  - Submit B0 (LB draw #1).
  - If quota allows, also push the paired A/B commit `pair-b2c` (B0 vs B2c = BAR_FIX+PROMPT_FIXES+SOLVED_MEMORY; about 8 h of quota).
- **Following days:**
  - Submit B2c once it passes the readiness gate and its paired delta is not negative. Otherwise submit B0 again; each B0 submission is another independent draw.
  - Test STUCK_NUDGE and TOOLS only in their own paired arms.
- **Final selection:** the two selected submissions should be different systems where possible (e.g. best B2 + best B0). The private score uses the better of the two, so diversity is worth more than two copies of the same system. Never choose by public score alone when its difference is within noise (SD ≈ 4).

## Frozen readiness gate for any Kaggle-pushed B2 (commit smoke run, real model, 2 games, 20 min)
A B2 variant may be submitted only if its commit run shows **all** of the following, judged against B0's own smoke run where one exists:
1. The model server reached ready, and the notebook finished with `submission.parquet` written.
2. Zero harness exceptions, and `b2_summary.errors == 0`.
3. Every enabled switch fired at least once where its trigger occurred. `bar_fix_applied ≥ 1` and the prompt fixes must appear in saved prompts. `levels_recorded` must equal the level-ups observed.
4. ≥95% of model turns produced a parseable tool call.
5. The levels cleared in 20 min on ls20+vc33 must not be fewer than B0's smoke run by more than 1. This is a gross-regression check only, not evidence of improvement.

These criteria are fixed now and are not to be relaxed after seeing results.

## Schedule (compressed)
- **Oct 12–14:** contract verified ✔. Local evaluator ✔. B1 ✔. Math library and fixtures ✔. B0 push and smoke submission (**needs Kaggle token**). Harness local integration (mock LLM).
- **Oct 15–18:** B2 switches 1–3 implemented and integration-tested. First GPU A/B on public games (Kaggle or rented RTX PRO 6000; needs user decision).
- **Oct 19–23:** A/B the remaining switches. Submit the best B2 only when its paired public delta clears the rule above.
- **Oct 24–27:** freeze candidates. Confirm entry and team status before Oct 26 23:59 UTC.
- **Oct 28–Nov 1:** full-duration reliability runs. Final selection: B0 or the best B2, plus one hedge. Submit no later than about Nov 1 because of queue risk.
