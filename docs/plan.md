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

## B2 candidate changes, in priority order (each behind its own switch)
1. **Probe-over-deliberate policy.** Change the prompt so it no longer demands minimal actions when uncertain. The agent should run cheap systematic probes instead of long reasoning, with an explicit statement that actions are cheap until the level's spend nears the human-par region. Hypothesis: more levels completed for the same tokens. Risk: more wasted actions on completed levels (bounded by the 2.6-point inefficiency pool).
2. **Systematic probe tools** in the sandbox, so the model does not rewrite them each time:
   - `probe_effects()`: try each available action once and return a compact effect table.
   - Undo-aware variants.
   - `distinguish(models, ...)`: shortest separating sequence between the model's own candidate simulators (`arc3/algebra/distinguish.py`).
   - `gf2_min_cost` for toggle puzzles.

   These save generation tokens and are measured by tool-usage counts and by B2-vs-B0 pairs.
3. **Cross-level memory pinning.** Keep the rules of solved levels and their winning action sequences in the system prompt. Another M2 winner, sirikilohit, used this; it is unmeasured by us.
4. **Stuck audit.** Add a strategy-audit prompt once a level consumes a large share of the game's budget, following lordhansolo (also an M2 winner).
5. **Throughput.** Move to the NVFP4 + built-in MTP model variant, whose reported peak decode is 1,135 vs 946 tok/s. Highest engineering risk; only attempt with GPU access.

## Evaluation protocol (fixed in advance)
- **Local dev split:** the 25 public games. All are contaminated: inspected by me, and used by Franzen and Tufa during development. No public-game split is a true holdout. The hidden LB is the only generalization evidence.
- **Primary metric:** official arc_agi score (RHAE). Also reported: levels completed, wins, actions, tokens, wall time, errors, and tool usage.
- **Paired comparison:** B2 arm vs B0 arm on the same games, ≥2 passes each, using fixed seeds where the server allows it. Report game-level paired bootstrap CIs.
- **Promotion rule:** promote a change only if its paired public-game delta in levels completed is ≥ 0, with no new error class, no runtime overrun, and RHAE not worse than −1 point. Single LB draws (SD ≈ 4) are never used to accept or reject a change.
- **Immutable fallback:** B0 stays submit-ready throughout.

## Schedule (compressed)
- **Oct 12–14:** contract verified ✔. Local evaluator ✔. B1 ✔. Math library and fixtures ✔. B0 push and smoke submission (**needs Kaggle token**). Harness local integration (mock LLM).
- **Oct 15–18:** B2 switches 1–3 implemented and integration-tested. First GPU A/B on public games (Kaggle or rented RTX PRO 6000; needs user decision).
- **Oct 19–23:** A/B the remaining switches. Submit the best B2 only when its paired public delta clears the rule above.
- **Oct 24–27:** freeze candidates. Confirm entry and team status before Oct 26 23:59 UTC.
- **Oct 28–Nov 1:** full-duration reliability runs. Final selection: B0 or the best B2, plus one hedge. Submit no later than about Nov 1 because of queue risk.
