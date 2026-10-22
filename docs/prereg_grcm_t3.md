# Pre-registration: GRCM T3 real-model reach

**Prepared 21 October 2026; pending user approval and freeze. No Kaggle push has been made for T3.**

## Question and cases

Can B0's Qwen3.8-Flash-Next call `hx_probe` on a prompted stall turn and execute a returned plan? This is a reach and integration test. It is not a GRCM performance or leaderboard test.

The fixed case set is the 25 public games: ar25, bp35, cd82, cn04, dc22, ft09, g50t, ka59, lf52, lp85, ls20, m0r0, r11l, re86, s5i5, sb26, sc25, sk48, sp80, su15, tn36, tr87, tu93, vc33, wa30. One pass per arm in each of two independent paired commit sessions. B0's stochastic analyzer has no reproducible seed exposed by this builder, so sessions are repeats, paired by game and counterbalanced by order rather than seed.

## Arms and execution

| Arm | Switches |
|---|---|
| A | B0, no B2 switches |
| B | `B2_HX_STALL=1,B2_HX_QUIET=1` |

Use `kaggle/eval/build_paired.py` with `--submit none --passes 1`: session 1 order AB, session 2 order BA. These are Kaggle **commit runs only**, never scored submissions. Each arm has a 6,840 s time cap per pass. The planned budget is approximately 8.5 RTX hours including startup.

The existing HX stall switch fires at analyzer turn 5 or after 20,000 level tokens, with a same-level changed-state prerequisite. T2's selected recent-token-rate trigger is **not** implemented in this switch; it is reserved for T6. T3 measures the existing forced-tool interface, not T2 trigger selectivity.

## Endpoints and decision

The primary denominator is every B-arm analyzer turn on which the stall prompt was delivered. The numerator is a `hx_probe` call in Python code on that same turn, verified from the request/transcript log; tool exceptions and invalid models are counted separately. Report the pooled rate, each session rate, triggered-level count and per-game counts. Cross-check against the host's `probe_calls` counter. The existing `stall_invocations` counter is diagnostic only; its code can also count a later call on a previously triggered level.

A returned plan is **executed** when at least one nonempty action plan returned by `hx_probe` is followed in full, in order, before reset or an unrelated action. The automatic count is a conservative lower bound: it verifies a complete plan within one logged `action()` batch; a continuation across batches counts only if the saved trace proves every step. Unresolved follow-through is not counted. Report full-plan count separately from first-action agreement, invalid model pairs, tool errors, and abstentions. The reach gate passes only if the same-turn invocation rate is at least 50% **and** at least one full plan is executed. If it fails, T6 uses host execution. No threshold changes after results are seen.

Secondary outcomes: completed levels and official score per game and arm, throughput, timeouts, harness exceptions, prompted turns, tool calls, first-action follow-through, and levels won after a followed plan. Paired A/B differences use a 10,000-resample game-cluster bootstrap with seed 20261022; all 25 games are resampled as units, retaining both sessions within each game. Report intervals as descriptive: T3 cannot establish a score gain. Any new error class or arm timeout is a harm signal and is reported verbatim.

## Protocol

Before the first push, freeze SHA-256 hashes of the builders, HX sources, runner, generated AB and BA notebooks and metadata, this registration, and the 25-game list in `runs/grcm/t3/FROZEN.json`. Verify hashes before each push and after both outputs are downloaded. Smoke the generated notebooks locally for parseability and switch/order/submit values. Download both outputs, preserve raw logs, run the frozen analysis, write `docs/grcm_t3_reach_report.md`, and append both sessions to `runs/ledger.jsonl`.

Each Kaggle push requires a separate user confirmation under `AGENTS.md` §2 and `docs/grcm_action_plan.md`; approval of this registration alone does not authorize both pushes. No leaderboard submission is in scope.

## Amendment 1 (21 October 2026, made before the freeze, before any T3 push, and before any T3 outcome existed)

This amendment came out of a review of `scripts/analyze_grcm_t3.py` against the four local MiniMax stall runs, `runs/openrouter_minimax_stall{,_v2,_v3,_v4}_ls20`. Those runs used the same `B2_HX_STALL` / `hx_probe` path. No Kaggle T3 data exist yet.

**1. Parser defect, fixed.** When a snippet executes no actions, the harness re-prompts within the same `analysis_step` and resets `request_index_within_turn` to 1. The original parser overwrote a turn's "prompted" flag on every such request. A later re-prompt without the stall text therefore erased an earlier delivery.

On the MiniMax logs the original parser undercounted badly:

| Run | Prompted turns, original parser | Host deliveries | Same-turn calls, original parser | Same-turn calls, true |
|---|---:|---:|---:|---:|
| v1 | 1 | 2 | 0 | 1 |
| v2 | 1 | 3 | 0 | 2 |
| v4 | 0 | 2 | 0 | 1 |

That would have biased the reach gate toward failing. A turn is now "prompted" if **any** request in that `analysis_step` delivered the stall prompt.

**2. Denominator, clarified.** The primary denominator is unchanged: analyzer **turns**, meaning distinct `analysis_step` values on the B arm, on which at least one stall prompt was delivered. The numerator is turns in that set whose logged code contains an `hx_probe(` call. The script also reports **deliveries**: requests whose final user message carries the stall prompt.
- One turn can receive several deliveries. In MiniMax v3, two were delivered within step 2.
- Deliveries are diagnostic only and do not enter the gate.

**3. Counter names, corrected.** The host counters live in `arm_b/arm_meta.json → b2_summary.hx` under these names:
- `stall_prompts_delivered`
- `s_probe_calls`
- `s_stall_invocations`
- `s_stall_triggers`
- `s_probe_actions_followed`

They replace the names `probe_calls` and `stall_invocations` used above. The analysis now reports, per session:
- parsed deliveries against `stall_prompts_delivered`, which must match exactly; they do on all four MiniMax runs;
- parsed `hx_probe` snippets against `s_probe_calls`. Agreement here means both are zero or both are non-zero, because one snippet may call the tool more than once (v1: 1 snippet, 2 calls).

**A failed cross-check is reported as a measurement defect.** It never changes the gate.

**4. Unchanged.** The arms, the cases, the reach-gate thresholds (same-turn rate ≥ 50% and at least one full plan executed), the bootstrap, and the push and approval rules are all as stated above.
