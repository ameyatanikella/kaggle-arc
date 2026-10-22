# GRCM T3: real-model reach of the stall probe

**Results from 22 October 2026.** Registration: [prereg_grcm_t3.md](prereg_grcm_t3.md), including Amendment 1, which was made before the freeze and before the push.

- **Freeze.** `runs/grcm/t3/FROZEN.json` holds 32 inputs. It was verified before the push and again after both outputs were downloaded.
- **Runs.** Kaggle commit runs `axizcc/arc3-grcm-t3-ab` and `axizcc/arc3-grcm-t3-ba`, version 1, pushed on 22 October at 02:04 UTC. Nothing was submitted.
- **Arms.**
  - **A:** B0.
  - **B:** B0 + `B2_HX_STALL=1, B2_HX_QUIET=1`.

  Each arm played the 25 public games for one pass, under a 6,840 s cap.
- **Raw outputs:** `runs/kaggle/grcm_t3_{ab,ba}/`.
- **Analysis:** `runs/grcm/t3/analysis.json`.

## Bottom line

**The reach gate passes.** When a stall prompt is delivered, B0's real model (Qwen3.8-Flash-Next) calls `hx_probe` on that same turn about two-thirds of the time, and it fully executed returned plans many times.

- **This answers T3's question.** HX's earlier failure, where the tool was never called, came from making the tool optional. A prompted stall turn reaches the model.
- **What this means for T6 in the [action plan](grcm_action_plan.md).** The recommendation can stay **chosen by the LLM**. Host execution is not required to get reach.

**The score comparison is not valid.** In both sessions, the arm that ran first left 9 of 25 games unplayed: 0 actions and 0 tokens. The cause is a notebook start-up fault, described under the infrastructure incident below. It hits position, not treatment, so the registered paired score and levels figures are reported but cannot be read as an effect.

## Primary endpoint: reach

| | AB session | BA session | Pooled |
|---|---:|---:|---:|
| Prompted stall turns (B arm) | 244 | 174 | **418** |
| Turns with a same-turn `hx_probe` call | 168 | 115 | **283** |
| **Same-turn invocation rate** | 68.9% | 66.1% | **67.7%** (gate ≥ 50%) |
| Plans fully executed, verified within one `action()` batch (lower bound) | 14 | 12 | **26** (gate ≥ 1) |
| Interventions retained by the host | 200 | 172 | 372 |

The registered gate is a same-turn rate ≥ 50% **and** at least one full plan executed. Both conditions hold, so **the gate passes**. The probe was called in every game that had a prompted turn, in both sessions; per-game counts are in `analysis.json → sessions[].per_game_reach`.

### Measurement cross-checks (Amendment 1; diagnostic only)

| | AB | BA |
|---|---:|---:|
| Prompt deliveries parsed from the logs / host `stall_prompts_delivered` | 254 / 254 | 182 / 182 |
| `hx_probe` snippets parsed / host `s_probe_calls` | 276 / 265 | 202 / 188 |
| Host `s_stall_triggers` / `s_stall_invocations` | 254 / 245 | 182 / 163 |

**What the cross-checks show.**
- Deliveries match the host exactly in both sessions.
- Snippets slightly outnumber host calls. That fits snippets that error before reaching the call; it is consistent under the registered zero/non-zero rule.

The original parser, before Amendment 1, would have undercounted prompted turns. The fix was validated on the local MiniMax logs before the freeze.

## Secondary outcomes (descriptive)

### Follow-through beyond full plans

| | AB | BA |
|---|---:|---:|
| Actions evaluated against a probe recommendation (`s_probe_actions_evaluated`) | 224 | 172 |
| Of those, the first action followed (`s_probe_actions_followed`) | 27 (12%) | 20 (12%) |
| Probe calls with no usable pair of models (`s_stall_proposal_failures`) | 272 | 189 |

Calling the tool is common. Acting on its first recommendation is not: about 12% of evaluated actions. Many calls produce no usable model pair.

`s_stall_levels_won` (AB 76, BA 51) counts levels completed after a stall was triggered **by any route**. It is not attributable to the probe, so it is not reported as a probe effect.

### Score and levels as registered

These are B − A, game-cluster bootstrap, seed 20261022, 90% interval:

| | Value |
|---|---|
| Mean score | +1.88 [−3.24, +6.77] |
| Mean levels | 0.00 [−0.38, +0.36] |

Session totals in levels:

| Session | A | B |
|---|---:|---:|
| AB | 75 | 96 |
| BA | 93 | 72 |

Each session's **first** arm is the low one. The pattern tracks position, not the treatment.

### Post-hoc sensitivity: not registered, descriptive only

This restricts each session to the 16 games played by both of its arms: 17 distinct games, 15 of them in both sessions.

| | Value |
|---|---|
| Mean score B − A | +4.29 [−2.09, +10.70] |
| Mean levels B − A | +0.21 [−0.18, +0.59] |

There is no sign of harm and no evidence of a gain. Two short sessions cannot resolve effects of this size.

## Infrastructure incident (must be fixed before T8)

The cause appears in the kernel log of both sessions.

1. Franzen's warm-up logic released the benchmark at its deadline, 722 s after notebook start, **while the SGLang server was still loading**: "DEADLINE at 722s from notebook start; releasing the benchmark with the server still loading".
2. The first arm's analyzer requests then failed.
3. The harness abandoned games after 11 consecutive failures: "analyzer failed 11 times in a row; giving up on this game. The endpoint is probably down".

The arm in second position started on a warm server and played all 25 games.

Nine games were unplayed in each session's first arm:

| Session | First arm | Unplayed games |
|---|---|---|
| AB | A | bp35, cn04, dc22, lp85, m0r0, sk48, tu93, vc33, wa30 |
| BA | B | bp35, cn04, dc22, ka59, lp85, m0r0, sk48, tu93, wa30 |

**Scope of the fault.**
- **It is not a T3 code error.** The B2 error count is 0, every arm exited with code 0, and the kernel logs contain no tracebacks.
- **It does not touch B0's scored path,** where the early release is intended.
- **It is new to T3.** A check on 22 October found 0 unplayed games in every arm of the earlier paired sessions (`pair_b2c`, `pair_hx_ab/ba`, `pair_hx2q_ab/ba`), so those results are unaffected. The likely cause is a server load that now takes longer than the release deadline.

**Required before the T8 main screen.** In the paired builder only, make the first arm wait for a server-ready health check before it starts. Pre-register that change as an amendment, and re-freeze before the push.

**Status, 22 October:** done, in `kaggle/eval/build_paired_ready.py` with `tests/test_build_paired_ready.py`. See Amendment 1 of [grcm_action_plan.md](grcm_action_plan.md).

## Deviations and limits

- **Position fault.** This is the only deviation, and it is described above. No threshold was changed. The gate uses only prompted turns, and those occurred only in played games.
- **No B0 vs probe score claim.** None is made, and T3 was never designed to support one.
- **Public games are contaminated** development data.
