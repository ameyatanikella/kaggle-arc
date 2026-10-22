# Action plan: a guarded relational causal model (GRCM) inside B0, by 2 November 2026

Written 21 October 2026. It turns the research review of that date into dated tasks, gates and a quota budget. **It is a plan, not a registration.** Each test below still needs its own `docs/prereg_*.md`, the user's approval, and a freeze before it runs ([AGENTS.md](../AGENTS.md) §8).

## Goal and constraints

**Goal.** Build GRCM and test it inside B0 before the deadline. GRCM learns conditional effects of actions on other objects and turns them into executable probes. "Inside B0" means it is an OFF-by-default switch behind B0's existing stall interface. The result is either a promotable second final selection, or a clean negative.

**Constraints.**
- **The submission stays B0** unless GRCM passes the existing promotion rule in [plan.md](plan.md).
- **Kaggle quota.** 45 h are available now, and they reset around 28 October to another 45 h. The user offered this quota for this plan. Confirm each push before making it. A leaderboard submission needs an explicit "yes" each time.
- **What the evidence says to aim at.**
  - About 95% of event steps are HUD changes or remote effects of keys and clicks. Local contact is 0.2% ([world_model_failure_analysis.md](world_model_failure_analysis.md)).
  - Better prediction alone changed no play ([world_v2_play_report.md](world_v2_play_report.md)).
  - Optional tools go unused: HX's `hx_probe` was never called.
  - RANK's gain largely vanished against a placebo ([causal_report_v2.md](causal_report_v2.md)).

## Timeline

| Date (2026) | Track | Task | Done when | Compute |
|---|---|---|---|---|
| **21–22 Oct** | Local | **T1. Diagnose why none of the 2,629 programs verified.** Use the E3 runs in `runs/causal_id/B/E3/` and `arc3/nsw/e3.py`. Assign each proposal one failure class: representation, perception, identification or verification. Also count "consistent but not yet tested" separately from "contradicted" | A table of classes with counts, and 5 worked examples per class | Mac |
| **21–22 Oct** | Local | **T2. Audit trigger selectivity** on the 100 B0 transcripts (`scripts/audit_stage1_stalls.py`). Compare the "5 turns" trigger (catches 92% of unfinished levels, but fires on 69% of completed ones) with a few candidates: hypothesis disagreement, goal contradiction, a token-per-action spike | Best trigger chosen: ≥70% recall on unfinished levels with ≤30% firing on completed levels. If none achieves that, the least-bad one is recorded | Mac |
| **22 Oct** | Kaggle, block 1 | **T3. Real-model reach test.** Pre-register, then push B0 vs B0 + `B2_HX_STALL=1 B2_HX_QUIET=1` with `build_paired.py`, one pass per arm, counterbalanced over two sessions (AB, BA). Question: does Qwen3.8-Flash-Next call `hx_probe` when the trigger fires, and does it execute the returned plan? | Both sessions finished. Invocation rate, plan execution, levels and score per arm are reported | about 8.5 h (2 sessions × 2 arms × 1.9 h + boot) |
| **23–26 Oct** | Local | **T4. Build GRCM** as new files under `arc3/grcm/`. Details below | Unit tests pass. No oracle imports | Mac |
| **26 Oct** | Admin | **T5. Entry and team deadline** (23:59 UTC). Confirm the competition rules are accepted and the team is final | Confirmed on Kaggle | none |
| **27 Oct** | Local | **T6. Wire GRCM behind the stall interface.** The switches are `B2_GRCM=1` and `B2_GRCM_PLACEBO=1`. The execution mode comes from T3: chosen by the LLM if Flash-Next used `hx_probe`, otherwise **host-executed** with a cap on plan length. Mock-LLM integration test included | With all switches off, B0 is byte-identical (builder regression tests). With switches on, the mock run produces a recommendation ledger | Mac |
| **27–28 Oct** | Local / OpenRouter | **T7. Reach screen** with MiniMax (approved) on ls20 plus 3–4 stall-heavy games, for example sk48, wa30, lf52 and g50t. **Gate R:** see below | Gate R passed or failed. **If it fails, stop:** no Kaggle spend on GRCM | Mac + OpenRouter |
| **28 Oct, evening** | Kaggle | **T8. Freeze and push the main screen,** if Gate R passed. Pre-register first. Two contrasts, each counterbalanced over AB and BA sessions of one pass per arm: **GRCM vs PLACEBO** (primary) and **B0 vs GRCM** (harm). That is 4 sessions | All 4 sessions pushed by 29 October, early morning (UTC). RTX queues reach 9–17 h | about 17 h |
| **29–30 Oct** | Kaggle | **T9. Optional replication session** of the primary contrast, if the result is close | — | about 4.3 h |
| **31 Oct** | Decision | **T10. Apply the pre-registered rule.** Results get collected and the report written | `docs/grcm_report.md` written and the ledger appended | none |
| **1 Nov** | Submit | **T11. Final selections.** B0 plus GRCM if it passed; otherwise B0 alone. Each submission only with the user's "yes" | Both selections made on Kaggle | Scored reruns don't use quota |

## T4: what GRCM is

Each part reuses existing code.

1. **Evidence records.** Each action produces one record:
   - the action's source object, taken from `arc3/mcw/objects.py`;
   - every changed target, kept separate;
   - the HUD class: budget bar, counter or status, or unknown. This replaces the blanket mask.

   Level transitions are taken from `arc3/hx/events.py` and are never diffed as dynamics.
2. **Rule beam.** For each (action type, target), keep a small set of candidate rules of the form guard ⇒ effect.
   - **Guards** are conjunctions of at most 2 typed literals: enabled, adjacent, linked, counter > 0, or colour.
   - **Effects** are no-effect, set, toggle, move or delete.
   - Search goes from simple to complex, scored by error plus complexity in the OHCAM style.
   - Toggle and counter rules are expressed with the existing GF(2) and Z_m algebra (`arc3/algebra/`).
3. **Rule status.** Each rule is one of: untested, consistent but underdetermined, contradicted, or supported by experiment. An observed non-effect counts against a rule only when perception of its target is certain.
4. **Probe selection.** Among reachable, legal probes, pick the one whose predicted outcomes differ most across the surviving rules, then filter to probes relevant to the current goal candidate. The recommendation is a structured object: hypotheses, action plan, expected outcomes, and the evidence it rests on.
5. **Placebo.** It uses the same trigger, the same prompt and token overhead, and the same length of plan, but **draws a random target from the same probe set**.

**Hard rule.** GRCM never imports `arc3/causal_id/engine.py` or the benchmark scripts. It sees only frames, actions and public flags.

## Gates

None of these thresholds may be relaxed after outcomes are seen. They are copied into each test's pre-registration.

| Gate | Where | Pass condition | If it fails |
|---|---|---|---|
| **T1 outcome** | Local | Informational. It sets T4's priority: fix the representation, the perception, or the verifier first | — |
| **T3 reach** | Kaggle | Records whether Flash-Next calls the tool on ≥50% of triggered stalls and executes its plan at least once | It decides the T6 mode. If the model rarely calls the tool, use host execution |
| **R, reach** | MiniMax | All of: invoked on ≥80% of eligible stalls; changes ≥5% of next actions on triggered levels; ≥1 level won through an executed GRCM plan; 0 harness exceptions | Stop GRCM for the deadline and report it. The block-2 quota stays unused |
| **Promotion** | Kaggle | GRCM − PLACEBO in levels is ≥ 0, with GRCM ahead on at least 2 of the 3 sessions. B0 − GRCM shows no harm by the [plan.md](plan.md) rule: paired delta in levels ≥ 0 within noise, no new error class, no runtime overrun, official score not worse than −1. The readiness gate passes on the commit smoke run | Submit B0 only. GRCM stays a research result |

**What a pass can mean.** About 3 paired sessions can't resolve effects under about 8 points. A pass makes GRCM an eligible second final selection. It is not evidence of a gain.

## Quota budget

| Block | Use | Hours |
|---|---|---:|
| 1 (now to 28 Oct) | T3 reach test | about 8.5 |
| 1 | Smoke and readiness commits, re-runs | about 3 |
| 1 | Reserve, which expires at the reset | about 33 |
| 2 (after 28 Oct) | T8 main screen, 4 sessions | about 17 |
| 2 | T9 replication | about 4.3 |
| 2 | Commit smoke for the submission notebook | about 1 |
| 2 | Reserve | about 22 |

Block 1 has slack before it expires. If T3 finishes early, one B0-only session (about 4 h) would measure session-to-session noise directly; the promotion decision can use that. Pre-register it with T3.

## Kill criteria

Stop GRCM for the deadline, leave B0 as the submission, and write the negative report if any of these happens:
- T4 is not working by 26 October: unit tests failing, or no rule ever reaching "supported" on the procgen sealed splits;
- Gate R fails on 28 October;
- the main-screen notebooks can't be pushed by 29 October, early morning UTC.

## Risks

| Risk | Mitigation |
|---|---|
| The trigger fires mostly on levels B0 would clear anyway (69% for the current one) | T2 picks a tighter trigger. Host execution caps plan length |
| Object perception is usable on only 10 of 17 player games, and click-only games are untested | GRCM abstains when it can't identify a target. Report coverage per game |
| Queue delays push results past 31 October | Push by the morning of 29 October; no later pushes for decisions |
| Sleep or Ollama incidents during local screens | AC power, lid open, `caffeinate -i -s`; `num_ctx` 16384 ([AGENTS.md](../AGENTS.md) §2) |
| Another session edits `kaggle/b2/hx_patch.py` or `hx_child.py` (both currently uncommitted) | Coordinate before T6. GRCM goes in new files with a thin hook only |

## Amendment 1 (22 October 2026, after T3 and before any T8 build, freeze or push)

**Why.** T3 found a start-up fault in the paired commit notebook ([grcm_t3_reach_report.md](grcm_t3_reach_report.md)).
- Franzen's start-up cell releases the benchmark about 722 s after notebook start, even while SGLang is still loading.
- The first arm then abandons games after 11 failed analyzer requests. That left 9 of 25 games unplayed in each T3 session's first arm.
- Earlier paired sessions were not affected: each arm of `pair_b2c`, `pair_hx_*` and `pair_hx2q_*` had 0 unplayed games.

Without a fix, every T8 contrast would be confounded by arm position.

**Change.** T8 and T9 notebooks are built with `kaggle/eval/build_paired_ready.py`, not `build_paired.py`.
- **Same arguments,** plus `--ready-timeout`, default 1,800 s.
- **The wait.** Before the first arm, the paired cell polls the same `http://127.0.0.1:8001/health` endpoint the start-up cell uses. It records `server_ready = {ready, waited_seconds}` in `paired_meta.json`.
- **If the server is never ready,** no arm is played, and that is stated in the log and the metadata.
- **Scope.** Only the commit run is affected; the competition rerun path is byte-identical. `build_paired.py` stays unchanged, because it is a frozen T3 input.
- **Tests.** `tests/test_build_paired_ready.py` has 5 tests, which pass alongside `tests/test_notebook_builders.py`. They show three things:
  - only the paired cell and the title cell differ from `build_paired` output;
  - the wait is called only under `not TRUE_SUBMISSION` and before the arm loop;
  - the wait handles both "ready after 503s" and "never ready", including running the patched cell end to end.

**Session validity rule, to be copied into the T8 pre-registration before its freeze.** A T8 or T9 session is valid only if all of these hold:
1. `paired_meta.json → server_ready.ready` is `true`;
2. every arm played all 25 games: no `summary.txt` line reads `actions=0, tokens=0`;
3. both arms exited with return code 0 and no timeout.

**If a session is invalid:**
- report it verbatim;
- exclude it from every contrast;
- re-run it once, quota permitting. If it can't be re-run, the contrast is reported as incomplete; it is never analysed with the invalid session.

**Other T8 inputs.**
- **Freeze contents.** The T8 freeze must include `build_paired_ready.py`, `build_paired.py`, `run_arm.py`, the B2 and HX sources, the built notebooks and their metadata, the T8 pre-registration and its analysis script.
- **Budget.** The wait adds at most about 30 minutes per session, which fits inside the T8 budget: a 6,840 s × 2 arm session plus start-up and the wait stays under the 9 h notebook limit.
