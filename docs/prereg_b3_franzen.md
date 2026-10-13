# Registration: B3 + Franzen reasoning (local Qwen) vs frozen B3 vs unchanged Franzen

Registered 2026-10-13 before any evaluation game start; the commit adding this file is authoritative for the time. Verification runs (mock model; one Qwen smoke game) are gates, not evaluation. Criteria are fixed now and never relaxed after results are seen.

## Arms (one total budget each: wall time and actions, everything charged)
1. **B3 (frozen).** `arc3/agents/b3` with config `{"plan_cache":"projected","noop_order":true,"reset_purpose":true}` (docs/qualification_b3.md, 65 levels on 25 × 5). It runs inside the same harness session as arm 2 with the reasoning layer disabled (`kaggle/local/b3_franzen.py --arm b3`), so execution path, guards and scoring are identical.
2. **B3 + adapted Franzen loop** (`--arm b3_franzen`).
   - The vendored Franzen/Duck harness is used as the reasoning layer, unedited at file level: ToolAgent, its system and user prompts, Python sandbox, retained functions, guards and history.
   - B3 owns execution and accounting: every action passes through the wrapped `_execute_action`, where B3 observes the outcome, checks its prediction and attributes the action to its source.
   - B3 acts autonomously and invokes an LLM turn on registered triggers:
     - level start;
     - no reachable unexplored state;
     - contradiction (≥ 50 autonomous actions apart);
     - 300 autonomous actions without progress;
     - delegated actions done;
     - 20 guard refusals.
   - The model gets a `b3(...)` sandbox channel: status, candidates, transitions, state frames, plan_to_frontier / plan_to_state, `b3_run`, delegate, revise.
   - Inference failures switch to B3 autonomy (retry after 200 actions); they never end a game.
3. **Unchanged Franzen notebook harness** (`--arm franzen`): `HarnessSolver` unpatched (no B3, no B2), same backend. **Recorded differences from the Kaggle notebook:**
   - backend: Qwen3.5-35B (MoE, Q4_K_M) on Ollama on an Apple M5 Pro, vs Qwen3.8-Flash-Next W4A16 + MTP on SGLang on an RTX PRO 6000;
   - 1 game at a time, vs 120 streams with priority scheduling;
   - a per-game budget of 20 min and 3,000 actions, vs about 532 min shared.

**Common.**
- Env flags: the B0 notebook's setup and priority flags (`kaggle/local/run_harness_local.py`), with `ARC3_MAX_ACTIVE_STREAMS = 1`.
- Model `qwen3.5:35b` at `localhost:11434` (context 262,144; prefix caching verified: 28.9K cached tokens, 2.9 s for an appended turn).
- Games run sequentially so the KV cache is not shared between games.
- Budget per game: **20 min wall and 3,000 actions**.
- Scoring: TAAF benchmark `final_score` (the same path for all arms). 1 pass.

**Games (fixed now; development set, contaminated).** vc33, r11l, sp80, lp85, tu93, ls20, ft09, cn04.
- The first five: frozen B3 completes level 1 in most seeds, so later levels test whether reasoning adds goal-directed progress.
- ls20 and ft09: B3 is unreliable.
- cn04: B3 never completes level 1.

## Measures (all reported)
- Levels and score per game.
- Actions per completed level.
- Later-level progression (levels ≥ 2).
- Inference wall time and turns.
- Failure causes per game: no level 1; stalled after level k (with the dominant source of actions in the stall); harness or LLM errors.
- **Attribution:**
  - actions and level completions by source: `llm` (model's action()), `b3_tool` (B3 choice requested by the model via b3_run), `b3_auto` (autonomy, including delegated), `b3_reset`, `auto_reset`;
  - which trigger preceded each LLM turn;
  - B3 tool usage by method;
  - representation revisions.

## Decision criteria (fixed now)
**Pre-evaluation gate (verification).** The complete loop must be shown with Qwen on one smoke game, **before** these evaluation runs:
- observation → LLM proposal (an action, plan or b3 call) → validation (guards / B3 legality) → execution through B3 → actual outcome back to the model → a later turn that uses the outcome or a contradiction.

Previous Qwen failure modes are checked and reported:
- runtime fields and tool schemas match the documented interface;
- failed code returns its exact error to the model;
- contradictions reach the model;
- unknown alternatives do not block planning;
- inference failure does not end the game.

**Adopt the integration (arm 2 over arm 1)** only if all of these hold:
- (a) total levels over the 8 games exceed arm 1's;
- (b) later-level completions (levels ≥ 2) ≥ arm 1's;
- (c) at least one level completion is attributed to a reasoning-layer source (`llm` or `b3_tool`, or `b3_auto` actions delegated by the model);
- (d) no new error class.

If (a) fails, or the completions are all B3-autonomous, the reasoning layer has not demonstrated a contribution and **frozen B3 is retained**.

Arm 3 is context, not a criterion:
- if arm 3 ≥ arm 2, B3 adds nothing over the LLM alone;
- if arm 1 ≥ arm 2, the LLM adds nothing over B3 alone.

With 1 pass × 8 games and a nondeterministic LLM, all comparisons are descriptive. There are no significance claims, and nothing is concluded about unseen games.

## Gate results and pre-evaluation amendments (2026-10-13, before any evaluation game)
**Verification.**
- **Mock model, scripted.** Every channel worked:
  - b3 status, candidates, transitions and plan returned the documented keys;
  - LLM actions (20) were attributed `llm`; B3 actions requested via `b3_run` (90) were attributed `b3_tool`;
  - delegation → 380 autonomous B3 actions → a "delegation done" trigger;
  - revisions were recorded;
  - an invalid method returned its exact error to the model (`unknown b3 method 'no_such_method'`);
  - contradiction triggers reached the model.
- **Qwen, vc33, 15 min** (`runs/b3f/verify_qwen_vc33`):
  - observation → Qwen's own click proposals → executed through B3 → NO-OP outcomes returned;
  - Qwen then switched to `b3("candidates" / "transitions" / "plan_to_frontier" / "status")` and executed B3's suggested frontier actions (effects observed);
  - no schema errors in any Qwen b3 call;
  - result: 2 levels, both completed by B3 autonomy; 6 LLM turns (263 s of inference).
  - No representation revision or contradiction occurred in that game, so the "useful revision" leg is shown only as a change of strategy after failed proposals, not as model revision.
- **Dead endpoint** (`127.0.0.1:9`, 3 min): 7 failed LLM turns; B3 continued (393 autonomous actions), completed level 1 and used the full budget. Inference failure does not end the game.

**Amendment 1 (engineering, all arms equal).** The harness serializes the whole history (every frame) to its runtime-state file after **every** action, which is O(n²): about 0.23 s per action by the time a few hundred actions have accumulated. The file is read only by the analyzer at the start of a turn, and `_llm_turn` writes it there. B3's autonomous actions now defer that write. Unchanged behaviour during LLM turns; the unchanged-Franzen arm is unaffected. Measured: 0.030 s per action, and the B3 arm reaches 3,000 actions within budget.

**Amendment 2.** `--minimal-diagnostics 1` (the competition-rerun setting) for every arm. Measured overhead was the same either way.
