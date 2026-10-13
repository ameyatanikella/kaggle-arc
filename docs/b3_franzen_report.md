# B3 + Franzen reasoning (local Qwen): decision report

**Outcome summary**
1. **The integration works mechanically.** B3 executes and accounts for every action and returns outcomes to Franzen's unmodified tool-agent loop, which reached B3 through a sandbox channel. The full loop was verified with a scripted model and with Qwen. Inference failure never ended a game.
2. **It did not help.** Over 8 development games at matched budgets (20 min wall, 3,000 actions):
   - **levels:** frozen B3 10, B3 + Franzen/Qwen 8, unchanged Franzen/Qwen 1;
   - **later levels (≥ 2):** 4 / 1 / 0;
   - **SDK score (mean):** 1.24 / 0.65 / 0.35.
3. **The reasoning layer completed nothing.**
   - All 8 completions in the integrated arm came from B3's autonomous actions.
   - The model's own 122 actions and 17 B3-run actions completed no level.
   - It never delegated or revised B3's representation.
4. **Inference time displaced exploration.** LLM turns took 6,446 of 9,600 game-seconds (67%). That cut B3's actions (tu93: 916 instead of 3,000) and cost levels on lp85 (3 → 1) and tu93 (3 → 2).
5. **Decision: reject the integration under this backend; retain frozen B3.** The result is specific to Qwen3.5-35B on a laptop under a 20-minute budget. It is not evidence about Franzen's notebook with its Kaggle model and hardware.

## What was built
`kaggle/local/b3_franzen.py` attaches B3 at the vendored harness's only engine boundary.

**Execution and accounting.** `_HarnessGameSession._execute_action` is wrapped. Every action — the model's `action()`, B3 autonomy, B3 actions the model requested via `b3_run`, automatic resets — passes through it. B3 observes the outcome, checks its prediction (contradictions recorded) and attributes the action to its source. One authoritative harness history and one budget per game.

**Control is B3-first.**
- B3 explores and follows cached plans, through expected unchanged steps, with no LLM call per action.
- An unmodified `ToolAgent.analyze` turn runs on registered triggers: level start; no reachable unexplored state; contradiction; 300 autonomous actions without progress; delegation done; repeated guard refusals.
- After 3 consecutive analyzer failures, B3 continues for 200 actions and then retries.

**Sandbox channel and prompts.**
- `b3(...)`: status, candidates, transitions, state_frame, plan_to_frontier / plan_to_state, next / `b3_run`, delegate, revise.
- The tools are documented in the system prompt (`B3_TOOL_DOC`), and a per-turn report states why the model was called and what B3 did.
- Franzen's prompts, sandbox, retained functions, guards and memory are unchanged at file level; the patches are applied at runtime and not to the reference arm.

**Engineering fix.**
- The harness re-serializes the whole history after every action, which is O(n²): 0.23 s per action after a few hundred actions.
- Writes are now deferred during B3 autonomy (the analyzer reads the file only at turn start), giving 0.03 s per action.
- Amendment 1 in `docs/prereg_b3_franzen.md`, logged before the evaluation.

## Loop verification and the earlier Qwen failure modes (`docs/prereg_b3_franzen.md`, gate)
| Check | Evidence |
|---|---|
| Observation → proposal → validation → B3 execution → outcome → revision | Qwen (vc33): its own click proposals executed through B3 and came back as NO-OPs. It then queried B3 (candidates, transitions, plan) and executed B3's frontier actions. The revision shown is a change of strategy, not a model revision. |
| Runtime fields and tool schemas match | 0 schema errors in all of Qwen's b3 calls (verification and evaluation) |
| Failed code returns exact errors | Mock: an invalid method returned `unknown b3 method 'no_such_method'` verbatim |
| Contradictions reach the model | Turns triggered by contradictions: ls20 5, tu93 2, sp80 1 |
| Unknown alternatives do not block planning | B3 plans over deterministic edges only, by construction |
| Inference exhaustion does not end the game | Dead endpoint: 7 failed turns; B3 continued, completed level 1 and used the full budget |

## Results (`runs/b3f/eval`, `scripts/b3f_evaluate.py`)
| Game | Frozen B3: levels (actions per completed level) | B3 + Franzen/Qwen: levels (LLM wall s; LLM actions) | Unchanged Franzen/Qwen: levels (actions) |
|---|---|---|---|
| cn04 | 0 | 0 (940 s; 18) | 0 (53) |
| ft09 | 1 (1958) | 1 (1040 s; 27 + 15 via b3_run) | 0 (299) |
| lp85 | **3** (121, 365, 1731) | 1 (1013 s; 9 + 2) | 1 (**17**, SDK 2.78) |
| ls20 | 0 | **1** (1060 s; 24) | 0 (39) |
| r11l | 1 (19) | 1 (596 s; 11) | 0 (61) |
| sp80 | 1 (**5**) | 1, after 207 actions (356 s; 11) | 0 (155) |
| tu93 | **3** (388, 148, 1653) | 2 (1174 s; 11) | 0 (22) |
| vc33 | 1 (30) | 1 (267 s; 11) | 0 (402) |
| **Total** | **10 levels, 4 later; SDK mean 1.24** | 8 levels, 1 later; SDK mean 0.65 | 1 level, 0 later; SDK mean 0.35 |

**Attribution (integrated arm).**
- Actions: `b3_auto` 19,817 (97%), `llm` 122, `b3_tool` 17, `auto_reset` 456, `b3_reset` 3.
- Level completions: `b3_auto` 8, reasoning sources 0.
- About 10 LLM turns per game, mostly the 300-action allowance trigger (8–9 per game).
- Tool use: status 13, candidates 12, transitions 17, plan_to_frontier 4, next 19, state_frame 2; delegate 0, revise 0.
- The ls20 gain over frozen B3 is a B3-autonomous completion. The model's 24 actions and 56 turns changed the trajectory, but no completion is attributable to them.

**Inference.**
- Integrated arm: 195 responses, 82K completion tokens, 4.96 M prompt tokens (90% served from the prefix cache), 6,446 s.
- Unchanged arm: 442 responses, 178K completion tokens, 21.2 M prompt tokens (95% cached), essentially the whole 9,600 s.
- 2 of the unchanged arm's requests failed with an Ollama protocol error (`content: null`).

**Failure causes.**
- Integrated arm: no level 1 in cn04 (B3 fails there too, and 11 LLM turns proposed no goal); stalled after level 1 in 6 games, where the model was consulted repeatedly but never redirected B3.
- Unchanged arm: Qwen alone completed only lp85 level 1. It did so efficiently (17 actions vs B3's 121), but turns of about 22 s each left 1,077 actions for 8 games.

## Decision against the registered criteria
- (a) levels 8 vs 10: **fail**;
- (b) later levels 1 vs 4: **fail**;
- (c) completions from reasoning sources 0: **fail**;
- (d) no new error class: pass.

**Reject the integration with this backend, and retain frozen B3.**

Context arms:
- B3 adds a lot over Qwen alone (8 vs 1).
- Qwen adds nothing over B3 alone; it costs levels by consuming the shared wall-time budget.

**Limits.**
- 8 contaminated development games, 1 pass, a nondeterministic LLM.
- The backend was Qwen3.5-35B Q4 on Ollama, on a laptop, at about 22 s per response.
- **Not tested:** Franzen's own model on Kaggle GPUs, where inference is far faster and the 9 h budget is shared across 110 games. That is the configuration in which the reasoning layer could still pay off.
- The integration is preserved and reusable for that test (`--arm b3_franzen` with another `--base-url` / `--model`).

**Reproduce.**
```
bash scripts/b3f_eval.sh
python3 scripts/b3f_evaluate.py
```
