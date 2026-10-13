# HX architecture: automata learning and contextual action algebra

HX is a module, not an agent. It runs beside an agent that keeps control:
- the LLM coding agent (B0 = Franzen M2 on the Duck harness);
- the CPU graph explorer (B3), which is also the fallback.

HX never takes an action by itself. Pre-registration: `docs/prereg_hx.md`. Results: `docs/hx/report.md`.

## 1. Interaction record (`arc3/hx/events.py`)
The history is h_t = (o_0, a_0, e_1, …, a_{t-1}, e_t).

**One engine call returns 1..N frames.** Each call becomes a `Step`, which records:
- the legal-action mask before the action;
- the click coordinates;
- the pre-state;
- one of these kinds:

| kind | meaning | dynamics transition? |
|---|---|---|
| `move` | frames[:-1] are animation; frames[-1] is the next decision state on the same level | **yes** (pre, a, post) |
| `win` | the level was completed. frames[-2] = WIN frame (last state of the old level; abstains if only one frame was returned); frames[-1] = INIT frame of the next level | terminal (pre, a, win) only; win→init is never diffed |
| `game_win` | the last level was completed | terminal |
| `death` | GAME_OVER; frames[-1] = death frame | terminal |
| `reset` | RESET, by the agent or automatic. It **costs an action** and returns to the level start | no |
| `refused` | the engine returned no frame (uncounted) | no |

The frame layout at a level-up was verified on the engine (vc33, lp85, ft09, r11l, sp80, tu93).

**The harness had a faithfulness gap.** Its sandbox `transitions` list paired every history entry with the previous one. Level-ups (old board → next level's first board) and resets therefore looked like game dynamics. In 146 of the 251 B0 tool calls that loop over transitions or history (out of 4,152 calls in the pair-b2c B0 arm), the code does not filter them out.

HX's events layer adds:
- `.kind` and `.level` on every transition;
- a `moves` list (current-level dynamics only);
- a `frame_diff()` default that refuses to diff across a level change, death or reset.

**Object parse (`objects.py`).** Objects are same-colour 4-connected components, outside the learned HUD mask. Matching between consecutive decision states pairs objects by colour and shape. The parse **abstains**:
- several identical objects give an ambiguous identity, and no displacement is claimed;
- touching same-colour objects merge, and this is reported as a size change;
- a frame with more than N components gives no parse.

## 2. Contextual action algebra (`algebra.py`)
**Finite-monoid fact.** In a finite system each action word acts as a transformation. The orbit of every state under an action is eventually periodic: a^{k+p} = a^k, with index k and period p (this is the finite-monoid view in `/Users/ameya/math`, Finite-Monoid Computations). HX tests these relations:
- a = 1, a² = a, a² = 1, a^m = 1;
- ab = 1, ab = ba, ab = b, aba = b.

**Instances.** An instance is one evaluation at one concrete state, possible only when every transition it mentions was observed from that state. Preconditions are part of the relation:
- ab = 1 is evaluated only where a changed the state;
- commutation only where both actions changed it.

**Context descriptors**, from coarse to fine:
- L0: the word, with clicks abstracted to (colour, size bucket) under the cursor;
- L1: L0 plus the object inventory (a mode proxy);
- L2: the concrete word plus the local 5×5 patch.

**Prediction rule.** A prediction at a new state uses the **coarsest descriptor with at least `min_support` distinct supporting states and zero counterexamples**. One counterexample disables that descriptor; finer ones remain. Nothing is generalised from a single state.

**Use.** Predictions are used only to predict an untried transition or to shorten a plan. Every prediction is scored when the transition is later observed.

**Pair effects.** Suppose the dynamics are a function of the visible state. Then two actions that each change nothing cannot change anything together. Block sensitivity (`/Users/ameya/math`, sensitivity vs block sensitivity) shows this composition argument is what licenses skipping pair tests: for general functions, single-flip insensitivity says nothing about blocks. So an observed pair effect, or any conflict where the same state and action give different results, is **evidence of hidden state**. It is reported, never modelled away.

**Measured on 100 real B0 trajectories (shadow mode, `scripts/hx_shadow.py`):**
- predictions were available for 1.6% of LLM transitions;
- their precision was 0.73;
- there were 91 hidden-state conflicts.

## 3. Goal evidence without dynamics (`goals.py`)
**Facts.** Facts are predicates over the parse:
- present / absent (colour);
- touch, inside, same row / column, same shape, equal counts (pairs of colours).

**Goal candidates.** A goal candidate is a fact true in the WIN frame and false at the level's start. Its counter-evidence is the number of earlier states of that level where it was already true without a win.

**Hazards** come the same way from death frames.

**At a new level,** candidates are re-tested on the first frame:
- one that is already true cannot be a sufficient goal there (or the goal changed);
- the rest are open, each with a heuristic distance.

**Measured on the same 100 B0 runs (291 later-level wins):**
- the top open candidate became true at the next win in 52% of cases, against 42% for a random open candidate;
- in about 30% of cases the top open candidate was also false in every earlier state of that level, so it marks the win exactly.

## 4. Candidate models and decision-valued probes
**Model form.** A hypothesis is a pair (model, goal). A model has:
- a state;
- `step(state, action)`, a transition T_i(s, a);
- an optional `observe`, a partial renderer O_i.

The goal is a predicate G_j(s).

**LLM host (`hx_probe`, sandbox).**
1. Plan with BFS for every model × goal.
2. If every plan starts with the same action → **act**.
3. Otherwise enumerate probes of length ≤ depth, including plan prefixes, so a probe can also advance a goal. Each probe partitions the hypotheses by predicted observation.
4. For each cell, take the best continuation value from the post-probe states.
5. Compute gain = E[best value after the probe] − best value now − λ·|probe|.
6. Probe only if the gain is > 0. Otherwise **follow** the plan that succeeds under the most weight.
7. If no hypothesis reaches a goal → **none**: write a better model rather than forcing one.

Hypothesis weights are optional.

**`hx_check` replays a model on the faithful `moves`.** It reports mismatches with a localisation:
- `hidden_state`: the same observation and action gave another result;
- `border`: the difference lies only in the 4-cell HUD band;
- `dynamics`: anything else.

**CPU host (`arc3/agents/hx`).** There are three generic families:
- the observed graph (B3);
- relation predictions (above);
- a guarded rigid mover per key action: a group of objects moving by (dy, dx), blocked by learned colours.

Models are weighted by **prospective** accuracy, that is, predictions made before the outcome was seen. A model prunes an action only above 0.9 accuracy on at least 8 checks; otherwise it only reorders actions. B3 remains the policy and the fallback.

## 5. Cross-level memory (`memory` switch, `Algebra.prior`)
**Carried across levels:**
- relation statistics, but not transitions;
- mover parameters.

**Re-verification.** A transferred relation counts only after one confirmation on the new level with a matching context descriptor.

**Separation.** Goal candidates are kept apart from mechanics. A win adds support to a goal candidate; it is never treated as proof.

## 6. Integration and triggers (LLM host, `kaggle/b2/hx_patch.py`, switch `B2_HX`)
Patches are applied at runtime; no harness file changes.

**Layers:**
- **`B2_HX_EVENTS`:** faithful transitions in the sandbox.
- **`B2_HX_TOOLS`:** `hx_check`, `hx_probe`, `hx_goals`, `hx_relations`, `hx_wins`. Documented in the system prompt, about 700 tokens, cached prefix.
- **`B2_HX_TRIGGERS`:** a short evidence report is appended to the next user prompt **only** when one of these fires:
  - **level completed:** the object-level changes of the winning step, the win-frame facts with their counter-evidence, and their status on the new level;
  - **hidden-state conflict:** the same board and action gave a different result;
  - **≥ 4 consecutive interior no-ops:** the no-op relations verified on this level.

**Cost and safety:**
- Reports are at most 9 lines, worth about 700–2,400 tokens per game on the B0 trajectories.
- Every hook is wrapped, so an HX failure cannot end a game; failures are counted in `hx_summary()`.
- With all switches off, the harness is untouched.

**Fallbacks.**
- If no candidate model is useful, the original coding agent continues (LLM host).
- On the CPU host, B3's graph exploration remains the policy.

## 7. Files
| Path | Role |
|---|---|
| `arc3/hx/{events,objects,algebra,goals,context,session}.py` | Library (numpy, scipy) |
| `kaggle/b2/hx_child.py` | Sandbox-side helpers (pure Python; `HX_DOC` is the model-facing text) |
| `kaggle/b2/hx_patch.py` | Harness binding |
| `kaggle/b2/b2_patches.py` | Switches |
| `arc3/agents/hx/agent.py` | CPU host (B3 + switches) |
| `scripts/hx_baseline.py`, `hx_shadow.py`, `hx_ladder.sh`, `hx_evaluate.py` | Evaluation |
| `tests/test_hx_*.py` | Correctness boundaries (not performance evidence) |
