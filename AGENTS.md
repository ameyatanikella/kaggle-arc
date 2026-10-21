# AGENTS.md: a guide to `kaggle-arc` for AI agents

This is the working repository for an **ARC Prize 2026 ARC-AGI-3** Kaggle entry, plus a research programme on causal world models built around it. Read this file first; it tells you what the project is, what has already been tried and decided, the rules you must follow, and where everything lives.

Last surveyed: **21 October 2026**. Facts that change quickly (branch, test count, run status, permissions) are dated; re-check them before relying on them.

---

## 1. The one-paragraph summary

The competition notebook is **B0**: a faithful reproduction of Daniel Franzen's Milestone-2 solution, an LLM coding agent (Tufa Labs' Duck harness + TAAF, Qwen3.8-Flash-Next served by SGLang on one RTX PRO 6000). It scored **26.92** on the public leaderboard. Everything else in this repo is an attempt to beat it, and **nothing has passed its pre-registered promotion rule yet**. B0 loses almost all of its points on **levels it never finishes** (50.9 points of loss on public games) rather than on inefficiency (2.6 points). Most recent work is local research on causal identification and a "mother–child" neurosymbolic world model. None of it is near the competition notebook.

---

## 2. Rules before you do anything

These come from the user and from incidents that have already cost runs. They override convenience.

### Compute and remote services

The permission records disagree, so **ask the user before any remote action**.

- **Local only (user memory, 20 October 2026).** The memory note says to run only on this Mac: CPU, the Mac's GPU, and local Ollama models. It rules out Kaggle pushes, Kaggle quota and submissions, and cloud or rented GPUs. The ban is on remote compute, not on LLMs: local LLM arms are expected.
- **Kaggle quota restored (`docs/long_term_67_program.md`, also 20 October).** That document says the user restored Kaggle quota on 20 October, with 45 h of GPU time refreshing on 24 October.
- **Submissions and public notebooks.** These always need an explicit "yes" from the user in chat. There is **one scored submission per day and one pending per team**.
- **OpenRouter.** `minimax/minimax-m3.1-flash:free` was approved for harness tests. It is remote: prompts and frames leave the machine.

### Secrets

| Secret | Location | How to use it |
|---|---|---|
| Kaggle token | `.kaggle/access_token` | `KAGGLE_API_TOKEN=$(cat .kaggle/access_token) ...` |
| OpenRouter key | `.openrouter/api_key` | `LOCAL_ANALYZER_API_KEY="$(cat .openrouter/api_key)"` |

Both files are git-ignored. Pass them only as environment variables. Never print, log, commit or copy them, and never accept a key pasted into chat.

### Integrity and process

- **Pre-register every comparison and get the user's approval before running it.** The registration is a `docs/prereg_*.md` file stating endpoints, thresholds and the decision rule. Freeze code by sha256 in `runs/<study>/FROZEN.json`. Never relax a criterion after seeing results. Changes to infrastructure go in as dated amendments, before any outcome is seen.
- **Never edit frozen files.** That means any file listed in a `runs/**/FROZEN.json` (there are 13 manifests), and `arc3/agents/b3/agent.py`, which is the frozen B3. Treat `arc3/runner.py` as frozen too: in-flight runs import it per process. To change behaviour, add a new file or a subclass.
- **Report honestly.** State negatives plainly, never hand-edit results, and append every run, including the losing ones, to `runs/ledger.jsonl` with `scripts/ledger_add.py`.
- **Competition score ranks above validating an architecture.** A better offline prediction does not count unless play changes.
- **Agents never read game source.** No agent may read game source, engine internals or solutions. Oracle and ground-truth code exists only for labelling:
  - `arc3/causal_id/engine.py`
  - `scripts/perception_bench.py`
  - `scripts/causal_id_bench.py`

  These must never be imported by an agent.
- **The public games are contaminated.** All 25 public games are development data that we and the upstream authors have inspected. Only the hidden leaderboard measures generalisation.

### Git and the shared checkout

- **Branch and commit (21 October 2026).** The branch is `causal-agent`. HEAD is `53338e9`, from 13 October. Most work since then is **uncommitted**: 12 modified tracked files and about 630 untracked paths.
- **Commit only when the user asks.** Commit only files you changed.
- **Other sessions share this checkout.** Never stash, reset, switch branches or delete other people's files.

### Machine hazards

- **Sleep contaminates timed runs.** Both `time.time()` budgets and runner wall caps keep counting while the machine sleeps.
  - Run long jobs on AC power with the lid open, under `caffeinate -i -s`. Note that `-s` works only on AC, and lid-close sleep cannot be prevented.
  - Check for sleep with `pmset -g log | grep -E " Sleep | Wake "`.
- **Ollama.** Always pass `"num_ctx": 16384`. One call without it loaded a second 34 GB copy of the model and swapped the machine. Run LLM harness games **one at a time**, because concurrent games evict each other's KV cache.
- **Multiprocessing.**
  - Run pools with `OMP_NUM_THREADS=1`.
  - Kill pool workers if you abort, because orphans keep running.
  - Don't wait on a process with `pgrep -f <pattern>` inside a shell loop: the loop's own command line matches the pattern and waits forever. Wait on a PID or a done-file instead.

---

## 3. Competition essentials

The full, sourced version is in `competition_contract.md`.

### Deadlines

| Event | Time |
|---|---|
| Entry and team merger | **26 October 2026, 23:59 UTC** |
| Final submission | **2 November 2026, 23:59 UTC** |

Scored reruns take up to 9 h, and RTX queues of 9–17 h have been reported, so start the last real submission by about **1 November**.

### Submissions

- 1 per day, and only 1 pending per team; a second pending submission fails with `FAILED_PRECONDITION`.
- You may select **2 final submissions**. The private score is the better of the two.
- The notebook runs for at most **9 h with the internet off**. The model must be open source (OSI).
- Competition reruns do **not** consume the weekly GPU quota. Commits and paired evaluation runs do.

### Interface (`arc-agi==0.9.9`, `arcengine==0.9.3`)

- **Observations.** Each one has:
  - 1..N frames of size 64×64; the last frame is the decision state, and earlier ones are animation;
  - `state`: `NOT_FINISHED`, `WIN` or `GAME_OVER`;
  - `levels_completed`;
  - `available_actions`.
- **Actions.**
  - RESET is 0. ACTION1–5 and ACTION7 are simple actions. ACTION6 is a click at `(x, y)` with both in 0..63.
- **Competition mode.**
  - There is one `make` per game, and every RESET is a **level** reset.
- **Reset accounting.**
  - A RESET costs **+1 action** and never refunds anything.
  - After `GAME_OVER`, only RESET is meaningful.
  - A RESET at a level's action 0 still costs an action, so never send one.
- **Idle scorecards.** A scorecard idle for 15 minutes may close, so never let a run go that long without an action.

### Scoring

The official calculator is replicated in `arc3/scoring.py` and pinned to it by tests.

- Level score: `s_l = min(115, 100·(baseline_l / actions_l)²)`. The action count includes resets, and an unfinished level scores 0.
- Game score: `Σ l·s_l / Σ l`, a weighted mean that favours later levels, capped at the share of weight the completed levels carry.
- The total is the mean over games.
- Each public game's human baselines are in `data/environment_files/<game>/<id>/metadata.json`, under `baseline_actions`.

---

## 4. What has been tried, and the verdicts

Read this before proposing anything; most obvious ideas have a recorded result. Unless stated otherwise, these are scores on the 25 contaminated public games.

### Leaderboard and the paired evaluations behind it

| Component | Verdict | Where |
|---|---|---|
| **B0** (Franzen M2 repro) | **Public 26.92.** That is within unedited copies' 25.8 ± 3.9, so it is a faithful reproduction. B0 is the submit-ready default | `kaggle/b0/`, `HANDOFF.md` |
| B2-algebra v2: B0 + bar/prompt fixes + solved-level memory + algebra sandbox tools | Public 24.20. No detectable effect | `docs/decision_report_03.md` |
| B2c paired vs B0 | Inconclusive: −5.60 [−15.44, +3.18]. **The same configuration run twice differed by +7.3**, and the second arm in a session loses about 12% of its throughput | `docs/decision_report_03.md` |
| Stuck nudge (`B2_STUCK_NUDGE`) | Not adopted | same |
| HX (events, contextual algebra, goal ledger) inside B0 | Both Kaggle paired exposures were negative: v1 −3.86 [−8.25, +0.35]; quiet HX2 −3.27 [−7.13, +0.74]. `hx_probe` was **never called**. Dropped | `runs/ledger.jsonl`, `runs/hx/` |
| DIAG (algebraic diagnostics) | Failed its pre-registered S2. Not adopted | `docs/decision_report_diag.md` |

### CPU agents and their LLM add-ons

| Component | Verdict | Where |
|---|---|---|
| **B1**, a CPU state-graph explorer | Score 0.115. Pipeline check only | `arc3/agents/explorer.py` |
| **B3**, the frozen CPU explorer | 65 levels over 25 games × 5 seeds. Mean score 0.275 on the same 0–100 scale where B0 gets about 46. Not competitive alone | `docs/b3_report.md`, `docs/qualification_b3.md`, `runs/b3/qual/B3c_noop_reset_fix` |
| B3 + Franzen with local Qwen3.5-35B | Rejected: 8 vs 10 levels, because LLM time displaced exploration | `docs/b3_franzen_report.md` |
| Causal agent v1 and v2 with local `qwen3.8:27b` (NAME, PROPOSE, GOAL2) | All three failed the "≥ 3 more levels" rule. CORE equals B3 | `docs/causal_report.md`, `docs/causal_report_v2.md` |
| RANK, a CPU rank-triggered interruption | +8 levels vs CORE2, but only +2 [−12, 20] vs a random-target PLACEBO. The gain is plausibly perturbation, not goal inference | `docs/causal_report_v2.md` |

### Mother–child world model and perception

| Component | Verdict | Where |
|---|---|---|
| Mother–child steps 1–2 (substrate, procgen, typed library) | Built. On procgen sealed splits, CHILD beats Q0 on all 4 splits. LIB beats CHILD on 2 of 4 | `docs/substrate_step1_report.md`, `docs/step2_report.md`, `docs/sealed_splits_report.md` |
| Step-2 child on the public games | No change: LIB − Q0 is −3 levels [−9, +3]. The child acts on 0.5% of actions, because its k-cell perception assumption breaks | `docs/child_public_report.md` |
| Object perception v1 | Agent IoU rises from 0.15 to 0.68. Usable on 10 of 17 games with a player | `docs/perception_bench_report.md` |
| World model v2 | Event-step prediction rises from 8.1% to 39.9% | `docs/world_v2_report.md` |
| World model v2 as the child's plan check | **0 change in all 500 paired runs** | `docs/world_v2_play_report.md` |
| Causal identification, Part B (local mother–child, E0–E3) | B1 and B2 fail. The symbolic child **never engaged**: 0 of 2,629 proposed programs were verified | `docs/causal_identification_report.md` |

### In-progress and rescue work

| Component | Verdict | Where |
|---|---|---|
| Causal identification, Part A (MiniMax via OpenRouter in B0's harness) | Running or pending as of 21 October | same |
| MiniMax stall screens, v1–v4 on ls20 | Integration checks only. v4 cleared level 1 through an executed probe plan, in a single stochastic run | `docs/minimax_stage1_screen.md` |
| Stall rescue by search (external Codex dir) | No action sequence of 5 or fewer clears any stalled level. Frozen B3 clears **0.10 of 67** stalled levels within 120 actions, and none at level ≥2 | §9 below |
| Goal-inference benchmark | Drafted, not approved, not run | `docs/prereg_goal_bench.md` |

### Facts that should shape any plan

- **Unfinished levels dominate the loss.** One extra cleared level on an unfinished run is worth about +13.5 points on that run. Scaling every completed level's actions by 1.25 costs only 3.2 points.
- **Variance is large.** B0's best pass per game would score 61.5 against a mean of 46.5. Of the 67 levels B0 left unfinished in one pass, B0 cleared 40 in another.
- **Measurement noise is large.** A single leaderboard draw has an SD of about 4. Paired Kaggle sessions cannot resolve effects under about 8 points, so always counterbalance arm order (AB and BA).
- **More tokens help, a little.** +20% tokens is worth about +2.3 points.
- **B0 fails on understanding, not on searching.** Its failures are goal inference and mechanic discovery (`docs/failure_analysis_b0.md`). Blind search does not rescue its stalls.
- **B0's model cannot run here.** It needs 94–120 GB. Any performance claim about B0 needs Kaggle; local LLMs and mock servers are integration checks only.
- **The long-term plan** is `docs/long_term_67_program.md`: keep B0 in charge and add switchable causal tools inside its loop, each gated by reach checks before score checks.

---

## 5. Repository map

### Top level

| Path | What it is |
|---|---|
| `README.md` | Candidate list and quick reproduction. Some counts are dated 12 October |
| `competition_contract.md` | Verified rules, interface, scoring and hardware. Every claim is status-coded V/O/P/U |
| `math_contract.md` | Each mathematical principle used, with its assumptions, code location and pinning test |
| `HANDOFF.md` | General project handoff (13 October) |
| `HANDOFF_causal_monitor.md` | Handoff for the causal-agent evaluation runs (15 October) |
| `HANDOFF_causal_identification.md` | Handoff for causal identification and the mother–child model (20 October); the newest |
| `docs/` | Plans, pre-registrations, reports. See the index below |
| `arc3/` | The Python package: environment, agents, research modules |
| `kaggle/` | Notebook builders, harness patches, paired-evaluation and submit scripts |
| `scripts/` | Run, benchmark and analysis entry points |
| `tests/` | pytest suite |
| `data/` | Public games (`environment_files/`, git-ignored) and B0's public-game output (`franzen_v3_output/`) |
| `runs/` | All run outputs, freeze manifests and `ledger.jsonl` |
| `build/b0src/` | Reconstructed B0 harness: `ARC3-Inference` (with its own `.venv`) and `tufa-arc-agi-framework`. Git-ignored |
| `third_party/` | Upstream sources: duck-harness, franzen-m2, kaggle-starter, siriki-m2. Git-ignored |
| `manifests/manifest.json` | Dependency and hash manifest |

### `arc3/`: core files

| File | What it does |
|---|---|
| `env.py` | `CompetitionEnv` wraps one `arc_agi` game with competition semantics and exact action accounting. `Observation` is the only thing agents see |
| `agent_api.py` | `Agent.act(obs, deadline) -> Act(action, x, y, why)`. Returning `None` stops the game |
| `runner.py` | Local evaluator. Runs one process per game and scores with the official in-process scorecard. Agent registry: `random`, `explorer` (B1), `b3`, `b3_ablation`, `hx`, `causal` |
| `scoring.py` | RHAE replica for counterfactuals. Official numbers always come from the scorecard |
| `perception.py` | Components, background colour, `VolatileMask` (HUD masking), diffs |

### `arc3/`: subpackages

| Subpackage | What it holds |
|---|---|
| `agents/b3/` | **Frozen B3**: graph explorer plus checked plan cache. Rejected ablations live in `research/b3_ablation/` |
| `agents/explorer.py` | B1, the CPU state-graph explorer |
| `agents/random_agent.py` | Uniform random legal actions, for pipeline checks |
| `agents/causal/` | Object-centric causal agent on top of B3 (`agent`, `model`, `planner`, `goals`, `clicks`, `llm`, `rank`, `placebo`). `substrate.py` is controller Q with an explicit trigger/target ledger |
| `agents/hx/` | CPU host for the HX study (B3 + switchable HX components) |
| `hx/` | HX module beside an LLM host: faithful typed events, contextual algebra, goal ledger. It never acts. Design: `docs/hx/architecture.md` |
| `algebra/` | Exact algebra tools: Boolean/Möbius, linear algebra over ℚ, GF(2) and Z_m solvers, experiment design, distinguishing sequences. `ncsig.py` is the noncommutative zero test |
| `diag/` | DIAG algebraic diagnostics. Not adopted |
| `mcw/` | Mother–child world model, steps 1–2 and later: `cells.py` (k-cell perception), `models.py` (CHILD and LIB), `child.py`, `child_v2.py`, `gf2.py`, `objects.py` (object perception v1), `world_v2.py` and `world_v2_mech.py` (world model v2), `public.py` (read-only recorder) |
| `nsw/` | Neurosymbolic arms for causal identification: E1 neural child (numpy CNN ensemble with curiosity RL), E2 mother, E3 symbolic. Vendored `arccausal` and `arcduck` |
| `causal_id/` | Laws vocabulary (`laws.py`) and an engine oracle (`engine.py`, **truth construction only**) |
| `learning/causal_beliefs.py` | Belief bank: predictions recorded before actions, field-level credit |
| `procgen/` | Procedural 64×64 worlds with known mechanisms, optimal-solution baselines and sealed splits. Same `Observation` interface (`docs/procgen.md`) |
| `viz/render.py` | Render frames to PNG in the ARC palette |

### `kaggle/`

The runbook is `kaggle/RUNBOOK.md`.

| Path | What it is |
|---|---|
| `b0/build_b0.py` | Builds the B0 notebook. It is verbatim on the rerun path, with a 20-minute smoke test on commit. `harness-changes.patch` is Franzen's diff to duck-harness@7652836 |
| `b2/build_b2.py` | **Builds every notebook you push.** It inserts the input-path resolver |
| `b2/b2_patches.py` | Switchable patches, all OFF by default, where OFF equals B0. Switches: `B2_BAR_FIX`, `B2_PROMPT_FIXES`, `B2_SOLVED_MEMORY`, `B2_TOOLS`, `B2_STUCK_NUDGE`, `B2_HX*` |
| `b2/sandbox_tools.py` | Algebra tools exposed in the sandbox |
| `b2/hx_patch.py`, `b2/hx_child.py` | The HX binding and the `hx_probe` stall tool. **Uncommitted edits by another session** |
| `eval/build_paired.py`, `eval/run_arm.py` | Paired A/B commit notebooks. Use `--order ba` to counterbalance |
| `local/run_harness_local.py` | Runs B0's harness locally against any OpenAI-compatible endpoint |
| `local/mock_llm_server.py` | Scripted mock LLM for integration tests |
| `local/b3_franzen.py` | The B3 + Franzen integration |
| `kaggle_ready.sh`, `submit_algebra.sh`, `submit_when_free.sh` | Read-only readiness check, and submission helpers. `submit_when_free.sh` waits out the one-pending limit |

### `scripts/` (62 entries), grouped

| Group | Scripts |
|---|---|
| Setup | `download_public_games.py`, `initial_frames.py` |
| B0 log and transcript analysis | `analyze_franzen_log.py`, `transcript_stats.py`, `sched_replay.py`, `noop_loops.py`, `audit_stage1_stalls.py`, `causal_harness_extract.py` |
| Paired and Kaggle evaluation | `paired_compare.py` (game-level bootstrap), `hx_kaggle_eval.py` |
| B3 | `b3_dev_runs.sh`, `b3_qualify.sh`, `b3_evaluate.py`, `b3_qual_eval.py`, `b3_failure_diag.py`, `b3f_*` |
| Algebra on games | `algebra_on_games.py`, `bfs_on_games.py`, `summarize_algebra_on_games.py` |
| HX and DIAG | `hx_*`, `hx2_*`, `diag_shadow.py`, `diag_evaluate.py` |
| Causal agent | `analyze_causal*.py`, `rank_goal_test.py`, `analyze_rank_*.py`, `run_*_rank*` |
| Mother–child and world model | `run_step2.py`, `run_sealed.py`, `run_child_public.py`, `perception_bench.py`, `world_v2_bench.py`, `run_world_v2_play.py`, `procgen_baseline.py`, and the matching `analyze_*.py` |
| Causal identification | `causal_id_bench.py`, `run_causal_id.py`, `run_causal_part_a.py`, `train_causal_mother.py`, `analyze_causal_id.py`, `analyze_causal_part_a.py` |
| Bookkeeping | `ledger_add.py` |

### `docs/` index

- **Read first:** `plan.md` (candidates, evaluation protocol, promotion rule, the frozen readiness gate for any Kaggle push) and `long_term_67_program.md` (the current long-term direction).
- **Decisions:** `decision_report_01.md` to `03.md`, `decision_report_diag.md`.
- **Pre-registrations** (`prereg_*.md`), each paired with its report:

| Pre-registration | Report |
|---|---|
| `prereg_b3` | `b3_report`, `qualification_b3` |
| `prereg_b3_franzen` | `b3_franzen_report` |
| `prereg_hx`, `prereg_hx2` | results in the ledger and `runs/hx*` |
| `prereg_diag` | `decision_report_diag` |
| `prereg_causal` | `causal_report` |
| `prereg_causal_v2`, `prereg_rank_*` | `causal_report_v2` |
| `prereg_substrate_2x2` | `substrate_step1_report` |
| `prereg_step2_library` | `step2_report` |
| `prereg_sealed_splits` | `sealed_splits_report` |
| `prereg_child_public` | `child_public_report` |
| `prereg_world_v2_play` | `world_v2_play_report` |
| `prereg_causal_identification` | `causal_identification_report` |
| `prereg_goal_bench` | unrun |

- **Analyses:**
  - `failure_analysis_b0.md`
  - `analysis_scheduler_elasticity.md`
  - `algebra_on_games.md`
  - `algebra_verification.md`
  - `perception_bench_report.md`
  - `world_v2_report.md`
  - `world_model_failure_analysis.md`
  - `minimax_stage1_screen.md`
- **Research design:** `mother_child_world_model.md`, its review `mother_child_world_model_review.md`, `research/mcw/`, `procgen.md` and `hx/architecture.md`.
- **Hash drift:** `prereg_sealed_splits.md` and `prereg_step2_library.md` no longer match their `FROZEN.json` hashes, because both were edited minutes after freezing. That predates the 17 October perception session, which flagged it.

---

## 6. How to run things

Run everything from the repo root with the repo's virtualenv: `.venv/bin/python` (Python 3.12, `arc-agi 0.9.9`, `arcengine 0.9.3`, numpy, scipy). The repo venv has **no torch**; the neural child is pure numpy.

### Setup and tests

```bash
.venv/bin/python scripts/download_public_games.py data/environment_files   # the 25 public games, via the official client
.venv/bin/python -m pytest -q tests                                          # 190 passed in about 4 min on 21 Oct 2026
```

### CPU agents with official scoring

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m arc3.runner --agent b3 --games all --seeds 0,1,2,3,4 \
  --max-actions 3000 --game-seconds 600 --workers 12 \
  --config '{"plan_cache":"projected","noop_order":true,"reset_purpose":true}' --out runs/<name>
```

- `--games` takes comma-separated id prefixes (for example `ls20,vc33`) or `all`.
- Each game writes `games/<id>.result.json`, `.log.jsonl` and `.frames.npz`.
- Each run writes `summary.json` and `results.json`.
- The B3 config above is the **frozen B3** used as the comparator everywhere.

### B0's harness locally

These are integration checks, not performance evidence.

1. Start the mock LLM:

   ```bash
   python3 kaggle/local/mock_llm_server.py --port 8765 &
   ```

2. Run the harness against it:

   ```bash
   build/b0src/ARC3-Inference/.venv/bin/python kaggle/local/run_harness_local.py --src build/b0src \
     --games ls20,vc33 --minutes 2 --base-url http://127.0.0.1:8765/v1 --model mock --out runs/<name> --env B2_BAR_FIX=1
   ```

3. To use another endpoint, change `--base-url` and `--model`:
   - **Local Ollama:** `--base-url http://localhost:11434/v1 --model qwen3.8:27b`.
   - **OpenRouter MiniMax** (remote, approved): see `HANDOFF_causal_identification.md` §2 for the exact command and its rate limits.

The local models available through Ollama are `qwen3.8:27b`, `qwen3.5:35b` and `gemma4:26b`.

### Procedural worlds

```bash
.venv/bin/python scripts/procgen_baseline.py --tasks 25 --out runs/procgen/<name>.json
```

### Kaggle

Do this only with the user's explicit permission; see §2. Follow `kaggle/RUNBOOK.md`. The sequence is:

1. `bash kaggle/kaggle_ready.sh` (read-only).
2. Build with `kaggle/b2/build_b2.py <user> <variant> KEY=VAL ...`.
3. Push, and check the commit log against the readiness gate in `docs/plan.md`.
4. Submit only on a "yes" from the user. If a submission is already pending, use `kaggle/submit_when_free.sh`.

---

## 7. Semantics that trip people up

- **Frames.** `obs.grid` is `frames[-1]`, the decision state; the earlier frames are animation. On a level-up, `frames[-2]` is the settled win frame of the old level and `frames[-1]` is the first frame of the new level. **Never diff across a level transition as if it were dynamics.** `arc3/hx/events.py` classifies each step as move, win, death or reset.
- **Action counting.**
  - `CompetitionEnv` counts an action only when it returns a frame. A RESET always counts.
  - The env refuses actions not in `available_actions`, and non-RESET actions after `GAME_OVER`.
- **`ONLY_RESET_LEVELS`.** This switch is process-global and must be set *after* the opening `make`. `CompetitionEnv` does that for you. The runner uses one process per game because of it.
- **Arcade speed.** Creating an Arcade costs about 0.7 s. One Arcade per process plus `CompetitionEnv(arcade, game_id)` costs about 13 ms per environment. See `make_env` in the Codex `reachability_ceiling.py`.
- **Cloning.** `copy.deepcopy` of the env wrapper is **unfaithful for tn36**, which diverges 3 steps after a copy. Verify clones against a fresh replay, or simply replay the action prefix in a new environment, which is cheap and exact.
- **Replaying B0.** Replaying each run's `history` from `benchmark.json` through a fresh `CompetitionEnv` (RESET maps to action 0) reproduces **all 417 recorded level-ups**, and **100 of 100 runs** end on B0's final level. This was measured on 21 October 2026. It checks level counts, not every frame.
  - `HANDOFF_causal_monitor.md` and `docs/prereg_rank_goal.md` say only 25 of 100 runs reproduce their wins. That figure came from an earlier replay path and does not hold for this one; the cause was not investigated.
  - The causal-identification work also found exact frame-level replay of local and B0 Kaggle event logs (`HANDOFF_causal_identification.md` §2).
- **Benchmark data.** `data/franzen_v3_output/benchmark.json` holds B0's 100 public runs: 25 games × 4 passes. Each has `history`, `actions_per_level`, `base_actions_per_level`, `final_score` and the pass number in `solver_analysis_html`. `arc3.scoring.game_score` reproduces every `final_score` exactly.
- **HUD bars.** Most games have a step or budget bar that changes on every action. Mask it before comparing frames (`perception.VolatileMask`); otherwise every state looks new.
- **Seeds and keys.** Use `zlib.crc32` for seeds, never `hash()`, which is randomised per process. Masked-frame keys are raw bytes, so dtype matters (uint8 vs int16).
- **Kaggle mounts.**
  - Input mount layouts differ between kernels: `/kaggle/input/datasets/<owner>/<slug>` vs the legacy `/kaggle/input/<slug>`. Only notebooks built with `build_b2.py` handle both.
  - `TAAF_MINIMAL_DIAGNOSTICS=1` skips `benchmark.json`; parse the `[finished] ... per-level=` log lines instead.
- **The shell is zsh.** It does not word-split `$var`, and a bare `=====` at the start of a command is an error. macOS has no `timeout` command.

---

## 8. Conventions for new work

1. **Find the prior result first.** Check §4 and `runs/ledger.jsonl`; the idea may already have a recorded verdict.
2. **Write the pre-registration** in `docs/prereg_<name>.md`. Include:
   - the question;
   - the frozen cases and seeds;
   - the arms and comparators (B3 frozen, B0, or a matched random or placebo arm);
   - a primary endpoint, usually levels completed or official score;
   - a game-cluster bootstrap with a fixed seed;
   - the decision rule;
   - harm checks;
   - the compute budget.
3. **Get the user's approval.**
4. **Freeze.** Record the sha256 of the code and config in `runs/<name>/FROZEN.json`, and verify the hashes again after the run.
5. **Smoke test** on one game, usually ls20, and disclose it in the report.
6. **Run, then report** in `docs/<name>_report.md`. Give the bottom line first and the numbers with intervals, list deviations, and keep negatives prominent.
7. **Append to the ledger** with `scripts/ledger_add.py`, even if the run lost.
8. **Change behaviour with new files or subclasses,** never by editing frozen modules. Keep every new B0 feature behind a switch that is OFF by default, where OFF reproduces B0 exactly.
9. **Use clustered statistics.** Pair by game and resample games, not runs. Report distinct states when checkpoints repeat across passes.

---

## 9. Related locations outside this repo

| Path | What it is |
|---|---|
| `/Users/ameya/Documents/Codex/2026-10-13/key/` | Codex-built world-model and memory-RL rescue stack. Read `HANDOFF.md` there. It holds the reachability ceiling (`outputs/rescue_reachability_ceiling.md`), the B3 stall-rescue test (`outputs/b3_stall_rescue.md`, pre-registered; the frontier-ordering line was stopped) and the stall-rescue pilot pre-registration. It imports this repo and uses its `.venv`, but never modifies it |
| `~/kaggle_work/arc-agi-3/` | The older `arccausal` engine (object perception, rule DSL, hypothesis bank). Vendored into `arc3/nsw/vendor/` |
| `~/.claude/projects/-Users-ameya-kaggle-arc/memory/` | Claude's persistent project memory: project state, research direction, compute-policy feedback |
| `~/Library/Application Support/Claude/side-session-notes/` | Hand-over notes between parallel Claude sessions |

Attribution and licences for upstream code and models are at the end of `README.md`. The model-licence risk for Qwen is discussed in `competition_contract.md`.
