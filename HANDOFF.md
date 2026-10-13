# Handoff: ARC Prize 2026 (ARC-AGI-3) Kaggle entry

State as of 2026-10-13 11:00 UTC. Repo: `/Users/ameya/kaggle-arc` (git; latest commit 40ad0f4; 64/64 tests pass with `.venv/bin/python -m pytest -q tests`).

## 1. Deadlines and rules that bite
- **Final submission:** 2026-11-02 23:59 UTC. **Entry / team merge:** 2026-10-26 23:59 UTC. Submit by about Nov 1 (reruns take about 9 h).
- **Submissions:**
  - 1 per day and **only 1 pending per team** (otherwise `FAILED_PRECONDITION`).
  - 2 final selections; the final score is the better of their private scores. Private scores are computed at submission time.
- **Hardware and quota:**
  - Kaggle GPU: RTX PRO 6000, 9 h per notebook.
  - Weekly quota 45 h, **28.6 h left**, refreshes 2026-10-17 00:00 UTC.
  - **Competition reruns do not consume quota;** commits and paired runs do.
- **Integrity:** keep notebooks private unless authorized. Never hand-edit results. The Kaggle token is in `.kaggle/access_token` (gitignored); never print or commit it. All 25 public games are contaminated development data; only the leaderboard measures generalization.

## 2. Submissions so far
| Ref | System | Public score |
|---|---|---|
| 57112928 | **B0**: faithful reproduction of Franzen's M2 notebook (v3; scored path identical to v1) | **26.92** |
| 57126140 | B2-algebra v2: B0 + bar/prompt fixes + solved-level memory + algebra sandbox tools | 24.20 (no detectable effect) |

- Unedited copies of Franzen's notebook score 25.8 ± 3.9 per draw.
- The 33.68 / 34.30 shown on Franzen's page are the **best of many submissions**, not single runs.
- The leaderboard tops are 67.5 (#1) and 56.5 (Tufa Labs, who built the harness).

## 3. Key measured facts (read before planning)
- **Where B0 loses points** (public games): 50.9 points come from unfinished levels and only 2.6 from inefficiency. Completing levels is everything.
- **Variance is the hidden headroom.** B0's best pass per game would score 61.5, against 46.5 for a mean pass: many failures are unlucky stalls.
- **Compute:** "more tokens = more levels". +20% tokens ≈ +2.3 points. The priority scheduler is +5 over an equal split.
- **Measurement noise:**
  - a single leaderboard draw has an SD of about 4;
  - **the same configuration run twice in paired Kaggle sessions differed by +7.3 points;**
  - the second arm in a session lost about 12% throughput.

  Paired runs cannot resolve effects under about 8 points. Always counterbalance arm order (`build_paired.py --order ba`).
- **Paired results:** B2c (bar fix + prompt fixes + solved memory) vs B0 was inconclusive. The stuck nudge was not adopted.

## 4. Candidates and components (status)
**B0** (`kaggle/b0/`)
- Franzen M2: Duck harness + TAAF + Qwen3.8-Flash-Next W4A16 on SGLang.
- The submit-ready baseline. Every Kaggle push must go through `kaggle/b2/build_b2.py` (see the gotchas).

**B2 switches** (`kaggle/b2/b2_patches.py`)
- `B2_BAR_FIX`, `B2_PROMPT_FIXES`, `B2_SOLVED_MEMORY`, `B2_TOOLS` (algebra tools, `sandbox_tools.py`), `B2_STUCK_NUDGE`.
- None has shown a measurable gain.

**Algebra / math work** (`arc3/algebra/`)
- Verified tools.
- Real-game study: `docs/algebra_on_games.md`.
- Algebraic diagnostics (DIAG) were not adopted: `docs/decision_report_diag.md`.
- The exact noncommutative signature (`ncsig.py`) is an offline utility only.

**B3** (`arc3/agents/b3/agent.py`): CPU model-based explorer, frozen.
- Config `{"plan_cache":"projected","noop_order":true,"reset_purpose":true}`.
- 65 levels on 25 games × 5 seeds; about 0.3 SDK score. Not competitive alone.
- Rejected ablations are preserved in `arc3/research/b3_ablation`.
- See `docs/b3_report.md` and `docs/qualification_b3.md`.

**B3 + Franzen integration** (`kaggle/local/b3_franzen.py`)
- Franzen's unchanged tool-agent loop; B3 executes and attributes every action and is exposed via a `b3(...)` sandbox channel.
- Arms: `--arm b3 | b3_franzen | franzen`.
- **Verified, but rejected with local Qwen3.5-35B:** 8 vs 10 levels; no level was completed by the reasoning layer; LLM time displaced exploration.
- **Never tested with Franzen's real model on Kaggle.** A proposal card exists for a Franzen-first variant (task_96d54b3d).
- See `docs/b3_franzen_report.md`.

## 5. How to run things
- **Local evaluation (CPU agents):**
  - `.venv/bin/python -m arc3.runner --agent b3 --games all --seeds 0,1,2,3,4 --max-actions 3000 --game-seconds 600 --config '{...}' --out runs/...`
  - Official scorecard scoring.
- **Local harness with an LLM:**
  - `build/b0src/ARC3-Inference/.venv/bin/python kaggle/local/b3_franzen.py --arm franzen|b3_franzen|b3 --games ... --minutes 20 --base-url http://localhost:11434/v1 --model qwen3.5:35b --out ...`
  - Ollama context is 262K and prefix caching works.
  - Run **one game at a time**: concurrent games evict each other's KV cache.
- **Build a Kaggle notebook:** `.venv/bin/python kaggle/b2/build_b2.py axizcc <variant> --commit=minimal KEY=VAL ...`
  - Then simulate the rerun install cell locally before pushing (pattern in `docs/algebra_verification.md` §3).
- **Paired A/B on Kaggle:** `kaggle/eval/build_paired.py` (about 3.85 h per arm pass; check quota first).
- **Submit:**
  - `bash kaggle/submit_algebra.sh <version>` (push, validate the commit log, submit);
  - or `bash kaggle/submit_when_free.sh <kernel> <version>`, which waits out the 1-pending limit.
- **Analysis scripts:** `scripts/paired_compare.py` (game-level bootstrap), `scripts/b3_qual_eval.py`, `scripts/b3f_evaluate.py`, `scripts/b3_failure_diag.py`.
- **Ledger:** `runs/ledger.jsonl` (append every run, including losing ones).

## 6. Gotchas that already cost time
- **Input mount layout varies between Kaggle kernels:** `/kaggle/input/datasets/<owner>/<slug>` vs legacy `/kaggle/input/<slug>`. `build_b2.add_input_resolver` handles it. A plain `build_b0.py` notebook will fail on legacy-layout kernels.
- **`TAAF_MINIMAL_DIAGNOSTICS=1`** (the competition setting) skips `benchmark.json`. Parse the `[finished] ... per-level=` log lines instead.
- **Harness performance:** it re-serializes the whole history after every action, which is O(n²). It is deferred during B3 autonomy in `b3_franzen.py`.
- **Ollama** rejected 2 of the harness's requests with `content: null` (rare).
- **Keys and seeds:**
  - masked-frame keys are raw bytes, so dtype matters (uint8 vs int16);
  - use `zlib.crc32`, not `hash()`, for seeds (hash randomization).
- **Shell:** zsh does not word-split `$var`; use explicit commands rather than `set -- $spec` loops.
- **Watchers:** they must act only on explicit Kaggle states. Network errors made early watchers exit falsely.

## 7. Recommended next steps (ranked; see the last discussion)
1. **Read the leaders' writeups** (Franzen: kaggle.com/writeups/dfranzen/arc-agi-3-milestone-2 and github.com/da-fr/arc-agi-3-solution; others if public). Cheap and high-information.
2. **Throughput:** test the faster model variant (NVFP4 with built-in MTP, about 1,135 vs 946 tok/s) and stream/context tuning. Expect +2–5. Low risk; one paired run.
3. **Stall re-rolling:** when a level stalls (tokens spent, no progress), restart the reasoning with a fresh context, a short learned-facts summary and a different temperature or seed. Ceiling suggested by best-pass variance: +5–15.
4. Smarter time allocation across games (+2–5). A mechanics catalogue and goal-hypothesis helpers (uncertain).
5. **Optional:** the Franzen-first B3 tools test on Kaggle (proposal card task_96d54b3d). Guess: about B0 ± 3.

**Budget reality:** about 3 paired tests per week of quota. Plan each test to be decisive (≥ 8-point effects) and replicated.

## 8. Other resources
- **SDSC student cluster** (`ssh atanikella@login.scc.sdsc.edu`): all GPU nodes were down. Limit 100 GB; work only in `~/arc3`, via Slurm; disturb nothing.
- The user also has Colab Pro.
- **Local model:** Ollama `qwen3.5:35b` (about 38 tok/s decode; too weak and slow to evaluate reasoning changes meaningfully).
- **Key docs:** `competition_contract.md`, `math_contract.md`, `docs/plan.md`, `docs/decision_report_0{1,2,3}.md` and the reports listed above.
