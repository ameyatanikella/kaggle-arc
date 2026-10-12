# ARC Prize 2026: ARC-AGI-3 (Kaggle)

Working repository for our competition entry. The authoritative documents are:
- `competition_contract.md`: verified rules, deadlines, interface and scoring.
- `math_contract.md`: every principle used, with its assumptions and a test.
- `docs/plan.md`: candidates, B2 priorities and the evaluation protocol.
- `docs/decision_report_*.md`: what ran, what is established, and the next action.
- `runs/ledger.jsonl`: all local runs, including losing ones.

## Candidates
| ID | Description | Where |
|---|---|---|
| **B0** | Daniel Franzen's Milestone-2 solution (Apache-2.0): Tufa Labs' Duck harness + Qwen3.8-Flash-Next (W4A16) served with SGLang on one RTX PRO 6000. Our only change is a 20-minute smoke test on the non-scored commit run | `kaggle/b0/` |
| **B1** | CPU-only state-graph explorer: HUD masking, object-centred clicks, fatal-edge avoidance, costed level resets. Validates the pipeline and serves as a fallback. Not competitive (0.115 on the public 25) | `arc3/agents/explorer.py` |
| **B2** | B0 plus switchable patches, all OFF by default (OFF = B0). Switches: `B2_BAR_FIX`, `B2_PROMPT_FIXES` (corrections only), `B2_SOLVED_MEMORY` (port of rellik13's M85, Apache-2.0), `B2_TOOLS` (compute-only sandbox helpers: `bfs_plan`, `distinguish`, `gf2_solve`, `zm_solve`) | `kaggle/b2/` |

## Reproduce locally (no GPU)
```bash
python3.12 -m venv .venv && .venv/bin/pip install arc-agi==0.9.9 arcengine==0.9.3 numpy scipy pandas pyarrow pytest kaggle
.venv/bin/python scripts/download_public_games.py data/environment_files   # 25 public games via the official client
.venv/bin/python -m pytest -q tests                                          # 22 tests: math fixtures, engine semantics, scoring replica, tools
.venv/bin/python -m arc3.runner --agent explorer --games all --workers 12 --max-actions 3000 --game-seconds 600 --out runs/b1
# B0 harness against a scripted mock LLM (integration only):
#   rebuild build/b0src = duck-harness@7652836 + kaggle/b0/harness-changes.patch, `uv sync --locked` in ARC3-Inference, then
python3 kaggle/local/mock_llm_server.py --port 8765 &
build/b0src/ARC3-Inference/.venv/bin/python kaggle/local/run_harness_local.py --games ls20,vc33 --minutes 2 --out runs/mock --env B2_BAR_FIX=1
```
Kaggle steps (push, paired A/B, submit) are in `kaggle/RUNBOOK.md`.

## Key measured facts (2026-10-12)
- B0's own public-game log (100 runs) loses **50.9 points to unfinished levels** and **2.6 points to action inefficiency**. Completed levels already use about 1.55× fewer actions than humans.
- Raising every completed level's actions by 25% would cost 3.2 points. One more cleared level on an unfinished run is worth about 13.5 points on that run.
- Single leaderboard draws of the same notebook vary with an SD of about 4 points (forum analysis of 1,200+ copies).

## Attribution
- Duck harness and TAAF: Tufa Labs (Jeroen Cottaar). Upstream `github.com/Tufalabs/duck-harness`; no repo license file; publicly shared in the competition.
- Milestone-2 harness patch, serving setup and notebook: Daniel Franzen (Apache-2.0, `github.com/da-fr/arc-agi-3-solution`).
- Solved-level memory idea and code: rellik13 / sirikilohit (Apache-2.0, `github.com/LohitSiriki/arc-agi-3-milestone2-solution`).
- Qwen3.8-Flash-Next: Qwen Community License 1.0 (see the license risk note in `competition_contract.md`). Quantizations by Intel (AutoRound) and Albucino (draft).
- SGLang Pennyroyal fork: John Pezzulli, with patches by Mamy Ratsimbazafy and Gabriel Olympie.
- `arc-agi` / `arcengine`: ARC Prize Foundation (MIT).

## Limitations
- No B0 or B2 run with the real model has been done by us yet. Everything model-dependent is blocked on Kaggle access or RTX PRO 6000 hours.
- The public 25 games are contaminated: inspected by us and used by the upstream authors. They do not form a holdout. Only the hidden leaderboard measures generalization.
