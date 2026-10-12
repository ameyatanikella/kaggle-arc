# Scheduler replay and token elasticity (B0 public-game data)

Data: Franzen v3 `benchmark.json` (100 runs = 25 public games × 4 passes, B0 configuration; Kaggle output of the public notebook, downloaded 2026-10-12). Tool: `scripts/sched_replay.py`, 200 simulations per cell.

**Model assumptions.**
- Progress depends only on a game's generated tokens.
- Progress is Markov at level boundaries.
- Right-censoring is respected: an attempt that ran d tokens unfinished only means "not cleared within d".
- Beyond every logged attempt the model is pessimistic: no progress.

**Validation.** At the observed mean budget of 175K tokens/game, the B0 priority policy replays to **46.20**, against the actual run's **46.49**.

| Budget per game | Equal split | Franzen priority (B0) | Cap 60K | Level-first |
|---|---|---|---|---|
| 100K | 24.2 | 32.1 | 31.1 | 30.9 |
| 140K | 33.3 | 38.8 | 38.7 | 39.6 |
| **175K (observed)** | 41.0 | **46.2** | 46.2 | 45.2 |
| 210K (+20%) | 44.3 | 48.5 | 48.5 | 48.7 |
| 260K (+50%) | 46.6 | 48.9 | 48.6 | 48.7 |

Simulation SD of the mean is about 3.5–4.0.

**Findings**
1. B0's scheduler adds about +5 points over equal allocation. The simple alternatives tried do not beat it, so scheduler re-tuning is low priority.
2. B0 operates near a knee. −20% tokens costs about 7 points, while +20% gains about 2.3 and +50% gains about 2.7. This is pessimistic because of censoring, so the true gain is likely somewhat larger. **Throughput work is worth roughly 2–3 points.** That is less than often assumed, but it is the only lever with a model-free estimate.
3. Saturation comes from games stuck in every pass (bp35, lf52, sk48, wa30 and others). Matching each game's best pass would score 61.5. The large remaining headroom is in **breaking stuck levels** (reasoning quality per token), not in allocation.

**Implication for B2.** Prefer changes that alter how a stuck agent spends tokens, for example probing on token-heavy, action-light levels and keeping evidence across trims. Inference speed-ups are worth pursuing only if cheap and low-risk.
