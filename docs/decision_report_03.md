# Decision report 03: paired runs pair-b2c and pair-nudge (2026-10-13)

Data:
- `runs/kaggle/pair_b2c` (arm A = B0, arm B = B2c);
- `runs/kaggle/pair_nudge` (arm A = B2c, arm B = B2c + stuck nudge).

Each arm ran 2 passes × 25 public games, about 3.85 h, 10 streams, on one RTX PRO 6000. Within a session the arms run sequentially, A then B. Scores are descriptive: these are contaminated public games.

## Results
| Contrast | Mean A | Mean B | Paired Δ (B − A) | 90% bootstrap CI (games) |
|---|---|---|---|---|
| B0 → B2c (pair-b2c) | 45.16 | 39.56 | −5.60 | [−15.44, +3.18] |
| B2c → B2c + nudge (pair-nudge) | 46.86 | 43.76 | −3.10 | [−9.42, +3.08] |
| **Same config twice:** B2c (2nd, pair-b2c) → B2c (1st, pair-nudge) | 39.56 | 46.86 | **+7.30** | [−1.74, +16.83] |
| Cross-session: B0 (1st) → B2c (1st) | 45.16 | 46.86 | +1.71 | [−4.83, +8.76] |

## Diagnosis of the pair-b2c drop (done before deciding)
**Three games collapsed to 0 levels in both B2c passes:** cd82 (B0: 6 levels, won), sc25 (5) and g50t (2).
- **No crash.** `b2_summary.errors` = 0. Request timeouts were equal across arms (9 each). The only timeout in cd82 was the final pending request at the arm's deadline.
- **Not reproducible.** In pair-nudge, where B2c ran first, the same three games scored 72.2, 79.9 and 28.5.
- **The scheduler amplifies early luck.** A game stuck on level 1 is deprioritized (B2c cd82: 81 requests vs 168 for B0).
- **Order effect.** Both second-position arms scored lower (39.56 and 43.76) than both first-position arms (45.16 and 46.86). The second arm of pair-b2c generated 12% fewer completion tokens in the same wall time (7.57 M vs 8.64 M).

## Decisions (pre-registered rules applied as written)
- **Rule 4 (gross regression) is not triggered** for B2c: the CI upper bound is +3.18 > 0. No B2c defect was found, so the algebra submission (B0 + B2c + TOOLS, `axizcc/arc3-b2-algebra` v2) proceeds.
- **The stuck nudge is not adopted.** There is no evidence of benefit (−3.10, CI spans 0), and it ran in the disadvantaged second position.
- **Evaluation lesson:**
  - The same-configuration replicate differs by +7.3 points.
  - Single-session sequential paired runs at this budget cannot resolve effects smaller than about 8 points.
  - Order must be counterbalanced (`build_paired.py --order ba` exists), and differences under about 8 points must not be interpreted.
  - Future ablations should be token-matched (`max_generated_tokens_per_game`) and replicated across sessions.
