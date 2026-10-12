# Decision report 01 (2026-10-12, about 10:00 UTC)

## What changed
- Verified the contract (`competition_contract.md`):
  - 1 submission per day, 2 final selections, 9 h runtime, no internet.
  - Model and weights must be open source. Winners license under CC-BY-4.0.
  - The final deadline is Nov 2 23:59 UTC. RTX queues of 9–17 h make about Nov 1 the practical cutoff.
- Built a local competition-semantics evaluator on the official `arc_agi 0.9.9` (`arc3/`). It pins:
  - level reset = +1 action;
  - GAME_OVER handling;
  - one `make` per game;
  - scoring replica = official calculator (300 random cases).
- Ran B1 (CPU state-graph explorer) end to end on the 25 public games.
- Reconstructed B0's exact harness source: upstream duck-harness 7652836 plus Franzen's notebook patch.
- B0 now runs locally against a scripted mock LLM, with prompts, sandbox, actions and scoring all working.
- Built the B0 notebook (`kaggle/b0/build_b0.py`). It is verbatim on the rerun path; the commit run is a 20-min smoke test.
- Built the B2 patch layer (`kaggle/b2/b2_patches.py`). All switches OFF equals B0. Two switches exist and have been integration-tested:
  - `B2_BAR_FIX`: deletes a misleading prompt line;
  - `B2_SOLVED_MEMORY`: a port of rellik13's M85.
- Algebra and experiment-design library with 14 math fixtures, documented in `math_contract.md`.

## What actually ran
| Run | Split | Official mean | Levels | Errors | Wall time |
|---|---|---|---|---|---|
| B1 v0 (per-pixel HUD mask) | public-25 | 0.026 | 8 | 0 | 5.7 s |
| B1 v1 (edge-line HUD mask) | public-25 | 0.115 | 9 | 0 | 6.6 s |
| B0 harness + mock LLM (integration only, no reasoning) | ls20, vc33 | n/a | 1 | 0 | 93 s |
| B0+B2a harness + mock LLM (integration only) | vc33 | n/a | 2 | 0 | 91 s |

B0 has not been run with its real model by us. The only B0 numbers come from Franzen's own logged run (4 passes, public-25) and from public LB evidence (copies: mean 25.8, SD 3.9).

## What improved or regressed
- B1 rose from 0.026 to 0.115 after a unit test exposed a real HUD-mask bug: step bars change a different pixel each step, so per-pixel hit counts never masked them. B1 stays negligible as a competitor.

## Established
- **Bottleneck (measured on B0's own 100-run log).** 50.9 points are lost to levels never completed and 2.6 points to inefficiency. Completed levels run at a median of 1.55× fewer actions than humans.
- **Exchange rate.** Raising every level's actions by 25% costs 3.2 points. One more level on an unfinished run is worth +13.5 on that run.
- **Consequence.** Action-minimizing intervention design (min-cost bases) does not target the binding constraint. The scarce resource is tokens to a correct world model.
- **No local proxy exists.** Flash-Next has 180B parameters, and the smallest pruned builds are 71–74 GB, which exceeds this Mac's 64 GB. Meaningful A/B runs need RTX-class GPU time.

## Uncertain (not established)
- Whether `B2_BAR_FIX` or `B2_SOLVED_MEMORY` raise B0's score. rellik13's single runs suggest yes in a different harness. Unmeasured here.
- Whether any algebra tool will be used by the model or will change outcomes.

## Single next action supported by the evidence
**Get B0 onto the leaderboard.** That needs Kaggle API access (user action), and is the highest-value immediate step. The next step after that needs GPU hours for paired B0-vs-B2a runs on the public games: Kaggle quota or a rented RTX PRO 6000, both user decisions.
