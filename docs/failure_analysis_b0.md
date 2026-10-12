# B0 failure analysis: real transcripts of stuck games (Franzen v3, pass 0)

Source: Kaggle output of the public Franzen notebook (transcripts for sk48, bp35, lf52 and wa30, pass 0), downloaded 2026-10-12.

Categories, per the brief: perception, model class, experiment choice, goal inference, planning, resource management. "First consequential error" means the earliest decision after which the level was not cleared.

| Game / level | What the model did | First consequential error | Category |
|---|---|---|---|
| sk48 L1 (0/4 passes clear it) | Probed every action. Found the rod extend/retract, grab and push-cascade mechanics. Wrote a simulator, then BFS, then verified execution. Reached the exact "legend" layout. No level-up. | Assumed "match the legend picture" was the win condition. A buggy replan loop in its own executor then burned 58 actions | **Goal inference** (primary). Planning/code bug (secondary, cost about 0.9× the human baseline in actions) |
| bp35 L2 | Clicked blocks to let a bubble rise in a vertically **scrolling** world. Tried to build a world map in world coordinates | Lost track of world vs screen coordinates. Its search found "no frontier" because of a perception/map error | **Perception** (camera scroll), then model class |
| lf52 L2 | Peg-solitaire panels plus a token moving along black "wires" between panels. Discovered that the arrow keys move the token about 25 turns in | The win rule ("one peg total") was only inferred late, and the token's role stayed unclear | **Model class / mechanic discovery**, then goal inference |
| wa30 L2 | Sokoban-like box carrying under a budget. Correctly computed "93 actions needed but 61 remaining", so concluded a cheaper mechanic must exist (good reasoning). Then found an autonomous orange agent that chases boxes | Did not find the efficient mechanic within budget. Code bugs while modelling the chaser | **Model class** (multi-agent dynamics) and resource management |

## Quantitative context (`docs/analysis_scheduler_elasticity.md`)
- Failed final levels used a median of 99K tokens and 70 actions, against 20K tokens and 28 actions on completed levels.
- Tokens per action were about 1,384 on failed levels vs 695 on completed ones.
- The stuck levels above received only 25–36 turns, because the scheduler (correctly) starves non-progressing games. Their wall-clock span was about 5 h, mostly queue time.

## Implications
1. The dominant failures are **goal inference and mechanic discovery on genuinely complex levels**. They are not a simple lack of experimentation, so `B2_STUCK_NUDGE` is at best a partial remedy. Its result should be read with this in mind.
2. Model-written executors with replanning loops can burn many actions on a simulator mismatch (sk48: 58 actions). A harness guard that stops a snippet after K consecutive prediction mismatches would be a **correction-type** change. B0's existing guards cover no-ops and repeats, not divergence from the model's own predictions, and the harness cannot see those predictions.
3. More tokens per stuck game would not obviously help: the scheduler already deprioritizes these games, and the replay elasticity is about +2.3 points for +20% tokens.
4. Gains of the size the leaders show (+20–40 points) almost certainly come from model capability or fine-tuning, not from harness text. That is outside our compute budget. Within budget, the realistic aim is B0 + small, safe corrections, plus the best-of-2 final selection.
