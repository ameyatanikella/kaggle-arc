# Competition contract: ARC Prize 2026, ARC-AGI-3 (Kaggle)

Last verified: 2026-10-12, about 09:30 UTC. Sources were read on that date unless a row says otherwise.
Status codes:
- **V**: verified from an official source (link given).
- **O**: observed in a real artifact, such as a public notebook's logs or the installed package source.
- **P**: participant claim on the forum, not official.
- **U**: unknown or unavailable. No value has been invented for it.

## Deadlines (V: [Kaggle overview](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/overview))

| Event | UTC | America/Los_Angeles (PDT) |
|---|---|---|
| Entry deadline (accept rules) | 2026-10-26 23:59 | 2026-10-26 16:59 |
| Team merger deadline | 2026-10-26 23:59 | 2026-10-26 16:59 |
| Final submission deadline | 2026-11-02 23:59 | 2026-11-02 15:59 (PST, since DST ends Nov 1) |
| Winners announced | 2026-12-04 | |

On 2026-10-12 the overview page showed "22 days to go".
**Practical deadline.** RTX 6000 queues of 9 to 17 hours have been reported (P: [thread 745951](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/745951), [697944](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/697944)), and a scored run takes up to 9 h. So the last real submission must start **no later than about 2026-11-01**.

## Submission rules (V: [Rules](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/rules), [staff post 705405](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/705405))
- **1 submission per day** (Rules 2.2.a). A staff post says the 5/day window in May–June was a mistake.
- **Up to 2 final submissions** may be selected (Rules 2.2.b). If none are selected, Kaggle selects automatically (Rules 3.18.c).
- Maximum team size is 8. Mergers require that the combined submission count is at most 1/day × days elapsed.
- Code competition: you must submit a notebook. Both the CPU and GPU notebook limits are **9 h run time**, **internet is disabled**, and "freely & publicly available external data is allowed, including pre-trained models". The submission file is generated automatically (V: overview, "Code Requirements"; the limit was raised from 6 h to 9 h, per [staff post 697944](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/697944)).
- "As long as the agent takes action on any of the games, a submission file for all of the games is created" (V: overview).
- The private test set is not rerun at the end. "Your final score is already computed but will stay hidden" (P: CPMP in 697944). That post also says about 55 games are public and 55 private (it cites "100 games", which looks like a typo). Franzen's code assumes 110 (O).

## Licensing and open-source obligations (V: Rules 1.6, 1.7, 2.5, 2.6, 3.6)
- Winner license: **CC-BY 4.0**. Data access and use: Apache 2.0.
- "Submissions are required to have open source system, open source model, and open source weights/parameters" (OSI Open Source AI definition). **Closed API models cannot be used at run time.**
- Any open-source code used must carry an OSI-approved license that does not limit commercial use (Rules 3.6.c).
- Prize eligibility requires open-sourcing the solution. Winners must deliver code, documentation and a reproducible description (Rules 2.8).
- No private sharing outside the team. Public sharing must happen on Kaggle forums or notebooks (Rules 3.5.d, 3.6.a/b).
- Using game-test-data labels or human predictions is forbidden (Rules 3.4.b). Game source code is obfuscated on the hidden set (P: 705405 comments).

## Hardware and runtime
| Item | Value | Status |
|---|---|---|
| Accelerator for ARC-AGI-3 | RTX 6000 Pro, machine type `g4-standard-48`. The RTX may only be used for this competition | V (overview "Upgraded accelerators") |
| GPU | 1× NVIDIA RTX PRO 6000 Blackwell Server Edition, sm_120, 95.6 GB | O (Franzen notebook logs) |
| Host RAM | 176.9 GB | O (Franzen logs) |
| `/kaggle/working` disk | 19.5 GB | O (Franzen logs) |
| Python in image | 3.12.13 | O (notebook metadata) |
| Kaggle weekly GPU quota for RTX 6000 | U. Must be read from the user's account page | U |
| Time to start a model server | about 530–615 s for Qwen3.8-Flash-Next | P (thread 744792) |

## Evaluation interface (O: installed `arc-agi==0.9.9`, `arcengine==0.9.3`; V: [competition-mode docs](https://docs.arcprize.org/toolkit/competition_mode))
- Observation: `FrameDataRaw` with a list of 64×64 int frames (one action may return 1..N animation frames), `state` (NOT_PLAYED / NOT_FINISHED / WIN / GAME_OVER), `levels_completed`, `win_levels`, `available_actions` (subset of 1..7), `guid`.
- Actions: RESET (0); ACTION1..5 and ACTION7 (simple); ACTION6 (x,y in 0..63).
- Competition mode:
  - only **level** resets ("game resets are converted into level resets");
  - **one `make` per environment** (a second make returns an error);
  - **one scorecard** (a second open returns 409);
  - `get_scorecard` is unavailable in flight;
  - scoring covers every environment, including unplayed ones.

  The Kaggle rerun forces competition mode.
- Rerun detection: env `KAGGLE_IS_COMPETITION_RERUN`. Gateway at `http://gateway:8001/`, API key `test-key-123`. The arc-agi wheels are at `/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels`. The competition data also contains `ARC-AGI-3-Agents` and the public `environment_files` (O: official starter `build_notebook.py`, Franzen notebook).
- A commit (Save & Run All) is not scored. It must still leave a `submission.parquet` in `/kaggle/working`; the starter writes a dummy one. Only the competition rerun plays hidden games.
- **Reset accounting (O: `arcengine/base_game.py`, `arc_agi/scorecard.py`, `arc_agi/api.py`):**
  - A level RESET **adds +1** to the action counter (`inc_reset_count` increments `actions`). It never refunds actions.
  - In competition mode, a RESET when the level's `_action_count == 0` is not stepped, but the scorecard is still updated. It is a no-op that still costs an action, so never send it.
  - After GAME_OVER, non-RESET actions return an empty frame and are not counted. A RESET is required.
- **Idle scorecards:** the `arc_agi` server auto-closes a scorecard idle for `STALE_MINUTES` (default 15) when a close hook is registered (O). Whether the Kaggle gateway does this is U. Policy: never let the whole run go 15 min without an action. Franzen issues a warm-up RESET before the model server is ready (O).

## Scoring (O: `arc_agi/scorecard.py` 0.9.9; V: [methodology](https://docs.arcprize.org/methodology))
- Completed level: `s_l = min(115, 100·(b_l/a_l)^2)`. Here `b_l` is the human baseline (upper-median first-time human) and `a_l` is the agent's actions on that level, **including level resets**. An uncompleted level scores 0.
- Game: `S = Σ l·s_l / Σ l` over all levels l = 1..N, then capped at `100·Σ_{l: s_l>0} l / Σ l`.
- If a game has several plays, the game score is the maximum over plays. In competition mode there is only one play.
- The total score is the mean over all environments. Unplayed environments score 0.
- Human baselines for hidden games are not visible during play. A public game's baseline is in its `metadata.json`.

## Measured public reference points (2026-10-12)
| Reference | Score | Status |
|---|---|---|
| Public LB #1 (Yi-Chia Chen) / #2 (Tufa Labs) / #5 | 67.49 / 56.52 / 42.55 | V (leaderboard) |
| Franzen M2 notebook (Apache-2.0), public LB | 33.68 (v3), best 34.30 (v1) | V (notebook page) |
| Unedited Franzen copies, single draws | mean 25.77, SD 3.93, median 26.19 | P ([thread 745062](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/745062)) |
| Franzen v3 on the 25 public games, 4 passes, competition-scaled budget | mean 46.49 per run (33 of 100 runs won) | O (notebook output) |

**Implication.** A single LB draw has an SD of about 4 points. One submission per day cannot separate changes smaller than about 5 points. Local multi-pass evaluation on public games is needed for anything finer, and that needs RTX-class GPU hours.

## Open items (U) needed from the user
1. Kaggle account access: rules accepted, team status, and an API token at `.kaggle/access_token` (the user places it).
2. Kaggle RTX 6000 weekly quota, and whether submission reruns count against it.
3. Any paid GPU budget, for example a rented RTX PRO 6000 at about $2/h.
