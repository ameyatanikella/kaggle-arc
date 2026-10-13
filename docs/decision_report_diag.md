# DIAG (algebraic world-model diagnostics): report and adoption decision

Pre-registration: `docs/prereg_diag.md`, commit bb47368 (2026-10-12 23:00 UTC), with amendments 1–2 logged before the full run.
- Code: `arc3/diag/` (ledger, exact polynomials and implicitization, structure and cache, checkers B and C).
- Fixtures: `tests/test_diag_fixtures.py`.
- Shadow mode: `scripts/diag_shadow.py`; criteria: `scripts/diag_evaluate.py`.
- Data: `runs/diag/`.

## What the module encountered
**Stage 1: fixtures.** All 7 pass for C:
1. correct nonlinear model;
2. one fault;
3. 2–3 faults;
4. full rank with unresolved sign;
5. context change invalidating the cache;
6. incomplete family;
7. tracking swap, tagged and untagged.

B reached the same verdicts and diagnoses on every fixture. C added three things:
- the sign ambiguity with a resolving context (F4);
- regime/change-point reports (F6, F7);
- counted cache invalidation (F5).

**Stage 2: shadow mode** on B0's level-1 traces of the 25 public games (development set only).
- **Components proposed:** 21 games yielded supported components, from frames only: movers (linear over ℚ) in 15 games, toggles (linear over GF(2)) in 6.
- **Encounter (M1):** checkable relations on ≥ 30% of eligible steps in **13 games**, so the checks do meet real decisions (S1 pass).
- **Real contradictions on uninjected traces:** 9 games, including g50t's hidden state, the toggle model failing in s5i5, vc33, r11l and tn36, and the mover model failing in sc25, tr87, wa30 and bp35.
- **Agreement:** on all 81 real component checks, B and C gave the same verdict category.
- **No-observation games:** m0r0, re86, sk48 and tu93 produced none, because every observation was identity-uncertain (several identical sprites).

## What it changed
**Nothing behavioural.** It ran in shadow mode. It never selected, banned or interrupted an action and never ended a game. Lost plans, completion, action efficiency and SDK score were therefore **not measured**: stage 3 was gated on stage 2.

## Did it beat the ordinary checker? No, under the registered criteria.
Injected perception faults: 1,620 trials, 7 clean component instances, k = 1–3, 30 trials each.

| | B (ordinary, sampled RANSAC) | C (algebraic) |
|---|---|---|
| k = 1 exact / correctly ambiguous / confident-wrong | 0.635 / 0.098 / 0.000 | **0.743** / 0.098 / 0.000 |
| k = 2,3 exact + correctly ambiguous | 0.521 | **0.662** |
| k = 2,3 confident-wrong (S2: needed ≤ B − 0.10) | 0.154 | 0.156: **S2 fail** |
| k = 2,3 "hypothesis inadequate" (search failure) | 0.203 | 0.000 |
| False alarms on 18 clean components (S3) | 2 | 0: pass |
| Repaired next-observation accuracy (S4) | 0.950 | 0.952: pass |
| Per check, median over games (S5) | 294 ms | **1.4 ms** (85 ms without cache): pass |
| Cache memory, max per game | — | 0.15 MB |

**Reading.**
- C's higher exact rate does **not** come from dependency information that B lacks. It comes from **exhaustiveness**:
  - C enumerates every minimal diagnosis exactly through circuits.
  - Sampled RANSAC sometimes misses the clean consensus (B "inadequate" in 20% of multi-fault trials).
- **Exploratory, not pre-registered:** B+ is B with the standard spanning-subset degeneracy fix, run on the identical trials. It did **not** close the gap, and was confidently wrong in 11.1% of k=1 trials and 20.6% of k=2,3 trials, where a tie existed that its sampling never saw. C reported those ties.
- Confident-wrong at k = 2,3 is dominated by cases where a **smaller** fault set explains the data. No checker can resolve those without new observations; C lists the larger alternatives explicitly.
- **Reobservation suggestions:** 271. The suggested observation was actually faulty in 46% of cases, and one reobservation produced a correct result in 39%.
- **Caching:** about 60× speedup (C vs C-nocache), with 0.15 MB memory. Structure is shared across trials and across cells with the same design.

## Adoption decision
**Not adopted. Retain the ordinary checker (B) as specified (pre-registered S2 failed).** C stays as research code with no behavioural role.

What the data supports, as a hypothesis for a future pre-registered test rather than a claim: exact circuit-based diagnosis is a cheaper and more reliable *implementation* of consistency-based multi-fault diagnosis than sampled RANSAC. Its measured edge is that it does not report a sampling artefact as a confident diagnosis. Testing it would need:
- an ordinary baseline that also enumerates exhaustively (expected to tie C on outcomes at much higher cost);
- stage 3: an agent whose planner consumes explicit world-model components, with matched budgets and SDK scores.

## Limits
- Development games only (public and contaminated), level 1 only, B0's action traces. No unseen-game claim.
- Only linear components (ℚ, GF(2)) were proposed on real traces. The nonlinear and polynomial machinery is validated on fixtures only.
- Injected faults are synthetic: integer offsets and bit flips.
- B is one reasonable "strong ordinary" design; B+ is one more. Neither is exhaustive.
