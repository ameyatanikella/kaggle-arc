# Math contract

For each principle the agent uses, this lists the assumptions it needs, the exact code that uses it, and the test or counterexample that pins it. Tests are in `tests/test_algebra_fixtures.py` and check correctness boundaries only. They are **not** evidence of ARC performance.

| # | Principle | Assumptions required | Implementation | Test / counterexample |
|---|---|---|---|---|
| 1 | Unique multilinear representation on {0,1}^n; Möbius inversion θ_S = Σ_{T⊆S}(−1)^{\|S\|−\|T\|} f(1_T) | Inputs are exactly Boolean. Real coefficients. Inversion needs the **complete** truth table | `algebra/boolean.py: mobius_coefficients`, `feature_row` | `test_mobius_exact_and_additivity_circuit` |
| 2 | Exact identities AND=xy, OR=x+y−xy, XOR=x+y−2xy, toggle=b+u−2bu | Boolean inputs only | used in `sparse_fit` checks | `test_or_xor_rank_and_probe_choice` |
| 3 | Identifiability of a coefficient = its unit vector lies in the row space of the observed design rows | Linear-in-parameters model. Exact rational arithmetic, no float thresholds | `boolean.interaction_estimable`, `linalg.IncrementalRank` | OR/XOR: rows (1,0),(0,1) have rank 2; adding (1,1) gives rank 3. θ12 is unidentified until (1,1) is observed |
| 4 | Low-degree sparse fit (MDL-style: smallest exact support first) | Sparsity is an **assumption**. Several surviving supports means "unidentified", not "pick one" | `boolean.sparse_fit` | Two observations leave the interaction unidentified |
| 5 | Algebraic circuits of the row matroid give consistency checks w^T y = 0 | Linear-in-parameters. A violation means an interaction, a wrong guard, a parse error or a hidden mode; passing is not proof of causality | `linalg.left_dependencies`, `boolean.additivity_check` | d11−d10−d01+d00=0 is the unique circuit of the additive 2-switch design. XOR violates it (−2) |
| 6 | GF(2) elimination gives feasibility and *a* solution | Genuine XOR/toggle dynamics. Commutation and repeat-activation must be validated first | `modular.gf2_solve_all` | `test_gf2_multiple_solutions_min_cost` |
| 7 | Minimum-cost GF(2) solution = minimum over particular + nullspace | Exact only when nullity ≤ 20 (enumeration). Beyond that the result is labelled `exact: False` | `modular.gf2_min_cost` | {b3} vs {b1,b2}: the cheapest choice flips when b3's travel cost is 5 |
| 8 | Z_m systems with non-prime m | No field division. Exact enumeration with a bounded search space | `modular.zm_solve_min` | 2u ≡ 1 (mod 4) is infeasible. 2 has no inverse mod 4 |
| 9 | Rado–Edmonds: cost-sorted greedy gives a minimum-cost basis | Genuine matroid (linear: rows of a fixed matrix), fixed additive element costs, exact independence oracle. **Not valid** for grouped observations, state-dependent travel, prerequisites or changing hypotheses | `design.greedy_min_cost_basis` | `test_greedy_basis_matches_bruteforce` (60 random linear matroids vs brute force) |
| 10 | For deterministic hypotheses, the entropy of the outcome partition = expected information gain | Hypotheses are deterministic. If the weights are heuristic, so is the number | `design.partition_entropy` | `test_partition_entropy_is_expected_info_gain` |
| 11 | Full Jacobian rank ⇏ global identifiability | Rank is local. Isolated branches survive | `design.select_rank_only` vs `select_combined` | y0=θ²: rank 1, branches ±2. y1=(θ+1)² adds **zero** rank but separates the branches. Rank-only stops, the combined selector picks y1 (`test_sign_ambiguity_...`) |
| 12 | Decision relevance: only distinctions that change the plan are worth paying for | Minimax regret is exact only if the true model is among those considered and the costs are exact | `design.minimax_regret`, `should_act`; `select_combined` scores 0 when all hypotheses recommend the same action | `test_value_equivalent_models_need_no_experiment`. Value-equivalent models give regret 0, so act without testing |
| 13 | Shortest distinguishing sequence by BFS over the product state space | Deterministic models with **known** current states. "None after exhaustion" proves non-separability only within the modelled finite space. Hitting the bound proves nothing. Not exact under partial observability | `distinguish.shortest_distinguishing_sequence` | Wall at 3 vs 4: RRRR. Identical models: exhausted, None. Depth 2: bounded, proves nothing |
| 14 | Same (visible obs, action) → different next obs refutes every visible-Markov model | Deterministic engine (confirmed for arcengine at the action level) | `distinguish.markov_conflicts` | `test_hidden_timer_detected_and_explained` |
| 15 | Guards before degree: saturation and collision are piecewise, not polynomial | — | `distinguish.fit_guarded_affine` | min(x,5): no polynomial of degree ≤3 fits exactly. The guarded fit is exact, and the guard position is only identified up to {5,6} |
| 16 | Empirical commutation is local | A few equal results ≠ global commutation. Transformations need not be invertible | `distinguish.commutes_on` | Commutes in mode 0, fails in mode 1 |

## Not claimed
- No compressed-sensing recovery guarantee is invoked. Game-generated designs violate RIP-type assumptions.
- No randomized Jacobian (Schwartz–Zippel) rank is used. All ranks here are exact over Q for small systems. If one is ever added, the minor degree bound D and the sampling-set size must be recorded, and P(false drop) ≤ D/|S| per trial.
- No 1−1/e guarantee is claimed for adaptive, cost-sensitive exploration. Any log-det or entropy scoring used in play is a **heuristic**.
- Polynomial or nonlinear CRL identifiability results are research hypotheses, not guarantees used by the agent.

## Measured relevance to the competition score (2026-10-12)
B0's public-game loss splits into 50.9 points from levels not completed and 2.6 points from action inefficiency (`scripts/analyze_franzen_log.py`). Tools that minimize **real actions** (principles 7 and 9) therefore address the smaller loss term. Raising every level's actions by 25% would cost 3.2 points. In this regime, the binding cost of an experiment is the model's **tokens** (deliberation), not its actions. Principles 11–13 (separation and decision relevance) remain relevant. Principle 9 is kept as a correct tool, but it is not on the competition path unless a measurement shows value.

## Sandbox tools exposed to the model (`B2_TOOLS`, `kaggle/b2/sandbox_tools.py`; verified in `docs/algebra_verification.md`)
| Tool | Principle (row above) | Assumptions the model must satisfy | Honest-output guarantees |
|---|---|---|---|
| `gf2_solve` | 6, 7 | Presses commute; each press adds its column mod 2; **additive** costs (order-dependent travel is not modelled) | Exact minimum when `exact` is True. `None` means the linear toggle model is wrong |
| `zm_solve` | 8 | The same, mod m (any m) | Exact by enumeration; raises if the space exceeds `max_states` |
| `fit_boolean` | 1–4 | Inputs exactly 0/1; degree ≤ 2; at most 4 terms (MDL preference, not proof) | `unidentified` = the nullspace support of the observed design (exact). `alternatives` counts equally sparse rules. `complete` False if the time budget cut the search |
| `fit_guarded` | 15 | One guard, 1-D, held at a constant beyond it | Slope from points that cannot be capped; wall location reported as an interval |
| `distinguish` | 13 | Deterministic model simulators with known current states | Shortest separating sequence within depth and time. Crashes are reported, not counted. `exhausted` proves only within the model space |
| `bfs_plan` | — (BFS) | Deterministic simulator, hashable states | Shortest plan in the model. "Unreachable" holds only within the model |

## Real-game evidence (public games, level 1; `docs/algebra_on_games.md`)
This evidence comes from engine access the agent does not have. It shows whether a class occurs and whether a tool is correct when given correct inputs, **not** agent performance.
- **Row 6/7 (`gf2_solve`):**
  - The class holds exactly in 2 of 15 click games (ft09, cn04), where plans were 35/35 exact.
  - In an order-dependent game (r11l) one confident plan failed on the engine.
  - The prerequisite check is therefore load-bearing. It has been stated in `TOOL_DOC` since commit 1ec396b.
- **Rows 1–4 (`fit_boolean`, degree ≤ 2):**
  - Predicts held-out 3-button combinations better than XOR where XOR fails, but not exactly: bp35 11/12 frames (99% vs 75% of pixels); s5i5 10/12 frames (85% vs 37%).
  - Correctly reports "no fit" for 3-way parity (cn04), which needs degree 3 over the reals.
- **Row 15 (`fit_guarded`):**
  - Exact on all 25 steady-motion tracks, with all 22 wall intervals containing the observed stop.
  - Correctly refuses the 3 wrap-around tracks.
- **Row 8 (`zm_solve`):** tr87's movement keys form Z₇ and Z₅; 20/20 cheapest-press plans matched on the engine.
- **Row 13 (`bfs_plan` "exhausted"):**
  - 7 of 25 games returned "exhausted" for goals humans reach. In 2 of them (g50t 7/101, sc25 4/4) row 14's test shows the masked frame key merged distinct states (raw-observation check pending). For the other 5 an incomplete action set is a hypothesis without a witness yet.
  - "Unreachable in the model" is not "unreachable".
