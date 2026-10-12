# Verification of the algebra-based notebook (`pair-algebra`)

Date: 2026-10-12, about 17:45 UTC. Scope:
- the six sandbox tools (`kaggle/b2/sandbox_tools.py`);
- their prompt documentation;
- their integration into the B0 harness;
- the notebook's two execution paths;
- the evaluation design.

## 1. Defects found and fixed (each found by a test, not by inspection alone)
| # | Defect in the version built earlier today | Evidence | Fix |
|---|---|---|---|
| 1 | **Prompt/runtime interface mismatch.** The prompt said `bfs_plan` and `distinguish` "return the shortest action list/sequence", but they return dicts. `gf2_solve`'s text never said that `b` = target XOR current, nor which keys come back. This is the interface pitfall named in the brief | Probe: `type(bfs_plan(...))` is `dict` | The prompt text is now `TOOL_DOC` inside `sandbox_tools.py` itself, inserted verbatim. A test asserts that every documented key is returned and every returned key is documented |
| 2 | **Could exceed the sandbox's 30 s hard limit.** `gf2_solve` took 12.6 s at nullity 20. `fit_boolean` took 6.9 s on 8 switches. `bfs_plan`/`distinguish` defaulted to 20 s each, so two calls exceed 30 s | Timing probes on an M5 Pro (Kaggle CPUs are slower) | Every tool has an 8 s budget. `gf2_solve` uses bitmasks and Gray-code enumeration, flagged `exact: False` if cut. `fit_boolean` reports `complete: False`. Tests assert return within budget on adversarial inputs |
| 3 | `fit_boolean` silently accepted non-0/1 inputs, changing the model class | Probe: `[[2,0],[0,3]]` accepted | Raises `ValueError` |
| 4 | `fit_guarded` missed walls at the low end (`max(x,3)` → None) | Probe | `cap_low` kind added |
| 5 | `fit_guarded` reported the wall position as exact. The data only bound it to an interval: for `min(x,5)` on 0..8 the wall lies between 4 and 6 | Fixture (also in `math_contract.md` #15) | Returns `guard_between` |
| 6 | `fit_guarded` could take a slope from a point that might lie on the capped side | Test case `[0,5,5]` | The slope is estimated only from points with y ≠ cap |
| 7 | `distinguish` counted a crashing simulator as an observable difference, so it would propose a probe that separates a hypothesis from a bug | Probe | Exceptions go to `errors` and are never counted as a split |

## 2. Correctness evidence (`tests/test_sandbox_tools.py`; 35/35 tests pass repo-wide)
| Tool | Oracle | Instances |
|---|---|---|
| `gf2_solve` | Brute force over {0,1}^n, minimum additive cost. The returned presses re-checked against A u = b | 400 random systems, 0 mismatches |
| `zm_solve` | Brute force over {0..m−1}^n, m ∈ {2,3,4,6} (non-prime included) | 240 |
| `bfs_plan` | Iterative-deepening brute force: same shortest length, the plan reaches the goal, and "unreachable" agrees | 150 random graphs with blocked moves |
| `distinguish` | Brute-force shortest separating sequence | 150 random hypothesis pairs, plus crash and identical-model cases |
| `fit_boolean` | (i) No exact fit with fewer terms exists (exact rank test on every smaller support). (ii) The rule reproduces every observation. (iii) The `unidentified` set equals the support of the design's nullspace, computed by the independent implementation in `arc3/algebra/linalg.py` | 80 random sparse polynomials on random partial observations |
| `fit_guarded` | Random walls at either end: kind and cap recovered, the true wall lies inside `guard_between`, the rule reproduces the data. Two-guard data correctly returns None | 200 |

Fixtures from the brief that are covered:
- OR vs XOR, including the genuine ambiguity: without observing (0,0), the sparser rule `1 − b1·b2` fits, and the tool reports unidentified terms.
- Multiple GF(2) solutions with different costs.
- A non-prime modulus.
- A guard missed by a smooth fit.
- Proof only within the model space (`exhausted`).

## 3. Integration evidence (real harness, scripted model; no score claim)
- All six tools are called exactly as `TOOL_DOC` documents, inside the real sandbox subprocess, with correct outputs.
- A model-defined helper that calls a tool is retained across calls by the harness's persistence mechanism.
- `TOOL_DOC` appears verbatim in the system prompt.
- **Scored rerun path:** the notebook's install cell was executed exactly as the competition rerun would run it. It installs B2c+TOOLS and nothing else (not the nudge). The prompt corrections and tools are present, and all six names are registered in the sandbox.
- **Commit path:** arm A gets B2c without tools, and arm B gets B2c+tools. Switches inherited from the parent environment are cleared.
- The files embedded in the notebook are byte-identical to the repo versions.

## 4. Evaluation design: what the paired run can and cannot show
- **Contrast:** B2c vs B2c+TOOLS. The only difference is the six functions plus about 640 tokens of prompt text (2,244 characters; under 1% of the 128K context).
- **Budget:** 1.9 h per pass per arm, 10 streams, 50 runs per arm. That is about 94% of the competition's per-game compute share.
- **Confounds, stated rather than hidden:**
  1. The arms run sequentially on one server (A first, then B), so time-in-session effects cannot be separated within this run. B2c is also measured in two other sessions (as arm B of `pair-b2c` and arm A of `pair-nudge`), which gives a rough read on session and position variability.
  2. Per-arm compute is checked from generated tokens and tokens per second before any comparison.
- **Power.** With 2 passes × 25 games the paired score delta has an SD of roughly 5 points. Only large effects are detectable. **The score comparison is descriptive.**
- **Mechanism measure.** Tool use is counted from the code the model actually executed: textual call sites in the request logs, via `b2_patches.tool_usage_from_requests`. This is a proxy, because invocations inside the sandbox are not instrumented. If the tools are rarely called, any score difference cannot be attributed to them.
- **Generalization.** The public games are contaminated, so only the hidden leaderboard measures generalization.

## 5. Pre-registered decision rule for the Oct 14 submission (fixed before results)
Submit the `pair-algebra` notebook, whose rerun is B0 + B2c + TOOLS, **unless** one of these holds:
1. Either arm crashed or timed out, or `b2_summary.errors > 0`.
2. Tool injection is missing from arm B's prompts.
3. Fewer than 95% of model turns produced a parseable tool call.
4. Gross regression: a paired score delta (B − A) below −5 points **and** a 90% bootstrap upper bound below 0.

If a blocking condition occurs, fix the cause and submit the corrected algebra notebook if a smoke run passes. Otherwise submit B2c and keep the algebra tools as a research branch.

**Dependency.** The Oct 14 submission includes the B2c changes. If `pair-b2c` shows B2c itself regressing against B0 under rule 4, the algebra submission will be rebuilt on plain B0 (B0+TOOLS), with a 2-game smoke run first.
