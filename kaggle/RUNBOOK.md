# Kaggle runbook

All commands run from the repo root. The token file is `.kaggle/access_token` (git-ignored); the user places it.

## 0. Readiness check (read-only)
```bash
bash kaggle/kaggle_ready.sh
```
It prints:
- the weekly GPU quota and the competition entry status;
- whether my archived copy of Franzen's notebook equals the live v3;
- the live notebook's `kernel-metadata.json`, which gives the exact `machine_shape` and the input slugs.

Copy those slugs into `kaggle/b0/build_b0.py` if they differ.

## 1. B0: first valid submission (fallback, kept immutable)
```bash
.venv/bin/python kaggle/b0/build_b0.py <USERNAME>
KAGGLE_API_TOKEN=$(cat .kaggle/access_token) .venv/bin/kaggle kernels push -p kaggle/b0/push
KAGGLE_API_TOKEN=$(cat .kaggle/access_token) .venv/bin/kaggle kernels status <USERNAME>/arc3-b0-franzen-m2-repro
```
The commit run is a 20-min smoke test on 2 public games. Before submitting, check the log (`kaggle kernels logs ...`):
- the model server reached "ready";
- both games ran;
- `submission.parquet` was written.

Submitting uses the day's single submission and needs explicit user OK:
```bash
KAGGLE_API_TOKEN=$(cat .kaggle/access_token) .venv/bin/kaggle competitions submit arc-prize-2026-arc-agi-3 \
   -k <USERNAME>/arc3-b0-franzen-m2-repro -v <VERSION> -f submission.parquet -m "B0 Franzen M2 repro"
```
A finished commit run does not prove the rerun will score. Confirm the score on the Submissions page or with `kaggle competitions submissions`.

## 2. Paired public-game A/B (uses about 8 h of RTX quota; not a submission)
```bash
.venv/bin/python kaggle/eval/build_paired.py <USERNAME> pair-b2c b0 2 B2_BAR_FIX=1 B2_PROMPT_FIXES=1 B2_SOLVED_MEMORY=1
KAGGLE_API_TOKEN=... .venv/bin/kaggle kernels push -p kaggle/eval/push/pair-b2c
# after completion (~8 h):
KAGGLE_API_TOKEN=... .venv/bin/kaggle kernels output <USERNAME>/arc3-pair-b2c -p runs/kaggle/pair-b2c \
   --file-pattern '(arm_b0|arm_b2)/(benchmark\.json|summary\.txt|arm_meta\.json)|paired_meta\.json'
.venv/bin/python scripts/paired_compare.py runs/kaggle/pair-b2c/arm_b0 runs/kaggle/pair-b2c/arm_b2
```
Read B2 counters from `arm_b2/arm_meta.json` (did each switch fire?) together with the paired delta. Apply the promotion rule in `docs/plan.md`.

## 3. Submitting a B2 variant
```bash
.venv/bin/python kaggle/b2/build_b2.py <USERNAME> b2c B2_BAR_FIX=1 B2_PROMPT_FIXES=1 B2_SOLVED_MEMORY=1
```
Then push and submit as in step 1. Do this only after the paired A/B clears the promotion rule.

## 4. Franzen v3 transcripts for failure analysis (download: several GB)
```bash
KAGGLE_API_TOKEN=... .venv/bin/kaggle kernels output dfranzen/arc-agi-3-milestone-2-solution -p data/franzen_v3_output \
   --file-pattern '(benchmark\.json|summary\.txt|transcripts/.*|serve\.log)'
```

## Input mount layout varies between kernels (observed 2026-10-12/13)
- **Kernels pushed on Oct 12** (B0 smoke, pair-b2c, pair-nudge) mounted inputs under the new layout:
  - `/kaggle/input/datasets/<owner>/<slug>`
  - `/kaggle/input/models/<owner>/<model>/transformers/default/1`
  - `/kaggle/input/competitions/<slug>/...`
- **`arc3-b2-algebra` (pushed Oct 13)** mounted datasets and competition data under the legacy layout (`/kaggle/input/<slug>`), but models under the new layout.
  - v1 failed in cell 2 (`cp: cannot stat .../datasets/dfranzen/taaf-kaggle-source-bundle-copy`).
- **Fix:** `build_b2.add_input_resolver` inserts a resolver cell and wraps the notebook's 6 input-path literals.
  - When the expected path exists, it is used unchanged.
  - Otherwise the slug is searched under `/kaggle/input`.
  - The cell prints the mount tree, and the submit script refuses to submit if any path is `NOT FOUND`.
- **Every future push must go through `build_b2.py`.** A plain `build_b0.py` notebook would fail on a legacy-layout kernel.
- **Only one pending submission per team** (`FAILED_PRECONDITION`): `kaggle/submit_when_free.sh` waits for scoring to finish, then submits.
