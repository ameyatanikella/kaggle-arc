"""Build a paired-A/B notebook.

Commit run ("Save & Run All", not scored): plays arm A = B0 and then arm B = B2
on the 25 public games, each for `passes` passes. Each arm gets about
2.0 h per pass, which roughly matches the competition's per-game compute share
(110 games in about 8.9 h with 10 streams). Arms run as separate subprocesses
against the same SGLang server.

Competition rerun (scored): identical to B0, plus B2 patches iff SUBMIT_ARM == "b2".

Usage: python kaggle/eval/build_paired.py <user> <variant> <submit_arm b0|b2> <passes> KEY=VAL ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "kaggle" / "b0"))
import build_b0  # noqa: E402

ARM_SECONDS_PER_PASS = 6840   # 1.9 h; two arms x 2 passes = 7.6 h + about 10 min startup < 9 h


def main(user: str, variant: str, submit_arm: str, passes: int, switches: dict) -> Path:
    build_b0.main(user)
    pdir = ROOT / "kaggle" / "b0" / "push"
    nb = json.loads((pdir / "notebook.ipynb").read_text())
    meta = json.loads((pdir / "kernel-metadata.json").read_text())
    b2_src = (ROOT / "kaggle" / "b2" / "b2_patches.py").read_text()
    tools_src = (ROOT / "kaggle" / "b2" / "sandbox_tools.py").read_text()
    arm_src = (HERE / "run_arm.py").read_text()
    env_lines = "\n".join(f"os.environ[{k!r}] = {v!r}" for k, v in sorted(switches.items()))

    # 1) undo the B0 smoke limits in the commit path: the paired cell owns that path.
    hook = next(c for c in nb["cells"] if c["cell_type"] == "code" and "B0 commit-mode smoke test" in "".join(c["source"]))
    hook["source"] = "".join(hook["source"]).replace("B0_SMOKE = not TRUE_SUBMISSION", "B0_SMOKE = False")

    # 2) install cell: write helpers to /tmp; in the rerun install B2 iff submit_arm == b2.
    install_cell = (
        "# Paired-eval helpers + B2 switches (written to /tmp, not notebook outputs).\n"
        "import os, sys, importlib, json, subprocess, time\n"
        f"{env_lines}\n"
        f"SUBMIT_ARM = {submit_arm!r}\n"
        f"PAIRED_PASSES = {passes}\n"
        f"ARM_SECONDS = {ARM_SECONDS_PER_PASS} * PAIRED_PASSES\n"
        "open('/tmp/b2_patches.py', 'w').write(" + repr(b2_src) + ")\n"
        "open('/tmp/sandbox_tools.py', 'w').write(" + repr(tools_src) + ")\n"
        "open('/tmp/run_arm.py', 'w').write(" + repr(arm_src) + ")\n"
        "sys.path.insert(0, '/tmp')\n"
        "import b2_patches\n"
        "if TRUE_SUBMISSION and SUBMIT_ARM == 'b2':\n"
        "    print('B2 install (rerun):', b2_patches.install())\n"
    )
    hook_idx = nb["cells"].index(hook)
    nb["cells"].insert(hook_idx + 1, {"cell_type": "code", "metadata": {}, "outputs": [],
                                      "execution_count": None, "source": install_cell})

    # 3) paired-eval cell right before the benchmark cell; the benchmark cell then skips bm.run in commit mode.
    run_idx = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"
                   and "await bm.run(" in "".join(c["source"]))
    paired_cell = (
        "PAIRED_DONE = False\n"
        "if not TRUE_SUBMISSION:\n"
        "    os.environ.setdefault('RECORDINGS_DIR', str(WORKING_DIR / 'server_recording'))\n"
        "    results = {}\n"
        "    for arm in ('b0', 'b2'):\n"
        "        out = WORKING_DIR / f'arm_{arm}'\n"
        "        t0 = time.time()\n"
        "        p = subprocess.run([sys.executable, '/tmp/run_arm.py', arm, str(ARM_SECONDS), str(PAIRED_PASSES), str(out)],\n"
        "                           capture_output=True, text=True)\n"
        "        print(p.stdout[-6000:]); print(p.stderr[-3000:])\n"
        "        results[arm] = {'rc': p.returncode, 'seconds': time.time() - t0}\n"
        "    (WORKING_DIR / 'paired_meta.json').write_text(json.dumps(results, indent=1))\n"
        "    PAIRED_DONE = True\n"
    )
    nb["cells"].insert(run_idx, {"cell_type": "code", "metadata": {}, "outputs": [],
                                 "execution_count": None, "source": paired_cell})
    run_cell = nb["cells"][run_idx + 1]
    src = "".join(run_cell["source"])
    assert src.count("await bm.run(") == 1
    src = src.replace("await bm.run(", "if not PAIRED_DONE:\n    await bm.run(", 1)
    run_cell["source"] = src
    nb["cells"][0]["source"] = (
        f"# Paired eval ({variant}): B0 vs B2 on public games; rerun submits {submit_arm}\n\n"
        "Base: [Daniel Franzen's Milestone 2 solution](https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution) "
        "(Apache-2.0) on Tufa Labs' Duck harness. Solved-level memory ports rellik13's M85 (Apache-2.0). "
        f"B2 switches: `{json.dumps(switches)}`.\n\nCommit run: arm b0 then arm b2, {passes} pass(es) each on the 25 public games. "
        "Competition rerun: identical to B0" + (" plus B2 patches." if submit_arm == "b2" else "."))
    out = HERE / "push" / variant
    out.mkdir(parents=True, exist_ok=True)
    (out / "notebook.ipynb").write_text(json.dumps(nb, indent=1))
    meta["id"] = f"{user}/arc3-{variant}"
    meta["title"] = f"arc3-{variant}"
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print("wrote", out)
    return out


if __name__ == "__main__":
    user, variant, submit_arm, passes = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    main(user, variant, submit_arm, passes, dict(kv.split("=", 1) for kv in sys.argv[5:]))
