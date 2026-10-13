"""Build a paired-A/B notebook with two arbitrary switch sets.

Commit run ("Save & Run All", not scored): plays arm A then arm B (or B then A)
on the 25 public games, `passes` each. Each pass gets about 1.9 h of wall time per
arm, which roughly matches the competition's per-game compute share. Each arm runs
as its own subprocess against the same SGLang server, with a hard timeout.

Competition rerun (scored): identical to B0 plus the B2 patches of the chosen
submit arm (none if that arm has no switches).

usage:
  python kaggle/eval/build_paired.py --user axizcc --variant pair-nudge \
      --arm-a "B2_BAR_FIX=1,B2_PROMPT_FIXES=1,B2_SOLVED_MEMORY=1" \
      --arm-b "B2_BAR_FIX=1,B2_PROMPT_FIXES=1,B2_SOLVED_MEMORY=1,B2_STUCK_NUDGE=1" \
      --submit b --passes 2 --order ba
An empty --arm-a means plain B0.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "kaggle" / "b0"))
import build_b0  # noqa: E402
sys.path.insert(0, str(ROOT / "kaggle" / "b2"))
import build_b2  # noqa: E402

ARM_SECONDS_PER_PASS = 6840   # 1.9 h; two arms x 2 passes = 7.6 h + about 10 min startup < 9 h


def parse_switches(s: str) -> dict:
    return dict(kv.split("=", 1) for kv in s.split(",") if kv.strip()) if s else {}


def main(user, variant, arm_a: dict, arm_b: dict, submit: str, passes: int, order: str = "ab") -> Path:
    build_b0.main(user)
    pdir = ROOT / "kaggle" / "b0" / "push"
    nb = json.loads((pdir / "notebook.ipynb").read_text())
    meta = json.loads((pdir / "kernel-metadata.json").read_text())
    b2_src = (ROOT / "kaggle" / "b2" / "b2_patches.py").read_text()
    tools_src = (ROOT / "kaggle" / "b2" / "sandbox_tools.py").read_text()
    arm_src = (HERE / "run_arm.py").read_text()
    arms = {"a": arm_a, "b": arm_b}
    submit_sw = arms[submit] if submit in arms else {}
    seq = [("a", arm_a), ("b", arm_b)] if order == "ab" else [("b", arm_b), ("a", arm_a)]

    hook = next(c for c in nb["cells"] if c["cell_type"] == "code" and "B0 commit-mode smoke test" in "".join(c["source"]))
    hook["source"] = "".join(hook["source"]).replace("B0_SMOKE = not TRUE_SUBMISSION", "B0_SMOKE = False")

    install_cell = (
        "# Paired-eval helpers + B2 switches (written to /tmp, not notebook outputs).\n"
        "import os, sys, importlib, json, subprocess, time\n"
        f"SUBMIT_SWITCHES = {json.dumps(submit_sw)}\n"
        f"PAIRED_ARMS = {json.dumps(seq)}\n"
        f"PAIRED_PASSES = {passes}\n"
        f"ARM_SECONDS = {ARM_SECONDS_PER_PASS} * PAIRED_PASSES\n"
        "open('/tmp/b2_patches.py', 'w').write(" + repr(b2_src) + ")\n"
        "open('/tmp/sandbox_tools.py', 'w').write(" + repr(tools_src) + ")\n"
        "open('/tmp/run_arm.py', 'w').write(" + repr(arm_src) + ")\n"
        + build_b2.hx_embed_lines() +
        "sys.path.insert(0, '/tmp')\n"
        "if TRUE_SUBMISSION and SUBMIT_SWITCHES:\n"
        "    os.environ.update(SUBMIT_SWITCHES)\n"
        "    import b2_patches\n"
        "    print('B2 install (rerun):', b2_patches.install())\n"
    )
    hook_idx = nb["cells"].index(hook)
    nb["cells"].insert(hook_idx + 1, {"cell_type": "code", "metadata": {}, "outputs": [],
                                      "execution_count": None, "source": install_cell})

    run_idx = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"
                   and "await bm.run(" in "".join(c["source"]))
    paired_cell = (
        "PAIRED_DONE = False\n"
        "if not TRUE_SUBMISSION:\n"
        "    os.environ.setdefault('RECORDINGS_DIR', str(WORKING_DIR / 'server_recording'))\n"
        "    results = {}\n"
        "    for label, sw in PAIRED_ARMS:\n"
        "        out = WORKING_DIR / f'arm_{label}'\n"
        "        t0 = time.time()\n"
        "        try:\n"
        "            p = subprocess.run([sys.executable, '/tmp/run_arm.py', label, str(ARM_SECONDS), str(PAIRED_PASSES),\n"
        "                                str(out), json.dumps(sw)], capture_output=True, text=True, timeout=ARM_SECONDS + 900)\n"
        "            print(p.stdout[-6000:]); print(p.stderr[-3000:])\n"
        "            results[label] = {'rc': p.returncode, 'seconds': time.time() - t0, 'switches': sw}\n"
        "        except subprocess.TimeoutExpired:\n"
        "            results[label] = {'rc': 'timeout', 'seconds': time.time() - t0, 'switches': sw}\n"
        "            print('ARM TIMEOUT', label)\n"
        "    (WORKING_DIR / 'paired_meta.json').write_text(json.dumps(results, indent=1))\n"
        "    PAIRED_DONE = True\n"
    )
    nb["cells"].insert(run_idx, {"cell_type": "code", "metadata": {}, "outputs": [],
                                 "execution_count": None, "source": paired_cell})
    run_cell = nb["cells"][run_idx + 1]
    src = "".join(run_cell["source"])
    assert src.count("await bm.run(") == 1
    run_cell["source"] = src.replace("await bm.run(", "if not PAIRED_DONE:\n    await bm.run(", 1)
    nb["cells"][0]["source"] = (
        f"# Paired eval ({variant})\n\n"
        "Base: [Daniel Franzen's Milestone 2 solution](https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution) "
        "(Apache-2.0) on Tufa Labs' Duck harness. Solved-level memory ports rellik13's M85 (Apache-2.0).\n\n"
        f"Arm a: `{json.dumps(arm_a)}`; arm b: `{json.dumps(arm_b)}`; order {order}; {passes} pass(es) each on the public games "
        f"(commit run). Competition rerun: B0 + switches `{json.dumps(submit_sw)}`.")
    n_paths = build_b2.add_input_resolver(nb)
    assert n_paths == 6, n_paths                       # the six input paths of the B0 notebook
    out = HERE / "push" / variant
    out.mkdir(parents=True, exist_ok=True)
    (out / "notebook.ipynb").write_text(json.dumps(nb, indent=1))
    meta["id"] = f"{user}/arc3-{variant}"
    meta["title"] = f"arc3-{variant}"
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print("wrote", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--arm-a", default="")
    ap.add_argument("--arm-b", default="")
    ap.add_argument("--submit", default="b", choices=["a", "b", "none"])
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--order", default="ab", choices=["ab", "ba"])
    a = ap.parse_args()
    main(a.user, a.variant, parse_switches(a.arm_a), parse_switches(a.arm_b), a.submit, a.passes, a.order)
