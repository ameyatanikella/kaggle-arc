"""Regression tests for the Kaggle notebook builders (no network).

Invariants:
- B0: every original Franzen cell is unchanged except two commit-path-only edits.
  The rerun (TRUE_SUBMISSION) path is untouched.
- All builders: code cells parse; the commit path still writes submission.parquet;
  kernel metadata has competition, inputs and no internet.
- B2 / paired: patches and tools are written to /tmp (never to /kaggle/working outputs).
"""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle" / "b0"))
sys.path.insert(0, str(ROOT / "kaggle" / "b2"))
sys.path.insert(0, str(ROOT / "kaggle" / "eval"))


def _code_ok(src: str) -> bool:
    if src.lstrip().startswith("%%"):
        return True
    lines = [ln for ln in src.split("\n") if not ln.lstrip().startswith("!")]
    ast.parse("\n".join(lines).replace("await ", ""))
    return True


def _check_meta(meta):
    assert meta["enable_internet"] is False
    assert "arc-prize-2026-arc-agi-3" in meta["competition_sources"]
    assert meta["model_sources"] and meta["dataset_sources"]


def test_b0_rerun_path_unchanged():
    import build_b0
    build_b0.main("axizcc")          # same as the real build: the push dir stays valid
    orig = json.loads((ROOT / "kaggle/b0/franzen_m2_v3_original.ipynb").read_text())
    new = json.loads((ROOT / "kaggle/b0/push/notebook.ipynb").read_text())
    assert len(new["cells"]) == len(orig["cells"]) + 1          # one intro markdown cell added
    changed = []
    for i, (a, b) in enumerate(zip(orig["cells"], new["cells"][1:])):
        sa, sb = "".join(a["source"]), "".join(b["source"])
        if sa != sb:
            changed.append(i)
            # every changed line must be an addition guarded by B0_SMOKE (commit path only)
            added = [ln for ln in sb.split("\n") if ln not in sa.split("\n")]
            assert any("B0_SMOKE" in ln for ln in added), i
    assert changed == [16, 20]
    for c in new["cells"]:
        if c["cell_type"] == "code":
            assert _code_ok("".join(c["source"]))
            assert c["outputs"] == []
    run = "".join(new["cells"][21]["source"])
    assert "submission.parquet" in run and "if not TRUE_SUBMISSION" in run
    _check_meta(json.loads((ROOT / "kaggle/b0/push/kernel-metadata.json").read_text()))


def test_b2_builder_writes_to_tmp_and_parses():
    import build_b2
    out = build_b2.main("axizcc", "b2test", {"B2_BAR_FIX": "1", "B2_TOOLS": "1"})
    nb = json.loads((out / "notebook.ipynb").read_text())
    srcs = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert all(_code_ok(s) for s in srcs)
    inst = [s for s in srcs if "b2_patches.install()" in s]
    assert inst and "/tmp/b2_patches.py" in inst[0] and "/tmp/sandbox_tools.py" in inst[0]
    assert "/kaggle/working/b2" not in "\n".join(srcs)
    _check_meta(json.loads((out / "kernel-metadata.json").read_text()))
    import shutil
    shutil.rmtree(out)


def test_paired_builder_rerun_and_commit_paths():
    import build_paired
    out = build_paired.main("axizcc", "pairtest", {}, {"B2_BAR_FIX": "1"}, "b", 2, "ba")
    nb = json.loads((out / "notebook.ipynb").read_text())
    srcs = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert all(_code_ok(s) for s in srcs)
    joined = "\n".join(srcs)
    assert "if TRUE_SUBMISSION and SUBMIT_SWITCHES:" in joined          # rerun installs only the submit arm's switches
    assert 'SUBMIT_SWITCHES = {"B2_BAR_FIX": "1"}' in joined
    assert 'PAIRED_ARMS = [["b", {"B2_BAR_FIX": "1"}], ["a", {}]]' in joined   # order ba
    assert "if not PAIRED_DONE:\n    await bm.run(" in joined           # commit path skips the 8h benchmark
    assert "timeout=ARM_SECONDS + 900" in joined
    assert "B0_SMOKE = False" in joined
    run_cell = next(s for s in srcs if "await bm.run(" in s)
    assert "submission.parquet" in run_cell
    import shutil
    shutil.rmtree(out)
    out2 = build_paired.main("axizcc", "pairtest2", {}, {"B2_BAR_FIX": "1"}, "a", 1, "ab")
    j2 = "\n".join("".join(c["source"]) for c in json.loads((out2 / "notebook.ipynb").read_text())["cells"])
    assert "SUBMIT_SWITCHES = {}" in j2                                   # submit arm a = plain B0 rerun
    shutil.rmtree(out2)


def test_b2_minimal_commit_skips_only_benchmark():
    import build_b2
    out = build_b2.main("axizcc", "b2min", {"B2_TOOLS": "1"}, "minimal")
    nb = json.loads((out / "notebook.ipynb").read_text())
    srcs = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    run = next(s for s in srcs if "await bm.run(" in s)
    assert "if TRUE_SUBMISSION:  # minimal commit: no benchmark\n    await bm.run(" in run
    assert "submission.parquet" in run                      # commit still writes the required file
    assert any("b2_patches.install()" in s and "if TRUE_SUBMISSION" not in s for s in srcs)  # installs in commit too
    assert all(_code_ok(s) for s in srcs)
    import shutil
    shutil.rmtree(out)


def test_b2_input_resolver_handles_both_mount_layouts(tmp_path, monkeypatch):
    import shutil
    import build_b2
    out = build_b2.main("axizcc", "b2resolver", {"B2_TOOLS": "1"}, "minimal")
    nb = json.loads((out / "notebook.ipynb").read_text())
    shutil.rmtree(out)
    srcs = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert srcs[0].startswith("# B2 input-path resolver")
    assert all(_code_ok(s) for s in srcs)
    joined = "\n".join(srcs[1:])
    import re
    assert not re.search(r"""(?<!_b2_inp\()["']/kaggle/input/""", joined)       # every literal wrapped
    expected = {
        "/kaggle/input/datasets/dfranzen/taaf-kaggle-source-bundle-copy": "taaf-kaggle-source-bundle-copy",
        "/kaggle/input/datasets/dfranzen/pennyroyal-v253": "pennyroyal-v253",
        "/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1":
            "intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1",
        "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels": "arc-prize-2026-arc-agi-3/arc_agi_3_wheels",
    }
    # legacy layout: /kaggle/input/<slug>/...
    legacy = tmp_path / "legacy"
    for rel in expected.values():
        (legacy / rel).mkdir(parents=True)
    monkeypatch.setenv("B2_INPUT_ROOT", str(legacy))
    ns = {}
    exec(srcs[0], ns)
    for p, rel in expected.items():
        assert ns["_b2_inp"](p) == str(legacy / rel)
    # new layout: identity
    new = tmp_path / "new"
    for p in expected:
        (new / p[len("/kaggle/input/"):]).mkdir(parents=True)
    monkeypatch.setenv("B2_INPUT_ROOT", str(new))
    ns = {}
    exec(srcs[0], ns)
    for p in expected:
        assert ns["_b2_inp"](p) == str(new / p[len("/kaggle/input/"):])
