"""Build a B2 notebook = B0 notebook + one cell that installs kaggle/b2/b2_patches.py
with the chosen switches (env vars). Rerun path otherwise identical to B0.

Usage: python kaggle/b2/build_b2.py <kaggle_username> <variant_name> KEY=VAL [KEY=VAL ...]
e.g.   python kaggle/b2/build_b2.py me b2a B2_BAR_FIX=1 B2_SOLVED_MEMORY=1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "kaggle" / "b0"))
import build_b0  # noqa: E402


INPUT_RESOLVER = r'''# B2 input-path resolver. The B0 notebook hard-codes the new mount layout
# (/kaggle/input/{datasets,models,competitions}/...). A commit on 2026-10-13 found the dataset
# missing there. When the expected path exists it is returned unchanged (identical behaviour);
# otherwise the slug is searched under /kaggle/input (depth <= 3, e.g. the legacy layout).
import os as _b2os
_B2_ROOT = _b2os.environ.get("B2_INPUT_ROOT", "/kaggle/input")


def _b2_inp(p):
    q = p.replace("/kaggle/input", _B2_ROOT, 1)
    if _b2os.path.exists(q):
        return q
    parts = p[len("/kaggle/input/"):].split("/")
    if parts[0] in ("datasets", "models") and len(parts) >= 3:
        slug, tail = parts[2], parts[3:]
    elif parts[0] == "competitions" and len(parts) >= 2:
        slug, tail = parts[1], parts[2:]
    else:
        return q
    for root, dirs, _ in _b2os.walk(_B2_ROOT):
        if root[len(_B2_ROOT):].count("/") >= 3:
            dirs[:] = []
            continue
        if slug in dirs:
            base = _b2os.path.join(root, slug)
            for t in (tail, [x.lower() for x in tail], [x[:1].upper() + x[1:] for x in tail]):
                cand = _b2os.path.join(base, *t)
                if _b2os.path.exists(cand):
                    print(f"B2 input path: {p} -> {cand}")
                    return cand
    print(f"B2 input path NOT FOUND: {p}")
    return q


for _r, _d, _f in _b2os.walk(_B2_ROOT):
    _lvl = _r[len(_B2_ROOT):].count("/")
    if _lvl >= 3:
        _d[:] = []
        continue
    print("B2 inputs:", "  " * _lvl + (_r[len(_B2_ROOT):] or "/"), f"({len(_f)} files)")
'''


def add_input_resolver(nb: dict) -> int:
    """Insert the resolver before the first code cell and wrap every '/kaggle/input/...' string
    literal in code cells with _b2_inp(...). Returns the number of literals wrapped."""
    import re
    pat = re.compile(r"""(["'])(/kaggle/input/[^"'\n]+)\1""")
    n = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if "open('/tmp/b2_patches.py'" in src:
            continue                                   # embedded file contents are data, not paths
        new, k = pat.subn(lambda m: f"_b2_inp({m.group(1)}{m.group(2)}{m.group(1)})", src)
        if k:
            c["source"] = new
            n += k
    first = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code")
    nb["cells"].insert(first, {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                               "source": INPUT_RESOLVER})
    return n


def main(username: str, variant: str, switches: dict, commit_mode: str = "smoke") -> Path:
    """commit_mode: "smoke" = 2-game 20-min commit; "minimal" = commit boots the server and installs
    the patches but skips the benchmark (about 12 min). The rerun path is identical in both."""
    build_b0.main(username)
    pdir = ROOT / "kaggle" / "b0" / "push"
    nb = json.loads((pdir / "notebook.ipynb").read_text())
    meta = json.loads((pdir / "kernel-metadata.json").read_text())
    src = (HERE / "b2_patches.py").read_text()
    tools_src = (HERE / "sandbox_tools.py").read_text()
    env_lines = "\n".join(f"os.environ[{k!r}] = {v!r}" for k, v in sorted(switches.items()))
    cell_src = (
        "# B2 patches (switchable; all OFF == B0). Written to /tmp so it is not a notebook output.\n"
        "import os, sys, importlib\n"
        f"{env_lines}\n"
        "open('/tmp/b2_patches.py', 'w').write(" + repr(src) + ")\n"
        "open('/tmp/sandbox_tools.py', 'w').write(" + repr(tools_src) + ")\n"
        "sys.path.insert(0, '/tmp')\n"
        "import b2_patches\n"
        "importlib.reload(b2_patches)\n"
        "B2_INSTALL = b2_patches.install()\n"
        "print('B2 install:', B2_INSTALL)\n"
    )
    # insert right after the customization hook (B0 cell index 17 after the intro cell)
    hook_idx = next(i for i, c in enumerate(nb["cells"])
                    if c["cell_type"] == "code" and "B0 commit-mode smoke test" in "".join(c["source"]))
    nb["cells"].insert(hook_idx + 1, {"cell_type": "code", "metadata": {}, "outputs": [],
                                      "execution_count": None, "source": cell_src})
    if commit_mode == "minimal":
        run_cell = next(c for c in nb["cells"] if c["cell_type"] == "code" and "await bm.run(" in "".join(c["source"]))
        rs = "".join(run_cell["source"])
        assert rs.count("await bm.run(") == 1
        run_cell["source"] = rs.replace("await bm.run(", "if TRUE_SUBMISSION:  # minimal commit: no benchmark\n    await bm.run(", 1)
    # report B2 counters at the end
    nb["cells"].append({"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                        "source": "import b2_patches, json\nprint('B2 summary:', json.dumps(b2_patches.b2_summary()))\n"})
    nb["cells"][0]["source"] = (
        f"# B2 ({variant}): Franzen M2 solution + switchable harness patches\n\n"
        "Base: [Daniel Franzen's Milestone 2 solution](https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution) "
        "(Apache-2.0), built on Tufa Labs' Duck harness (Jeroen Cottaar). Solved-level memory is a port of rellik13's M85 "
        "(Apache-2.0, github.com/LohitSiriki/arc-agi-3-milestone2-solution). All credit to the original authors.\n\n"
        f"B2 switches: `{json.dumps(switches)}`. "
        + ("The commit run only boots the model server and installs the patches (no games); "
           if commit_mode == "minimal" else "The commit run is a 2-game smoke test; ")
        + "the competition rerun is the full run.")
    n_paths = add_input_resolver(nb)
    assert n_paths == 6, n_paths                       # the six input paths of the B0 notebook
    out = HERE / "push" / variant
    out.mkdir(parents=True, exist_ok=True)
    (out / "notebook.ipynb").write_text(json.dumps(nb, indent=1))
    meta["id"] = f"{username}/arc3-{variant}"
    meta["title"] = f"arc3-{variant}"
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print("wrote", out)
    return out


if __name__ == "__main__":
    user, variant = sys.argv[1], sys.argv[2]
    mode = "smoke"
    args = sys.argv[3:]
    if args and args[0].startswith("--commit="):
        mode = args.pop(0).split("=", 1)[1]
    sw = dict(kv.split("=", 1) for kv in args)
    main(user, variant, sw, mode)
