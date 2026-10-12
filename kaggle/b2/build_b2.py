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


def main(username: str, variant: str, switches: dict) -> Path:
    build_b0.main(username)
    pdir = ROOT / "kaggle" / "b0" / "push"
    nb = json.loads((pdir / "notebook.ipynb").read_text())
    meta = json.loads((pdir / "kernel-metadata.json").read_text())
    src = (HERE / "b2_patches.py").read_text()
    env_lines = "\n".join(f"os.environ[{k!r}] = {v!r}" for k, v in sorted(switches.items()))
    cell_src = (
        "# B2 patches (switchable; all OFF == B0). Written to /tmp so it is not a notebook output.\n"
        "import os, sys, importlib\n"
        f"{env_lines}\n"
        "open('/tmp/b2_patches.py', 'w').write(" + repr(src) + ")\n"
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
    # report B2 counters at the end
    nb["cells"].append({"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                        "source": "import b2_patches, json\nprint('B2 summary:', json.dumps(b2_patches.b2_summary()))\n"})
    nb["cells"][0]["source"] = (
        f"# B2 ({variant}): Franzen M2 solution + switchable harness patches\n\n"
        "Base: [Daniel Franzen's Milestone 2 solution](https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution) "
        "(Apache-2.0), built on Tufa Labs' Duck harness (Jeroen Cottaar). Solved-level memory is a port of rellik13's M85 "
        "(Apache-2.0, github.com/LohitSiriki/arc-agi-3-milestone2-solution). All credit to the original authors.\n\n"
        f"B2 switches: `{json.dumps(switches)}`. The commit run is a 2-game smoke test; the competition rerun is the full run.")
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
    sw = dict(kv.split("=", 1) for kv in sys.argv[3:])
    main(user, variant, sw)
