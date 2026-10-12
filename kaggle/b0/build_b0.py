"""Build the B0 notebook: Daniel Franzen's Milestone-2 solution (Apache-2.0), unchanged
on the competition-rerun path, with one change on the commit path.

Change (non-submission path only): the interactive "Save & Run All" commit plays
a 2-game, 1-pass, ~20-minute smoke test instead of the 8-hour 25x4 benchmark. This
saves RTX quota while still exercising install -> model serving -> harness ->
arc_agi offline games -> dummy submission.parquet. The TRUE_SUBMISSION
(KAGGLE_IS_COMPETITION_RERUN) path is byte-for-byte the original.

Usage: python kaggle/b0/build_b0.py <kaggle_username>
Writes kaggle/b0/push/{notebook.ipynb,kernel-metadata.json}.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "franzen_m2_v3_original.ipynb"
SRC_SHA256 = "589eaf393f416effe925c6830fc00fcdc0d6da91193b29ff0029282cbd6ee96e"

SMOKE_HOOK = '''
# --- B0 commit-mode smoke test (not used in the competition rerun) ---------
B0_SMOKE = not TRUE_SUBMISSION
B0_SMOKE_GAMES = ("ls20", "vc33")
B0_SMOKE_MINUTES = 20
if B0_SMOKE:
    bm.n_passes = 1
    bm.solver.concurrency = len(B0_SMOKE_GAMES)
    bm.solver.max_runtime_s_per_game = B0_SMOKE_MINUTES * 60
    print("B0 smoke mode: commit run limited to", B0_SMOKE_GAMES, B0_SMOKE_MINUTES, "min")
'''

RUN_ANCHOR = "    bm.games = _offline_games(competition_env_files)\n"
RUN_PATCH = (RUN_ANCHOR +
             "    if B0_SMOKE:\n"
             "        bm.games = [g for g in bm.games if any(g.env_name.startswith(p) for p in B0_SMOKE_GAMES)]\n"
             "        print('B0 smoke games:', [g.env_name for g in bm.games])\n")
SOFT_ANCHOR = "        soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(seconds=budget - min(600.0, budget / 2))\n"
SOFT_PATCH = (SOFT_ANCHOR +
              "    if B0_SMOKE:\n"
              "        soft_end = datetime.now() + timedelta(minutes=B0_SMOKE_MINUTES + 2)\n")


def main(username: str) -> None:
    raw = SRC.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == SRC_SHA256, f"source notebook changed: {sha}"
    nb = json.loads(raw)
    out = copy.deepcopy(nb)
    for c in out["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None
    hook = out["cells"][16]
    src = "".join(hook["source"])
    assert "print('benchmark.solver:', bm.solver)" in src
    hook["source"] = src.replace("# show config\n", SMOKE_HOOK + "\n# show config\n")
    run = out["cells"][20]
    src = "".join(run["source"])
    assert src.count(RUN_ANCHOR) == 1 and src.count(SOFT_ANCHOR) == 1
    src = src.replace(RUN_ANCHOR, RUN_PATCH).replace(SOFT_ANCHOR, SOFT_PATCH)
    run["source"] = src
    intro = {"cell_type": "markdown", "metadata": {}, "source": (
        "# B0: Franzen ARC-AGI-3 Milestone-2 solution (reproduction)\n\n"
        "Unmodified copy of [Daniel Franzen's Milestone 2 solution](https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution) "
        "(Apache-2.0), itself built on Tufa Labs' Duck harness (Jeroen Cottaar). All credit to the original authors.\n\n"
        "Only change: the non-scored commit run plays a 2-game smoke test instead of the 8-hour public-game benchmark. "
        "The competition-rerun path is unchanged.")}
    out["cells"].insert(0, intro)
    pdir = HERE / "push"
    pdir.mkdir(exist_ok=True)
    (pdir / "notebook.ipynb").write_text(json.dumps(out, indent=1))
    meta = {
        "id": f"{username}/arc3-b0-franzen-m2-repro",
        "title": "arc3-b0-franzen-m2-repro",
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaRtxPro6000",
        "dataset_sources": ["dfranzen/pennyroyal-v253", "dfranzen/taaf-kaggle-source-bundle-copy"],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "kernel_sources": [],
        "model_sources": [
            "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1",
            "dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1",
        ],
    }
    (pdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print("wrote", pdir)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "USERNAME")
