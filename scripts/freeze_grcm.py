"""Create or verify immutable input manifests for the approved GRCM T3/T4 tests."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T3 = (
    "docs/prereg_grcm_t3.md", "scripts/analyze_grcm_t3.py", "scripts/freeze_grcm.py",
    "kaggle/eval/build_paired.py", "kaggle/eval/run_arm.py",
    "kaggle/b0/build_b0.py", "kaggle/b0/franzen_m2_v3_original.ipynb",
    "kaggle/b2/build_b2.py", "kaggle/b2/b2_patches.py", "kaggle/b2/sandbox_tools.py",
    "kaggle/b2/hx_patch.py", "kaggle/b2/hx_child.py",
    "arc3/__init__.py", "arc3/perception.py",
    "kaggle/eval/push/grcm-t3-ab/notebook.ipynb",
    "kaggle/eval/push/grcm-t3-ab/kernel-metadata.json",
    "kaggle/eval/push/grcm-t3-ba/notebook.ipynb",
    "kaggle/eval/push/grcm-t3-ba/kernel-metadata.json",
)
T4 = (
    "docs/prereg_grcm_t4.md", "scripts/test_grcm_t4.py", "scripts/freeze_grcm.py",
    "tests/test_grcm.py", "arc3/grcm/__init__.py", "arc3/grcm/model.py",
    "arc3/grcm/algebra.py", "arc3/mcw/objects.py", "arc3/hx/events.py",
    "arc3/algebra/design.py", "arc3/algebra/modular.py", "arc3/procgen/families.py",
    "arc3/procgen/world.py",
)


def files(study: str) -> list[str]:
    if study == "t3":
        return sorted(set(T3) | {str(p.relative_to(ROOT)) for base in ("arc3/hx", "arc3/learning")
                                for p in (ROOT / base).glob("*.py")})
    if study == "t4":
        return sorted(set(T4))
    raise ValueError(study)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(study: str) -> Path:
    return ROOT / "runs/grcm" / study / "FROZEN.json"


def freeze(study: str) -> None:
    path = manifest(study)
    if path.exists():
        raise FileExistsError(f"never overwrite a freeze: {path}")
    paths = files(study)
    missing = [f for f in paths if not (ROOT / f).is_file()]
    if missing:
        raise FileNotFoundError(missing)
    record = {"study": f"grcm-{study}", "sha256": {f: sha(ROOT / f) for f in paths},
              "cases": ("25 public games x one pass x AB and BA" if study == "t3"
                        else "8 fixed first levels: heldout_comp 2, unseen_prim 2, goal_change 2, appearance 2")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"froze {len(paths)} inputs in {path}")


def verify(study: str) -> None:
    path = manifest(study)
    record = json.loads(path.read_text())
    drift = [f for f, digest in record["sha256"].items()
             if not (ROOT / f).is_file() or sha(ROOT / f) != digest]
    if drift:
        raise RuntimeError(f"frozen input drift: {drift}")
    print(f"verified {len(record['sha256'])} frozen inputs for {study}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "verify"))
    parser.add_argument("study", choices=("t3", "t4"))
    args = parser.parse_args()
    (freeze if args.action == "freeze" else verify)(args.study)
