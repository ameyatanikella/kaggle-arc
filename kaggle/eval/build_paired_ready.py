"""Paired-A/B notebook whose first arm waits for the model server to be ready.

Same as ``build_paired.py`` (which is left unchanged: it is a frozen GRCM T3 input), plus
one fix found in T3 (docs/grcm_t3_reach_report.md). Franzen's start-up cell releases the
benchmark at its deadline (about 722 s) even if SGLang is still loading. In a paired commit
run the first arm then starts against a dead endpoint and abandons games after 11 failed
analyzer requests: 9/25 games were unplayed in each T3 session's first arm.

The paired cell now polls the same /health endpoint the start-up cell uses before playing
any arm, records how long it waited, and refuses to play the arms (recording why) if the
server is still not ready after READY_TIMEOUT seconds. Only the commit run (not
TRUE_SUBMISSION) is affected; the competition rerun path is byte-identical to build_paired.

usage: same arguments as build_paired.py, plus --ready-timeout (default 1800 s).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_paired  # noqa: E402

READY_TIMEOUT = 1800

WAIT_FN = '''def _paired_wait_ready(url, timeout, interval=5.0):
    """Poll url until HTTP 200. Returns (ready, seconds_waited)."""
    import time as _t, urllib.request as _u
    t0 = _t.time()
    while True:
        try:
            with _u.urlopen(url, timeout=10) as r:
                if r.status == 200:
                    return True, _t.time() - t0
        except Exception:
            pass
        if _t.time() - t0 >= timeout:
            return False, _t.time() - t0
        _t.sleep(interval)
'''

OLD = "if not TRUE_SUBMISSION:\n    os.environ.setdefault('RECORDINGS_DIR', str(WORKING_DIR / 'server_recording'))\n    results = {}\n"


def new_block(timeout: int) -> str:
    return (
        "if not TRUE_SUBMISSION:\n"
        "    os.environ.setdefault('RECORDINGS_DIR', str(WORKING_DIR / 'server_recording'))\n"
        "    results = {}\n"
        f"    _ready, _waited = _paired_wait_ready(f'http://127.0.0.1:{{SERVED_MODEL_PORT}}/health', {timeout})\n"
        "    results['server_ready'] = {'ready': _ready, 'waited_seconds': round(_waited, 1)}\n"
        "    print(f'PAIRED server ready={_ready} after waiting {_waited:.0f}s', flush=True)\n"
        "    if not _ready:\n"
        "        print('PAIRED: model server never became ready; no arm is played', flush=True)\n"
        "        PAIRED_ARMS = []\n"
    )


def patch(nb: dict, timeout: int = READY_TIMEOUT) -> dict:
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"
             and "".join(c["source"]).startswith("PAIRED_DONE = False")]
    assert len(cells) == 1, len(cells)
    src = "".join(cells[0]["source"])
    assert src.count(OLD) == 1
    cells[0]["source"] = WAIT_FN + "\n" + src.replace(OLD, new_block(timeout), 1)
    nb["cells"][0]["source"] = "".join(nb["cells"][0]["source"]) + (
        f"\n\nPaired arms start only after the model server answers /health (up to {timeout} s).")
    return nb


def main(user, variant, arm_a, arm_b, submit, passes, order="ab", timeout=READY_TIMEOUT) -> Path:
    out = build_paired.main(user, variant, arm_a, arm_b, submit, passes, order)
    path = out / "notebook.ipynb"
    path.write_text(json.dumps(patch(json.loads(path.read_text()), timeout), indent=1))
    print("patched", path, "(server-ready wait before the first arm)")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--arm-a", default="")
    ap.add_argument("--arm-b", default="")
    ap.add_argument("--submit", default="b", choices=["a", "b", "none"])
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--order", default="ab", choices=["ab", "ba"])
    ap.add_argument("--ready-timeout", type=int, default=READY_TIMEOUT)
    a = ap.parse_args()
    main(a.user, a.variant, build_paired.parse_switches(a.arm_a), build_paired.parse_switches(a.arm_b),
         a.submit, a.passes, a.order, a.ready_timeout)
