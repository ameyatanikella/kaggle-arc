"""The paired notebook waits for the model server before its first arm (GRCM T3 start-up fault)."""
import copy
import http.server
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle" / "eval"))
import build_paired  # noqa: E402
import build_paired_ready as bpr  # noqa: E402

VARIANT = "pytest-ready-tmp"


@pytest.fixture(scope="module")
def notebooks():
    b0_push = ROOT / "kaggle" / "b0" / "push"
    saved = {p.name: p.read_bytes() for p in b0_push.iterdir() if p.is_file()}
    try:
        out = build_paired.main("u", VARIANT, {}, {"B2_HX_STALL": "1"}, "none", 1, "ab")
        plain = json.loads((out / "notebook.ipynb").read_text())
    finally:
        shutil.rmtree(ROOT / "kaggle" / "eval" / "push" / VARIANT, ignore_errors=True)
        for name, data in saved.items():             # building B0 rewrites its push dir; restore it
            (b0_push / name).write_bytes(data)
    return plain, bpr.patch(copy.deepcopy(plain), timeout=1)


def _paired_index(nb):
    return next(i for i, c in enumerate(nb["cells"]) if "".join(c["source"]).lstrip().startswith(
        ("PAIRED_DONE = False", "def _paired_wait_ready")))


def test_only_paired_and_title_cells_change(notebooks):
    plain, ready = notebooks
    assert len(plain["cells"]) == len(ready["cells"])
    i = _paired_index(plain)
    changed = [k for k, (a, b) in enumerate(zip(plain["cells"], ready["cells"]))
               if "".join(a["source"]) != "".join(b["source"])]
    assert changed == [0, i]


def test_wait_precedes_arms_and_rerun_path_untouched(notebooks):
    _, ready = notebooks
    src = "".join(ready["cells"][_paired_index(ready)]["source"])
    assert src.index("_paired_wait_ready(f'http://127.0.0.1:") < src.index("for label, sw in PAIRED_ARMS")
    head, body = src.split("if not TRUE_SUBMISSION:\n", 1)
    assert "_paired_wait_ready(f" not in head            # only called on the commit run


class _Flaky(http.server.BaseHTTPRequestHandler):
    calls = 0

    def do_GET(self):
        type(self).calls += 1
        self.send_response(200 if type(self).calls > 2 else 503)
        self.end_headers()

    def log_message(self, *a):
        pass


def _serve():
    _Flaky.calls = 0
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Flaky)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_wait_function_ready_and_timeout():
    ns = {}
    exec(bpr.WAIT_FN, ns)
    srv = _serve()
    try:
        ok, waited = ns["_paired_wait_ready"](f"http://127.0.0.1:{srv.server_port}/health", 10, 0.05)
    finally:
        srv.shutdown()
    assert ok and _Flaky.calls == 3
    ok, waited = ns["_paired_wait_ready"](f"http://127.0.0.1:{_free_port()}/health", 0.3, 0.05)
    assert not ok and waited >= 0.3


def _run_cell(src, port, tmp_path):
    g = {"TRUE_SUBMISSION": False, "WORKING_DIR": tmp_path, "PAIRED_ARMS": [["a", {}]], "PAIRED_PASSES": 1,
         "ARM_SECONDS": 5, "SERVED_MODEL_PORT": port, "os": os, "sys": sys, "json": json,
         "subprocess": subprocess, "time": time}
    exec(src, g)
    return g, json.loads((tmp_path / "paired_meta.json").read_text())


def test_cell_plays_no_arm_when_server_never_ready(notebooks, tmp_path):
    _, ready = notebooks
    src = "".join(ready["cells"][_paired_index(ready)]["source"])
    g, meta = _run_cell(src, _free_port(), tmp_path)
    assert g["PAIRED_DONE"] and meta["server_ready"]["ready"] is False and "a" not in meta


def test_cell_plays_arms_after_server_ready(notebooks, tmp_path):
    _, ready = notebooks
    src = "".join(ready["cells"][_paired_index(ready)]["source"])
    srv = _serve()
    try:
        g, meta = _run_cell(src.replace(", 1)\n", ", 10)\n", 1), srv.server_port, tmp_path)
    finally:
        srv.shutdown()
    assert meta["server_ready"]["ready"] is True and "a" in meta   # /tmp/run_arm.py is absent here, so rc != 0
