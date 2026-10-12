"""Scripted OpenAI-compatible chat server for harness integration tests (no GPU).

It does NOT reason. Every request gets a `python` tool call that inspects the
board through the harness runtime variables and executes one legal action. If
MOCK_CODE_FILE is set, its contents are used as the tool-call code instead
(e.g. to exercise new sandbox tools). Usage counts are synthetic so the
harness's token accounting runs.

  python kaggle/local/mock_llm_server.py --port 8765
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_CODE = r'''
import random
acts = [a for a in valid_actions if a not in ('RESET', 'UNDO')]
seg = current_frame.segmentation
nodes = seg['nodes']
print('level', current_frame.level, 'step', current_frame.step, 'nodes', len(nodes), 'valid', valid_actions)
a = random.choice(acts) if acts else None
if a == 'MOUSE':
    n = random.choice(nodes)
    r, c = n['boundary'][0]
    res = action([{'action': 'MOUSE', 'row': int(r), 'col': int(c)}])
elif a is not None:
    res = action([a])
else:
    res = {}
print('result keys', sorted(res.keys())[:8] if isinstance(res, dict) else type(res))
'''

STATS = {"requests": 0, "started": time.time()}
LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            return self._send(200, {"data": [{"id": os.environ.get("MOCK_MODEL", "mock"), "object": "model"}]})
        if self.path.rstrip("/").endswith("/stats"):
            return self._send(200, STATS)
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(n) or b"{}")
        with LOCK:
            STATS["requests"] += 1
            k = STATS["requests"]
        msgs = req.get("messages", [])
        prompt_chars = sum(len(json.dumps(m)) for m in msgs)
        code = DEFAULT_CODE
        f = os.environ.get("MOCK_CODE_FILE")
        if f and os.path.exists(f):
            code = open(f).read()
        time.sleep(float(os.environ.get("MOCK_LATENCY_S", "0.05")))
        msg = {
            "role": "assistant",
            "content": "",
            "reasoning_content": f"(mock reasoning #{k}) inspect board, then take one legal action.",
            "tool_calls": [{
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {"name": "python", "arguments": json.dumps({"code": code})},
            }],
        }
        resp = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.get("model", "mock"),
            "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": prompt_chars // 4, "completion_tokens": 120,
                      "total_tokens": prompt_chars // 4 + 120,
                      "prompt_tokens_details": {"cached_tokens": 0}},
        }
        self._send(200, resp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"mock LLM on http://127.0.0.1:{a.port}/v1", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
