"""Offline benchmark for restricted algebraic dedup (docs/prereg_b3.md component D).

Equality of two candidate expressions by three methods:
  expand   ordinary canonicalization (word dictionaries)
  matrix   established randomized identity test (random k x k matrices over F_p)
  exact    exact rational signature (ncsig), one common dimension sized for the difference
Reports time and peak memory (tracemalloc) and checks that all three agree. Not a world-model
inference benchmark.
"""
import json
import random
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arc3.algebra import ncsig as N  # noqa: E402


def power(e, k):
    out = e
    for _ in range(k - 1):
        out = N.mul(out, e)
    return out


def measure(fn):
    tracemalloc.start()
    t = time.perf_counter()
    r = fn()
    dt = time.perf_counter() - t
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return r, dt * 1000, peak / 1024


def compare(e1, e2, n):
    out = {}
    diff = N.sub(e1, e2)
    out["expand"] = measure(lambda: N.expand(e1) == N.expand(e2))
    out["matrix"] = measure(lambda: N.random_matrix_test(diff, n, max(1, N.degree_of(diff))))
    out["exact"] = measure(lambda: N.equal(e1, e2, n))
    return out


def main():
    A, B = N.var(1), N.var(2)
    rows = []
    for k in (4, 8, 12, 16):
        h = k // 2
        e1 = power(N.add(A, B), k)
        e2 = N.mul(power(N.add(A, B), h), power(N.add(A, B), k - h))   # equal, different shape
        r = compare(e1, e2, 2)
        rows.append({"workload": f"(A+B)^{k} vs split", "equal": r["expand"][0],
                     **{f"{m}_ms": round(v[1], 3) for m, v in r.items()},
                     **{f"{m}_kib": round(v[2], 1) for m, v in r.items()},
                     "agree": len({v[0] for v in r.values()}) == 1})
    rng = random.Random(3)
    agg = {m: [0.0, 0.0] for m in ("expand", "matrix", "exact")}
    agree = 0
    for i in range(60):
        e1 = N.random_formula(rng, 3, rng.randint(6, 14))
        e2 = e1 if i % 3 == 0 else N.random_formula(rng, 3, rng.randint(6, 14))
        if i % 3 == 0:
            e2 = N.add(N.sub(e1, e1), e1)                                  # equal, different shape
        r = compare(e1, e2, 3)
        agree += len({v[0] for v in r.values()}) == 1
        for m, v in r.items():
            agg[m][0] += v[1]
            agg[m][1] = max(agg[m][1], v[2])
    rows.append({"workload": "60 random formula pairs (size 6-14, n=3)", "agree": f"{agree}/60",
                 **{f"{m}_ms_total": round(v[0], 2) for m, v in agg.items()},
                 **{f"{m}_kib_peak": round(v[1], 1) for m, v in agg.items()}})
    for r in rows:
        print(r)
    out = Path(__file__).resolve().parents[1] / "runs" / "b3" / "dedup_bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
