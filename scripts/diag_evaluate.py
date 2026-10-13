"""Evaluate DIAG stage 2 (runs/diag/shadow.json) against the pre-registered criteria S1-S5."""
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
r = json.loads((ROOT / "runs" / "diag" / (sys.argv[1] if len(sys.argv) > 1 else "shadow.json")).read_text())

games = {g: v for g, v in r.items() if "M1" in v}
print(f"games with traces: {len(games)} of {len(r)}")

# S1: >= 1 checkable relation on >= 30% of eligible steps in >= 5 games
s1_games = [g[:4] for g, v in games.items() if v["M1"]["eligible_steps"] and v["M1"]["rate"] >= 0.30]
print("\nM1 encounter (checkable / eligible steps):")
for g, v in sorted(games.items()):
    m = v["M1"]
    print(f"  {g[:4]} {m['checkable_steps']:>3}/{m['eligible_steps']:<3} ({m['rate']:.2f})  comps={v['components']}  "
          f"real verdicts={v['real_verdicts']}")
S1 = len(s1_games) >= 5
print(f"S1: {len(s1_games)} games >= 30%: {s1_games} -> {'PASS' if S1 else 'FAIL'}")

rows = [x for v in games.values() for x in v["M2"]]
print(f"\nM2 trials: {len(rows)}  (components: {len({(x['comp']) for x in rows})}, fields: "
      f"{dict(collections.Counter(x['field'] for x in rows))})")


def rates(arm, ks):
    sel = [x for x in rows if x["k"] in ks]
    c = collections.Counter(x[arm]["outcome"] for x in sel)
    n = max(1, len(sel))
    return {k: c[k] / n for k in ("exact", "correctly_ambiguous", "confident_wrong", "ambiguous_wrong", "missed",
                                   "missed_undetectable", "inadequate")}, len(sel)


for ks, label in (((1,), "k=1"), ((2, 3), "k=2,3"), ((2,), "k=2"), ((3,), "k=3")):
    for arm in ("B", "C"):
        rt, n = rates(arm, ks)
        print(f"  {label:6s} {arm} n={n:4d} " + " ".join(f"{k}={v:.3f}" for k, v in rt.items()))
b1, _ = rates("B", (1,))
c1, _ = rates("C", (1,))
b23, _ = rates("B", (2, 3))
c23, _ = rates("C", (2, 3))
S2 = (c23["confident_wrong"] <= b23["confident_wrong"] - 0.10 and
      c23["exact"] + c23["correctly_ambiguous"] >= b23["exact"] + b23["correctly_ambiguous"] and
      c1["exact"] >= b1["exact"] - 0.02)
print(f"S2: CW(k=2,3) C={c23['confident_wrong']:.3f} vs B={b23['confident_wrong']:.3f} (need <= B-0.10); "
      f"exact+amb C={c23['exact'] + c23['correctly_ambiguous']:.3f} vs B={b23['exact'] + b23['correctly_ambiguous']:.3f}; "
      f"k=1 exact C={c1['exact']:.3f} vs B={b1['exact']:.3f} -> {'PASS' if S2 else 'FAIL'}")

for arm in ("B", "C"):
    tp = sum(x[arm]["tp"] for x in rows)
    fp = sum(x[arm]["fp"] for x in rows)
    tot = sum(x["k"] for x in rows)
    print(f"  {arm} precision={tp / max(1, tp + fp):.3f} recall={tp / max(1, tot):.3f}")

m3 = [x for v in games.values() for x in v["M3"]]
fa = {arm: sum(1 for x in m3 if x[arm] != "consistent") for arm in ("B", "C")}
S3 = fa["C"] <= fa["B"]
print(f"\nM3 clean components: {len(m3)}; false alarms B={fa['B']} C={fa['C']} -> S3 {'PASS' if S3 else 'FAIL'}")


def repair(arm):
    det = cor = n = 0
    for x in rows:
        rp = x[arm]["repair"]
        if rp:
            det += rp["determined"]
            cor += rp["correct"]
            n += rp["n"]
    return cor / max(1, det), det / max(1, n)


rb, db = repair("B")
rc, dc = repair("C")
S4 = rc >= rb - 0.02
print(f"M4 repaired next-observation accuracy (determined predictions): B={rb:.3f} (determined {db:.2f}) "
      f"C={rc:.3f} (determined {dc:.2f}) -> S4 {'PASS' if S4 else 'FAIL'}")

reo = [x["C"]["reobserve"] for x in rows if "reobserve" in x["C"]]
if reo:
    print(f"C reobservation suggestions: {len(reo)}; suggested obs was faulty in "
          f"{sum(r_['was_faulty'] for r_ in reo)}; outcomes after reobserving: "
          f"{dict(collections.Counter(r_['after'] for r_ in reo))}")

tC = [v["M5"]["C"]["median_ms"] for v in games.values() if v["M5"].get("C", {}).get("median_ms") is not None]
pC = [v["M5"]["C"]["p95_ms"] for v in games.values() if v["M5"].get("C", {}).get("p95_ms") is not None]
tN = [v["M5"]["Cnocache"]["median_ms"] for v in games.values() if v["M5"].get("Cnocache", {}).get("median_ms") is not None]
tB = [v["M5"]["B"]["median_ms"] for v in games.values() if v["M5"].get("B", {}).get("median_ms") is not None]
mem = [v["M5"]["cache"]["memory_bytes"] for v in games.values() if "cache" in v["M5"]]
S5 = bool(tC) and float(np.median(tC)) <= 50 and max(pC) <= 500 and max(mem) <= 50e6
print(f"\nM5 per-check ms (median over games of per-game medians): B={np.median(tB):.1f} C={np.median(tC):.1f} "
      f"C-nocache={np.median(tN):.1f}; worst per-game p95 C={max(pC):.1f}; max cache memory={max(mem) / 1e6:.2f} MB "
      f"-> S5 {'PASS' if S5 else 'FAIL'}")
fb = sum(1 for x in rows if x["C"]["fallback"])
print(f"C fallbacks to B: {fb}")
print(f"\nALL: S1={S1} S2={S2} S3={S3} S4={S4} S5={S5}")
