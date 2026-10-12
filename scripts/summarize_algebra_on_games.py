"""Summarize runs/algebra_on_games/{results.json, bfs_zm.json} into per-game model-class labels.

Click mechanics (level 1, games with ACTION6):
  xor_toggle    superposition, commutation and involution all hold on every tested pair/button
                -> gf2_solve's assumptions hold
  partial_xor   commutation holds, superposition holds on some pairs but not all
  inverse_pair  two buttons cancel each other (a then b = identity); counts follow fit_guarded
  last_click    commutation fails (the frame depends on click order, e.g. a selection highlight)
  none          no click changes the frame (outside the step counter)
Movement tracks (keys 1-4), from the raw mover coordinates:
  steady(+wall) constant step on >= 2 moves, optionally held at a wall -> fit_guarded exact
  single_jump   one change then constant (slope not identifiable)
  wrap          constant step with periodic resets (modular; fit_guarded correctly refuses)
  short_track / no_effect / no_rigid_mover / irregular
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "runs" / "algebra_on_games"


def click_class(c):
    if not c or c.get("effective", 0) == 0:
        return "none"
    if c.get("effective", 0) < 2:
        return "single_button"
    sup, comm = c.get("superposition_rate", 0), c.get("commutation_rate", 0)
    if comm < 1.0:
        return "last_click" if comm == 0 else "order_dependent"
    if sup == 1.0 and c.get("involution_rate", 0) == 1.0:
        return "xor_toggle"
    if c.get("inverse_pair_rate", 0) == 1.0:
        return "inverse_pair"
    if sup > 0:
        return "partial_xor"
    return "commuting_other"


def track_class(m):
    if "kind" in m:
        return m["kind"]
    ys, xs = m.get("ys"), m.get("xs")
    if ys is None or len(ys) < 4:
        return "short_track"
    out = []
    for s in (ys, xs):
        d = [b - a for a, b in zip(s, s[1:])]
        nz = [x for x in d if x != 0]
        if not nz:
            out.append("still")
            continue
        steps = collections.Counter(nz)
        main, cnt = steps.most_common(1)[0]
        if len(nz) == 1:
            out.append("single_jump")
        elif len(steps) == 1:
            first_zero = next((i for i, x in enumerate(d) if x == 0), None)
            walled = first_zero is not None and all(x == 0 for x in d[first_zero:])
            out.append("steady+wall" if walled else ("steady" if first_zero is None else "irregular"))
        elif len(steps) == 2 and all(x == main or x * main < 0 for x in nz) and cnt >= 2:
            out.append("wrap")
        else:
            out.append("irregular")
    moving = [o for o in out if o != "still"]
    return moving[0] if len(moving) == 1 else ("+".join(moving) if moving else "still")


def main():
    res = json.loads((D / "results.json").read_text())
    bz_path = D / "bfs_zm.json"
    bz = json.loads(bz_path.read_text()) if bz_path.exists() else {}
    click_tally, track_tally = collections.Counter(), collections.Counter()
    gf2_tot = collections.Counter()
    fit_ok = collections.Counter()
    rows = []
    for gid, r in sorted(res.items()):
        c = r.get("click")
        cc = click_class(c) if c is not None else "-"
        if c is not None:
            click_tally[cc] += 1
            g = c.get("gf2") or {}
            for k in ("targets", "ok", "rejected", "false_plan"):
                gf2_tot[(cc, k)] += g.get(k, 0)
        tracks = []
        for a, m in sorted(r.get("move", {}).items()):
            t = track_class(m)
            track_tally[t] += 1
            tracks.append(t)
            if t in ("steady", "steady+wall"):
                fit_ok[(t, bool(m.get("explained")))] += 1
            if t == "wrap":
                fit_ok[("wrap", bool(m.get("explained")))] += 1
        b = bz.get(gid, {}).get("bfs", {})
        z = bz.get(gid, {}).get("zm", [])
        b = dict(b, markov=bz.get(gid, {}).get("markov")) if b else b
        rows.append((gid, cc, c, tracks, b, z))
    print(f"{'game':6s} {'click class':16s} {'btn':>3s} {'gf2 ok/rej/false (of)':>22s} | {'movement (dirs 1-4)':44s} | bfs plan/baseline  | zm (period ok/targets)")
    for gid, cc, c, tracks, b, z in rows:
        g = (c or {}).get("gf2") or {}
        gtxt = f"{g.get('ok', '-')}/{g.get('rejected', '-')}/{g.get('false_plan', '-')} ({g.get('targets', '-')})" if g else "-"
        if b:
            btxt = (f"{b['plan_len']}/{b['baseline']} {'OK' if b.get('replay_completes_level') else 'REPLAY-FAIL'}"
                    if "plan_len" in b else f"none/{b['baseline']} ({b['nodes']}n{', exh' if b['exhausted'] else ''})")
        else:
            btxt = "-"
        ztxt = ", ".join(f"Z{x['period']} {x['replay_ok']}/{x['targets']}" for x in z) or "-"
        mk = (b or {}).get("markov")
        if mk:
            ztxt += f" | hidden-state conflicts {mk['conflicts']}/{mk['duplicate_histories_checked']}"
        print(f"{gid[:4]:6s} {cc:16s} {str((c or {}).get('effective', '-')):>3s} {gtxt:>22s} | {', '.join(tracks):44s} | {btxt:18s} | {ztxt}")
    print("\nclick classes:", dict(click_tally))
    print("movement tracks:", dict(track_tally))
    print("fit_guarded exact on (class, explained):", dict(fit_ok))
    print("gf2 by click class:", {k: v for k, v in sorted(gf2_tot.items())})
    solved = [b for *_, b, _ in rows if b and "plan_len" in b]
    if solved:
        print(f"bfs: {len(solved)}/{sum(1 for *_, b, _ in rows if b)} games solved, "
              f"replay ok {sum(b['replay_completes_level'] for b in solved)}/{len(solved)}, "
              f"plan/baseline median {sorted(b['plan_len'] / b['baseline'] for b in solved)[len(solved) // 2]:.2f}")


if __name__ == "__main__":
    sys.exit(main())
