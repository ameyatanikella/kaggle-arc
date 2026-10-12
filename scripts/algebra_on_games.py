"""Test the algebra sandbox tools against the REAL public games (level 1, offline engine).

This is analysis tooling: it uses fresh game instances freely (impossible in competition
play) to measure how often real mechanics fall inside each tool's model class, and whether
the solvers then act correctly on the real engine. Nothing here feeds into the agent.

Experiments
  A. Click algebra (games with ACTION6):
     single clicks on object centres -> effect masks; pairs (both orders), double clicks.
     * superposition: does frame(a,b) equal the XOR-superposition of single effects?
     * commutation: frame(a,b) == frame(b,a)?   * involution: frame(a,a) == start?
     * fit_boolean on per-pixel (d00,d10,d01,d11): pixels needing an interaction term.
     * GF(2) planning: random reachable target (random click subset S) -> gf2_solve on
       single-click effects -> replay presses on the real engine -> exact frame match?
     * held-out triple prediction: fit_boolean (degree <= 2) on 7 of the 8 subsets of a button
       triple, predict the 8th; XOR superposition as the reference predictor.
  B. Guarded movement (games with ACTION1-4): repeat each direction, track the mover,
     fit_guarded on each coordinate; exact explanation rate, step sizes, wall intervals.

usage: python scripts/algebra_on_games.py [--games all] [--out runs/algebra_on_games]
"""
import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kaggle" / "b2"))
os.environ["OPERATION_MODE"] = "offline"
os.environ.pop("ONLY_RESET_LEVELS", None)

import arc_agi  # noqa: E402
from arcengine import GameAction  # noqa: E402

import sandbox_tools as T  # noqa: E402
from arc3.perception import background_color, components  # noqa: E402
from arc3.runner import ENV_DIR, list_games  # noqa: E402

EDGE = 3


def quiet():
    lg = logging.getLogger("algebra_on_games")
    lg.handlers.clear()
    lg.addHandler(logging.NullHandler())
    lg.propagate = False
    logging.getLogger("arc_agi.scorecard").setLevel(logging.ERROR)
    return lg


class Game:
    def __init__(self, arc, gid):
        self.arc, self.gid = arc, gid
        w = arc.make(gid)
        self.f0 = np.array(w.observation_space.frame[-1], dtype=np.int16)
        self.avail = tuple(w.observation_space.available_actions)

    def play(self, acts):
        """Fresh instance; acts = list of ('click', x, y) or ('key', n). Returns (frame, info)."""
        w = self.arc.make(self.gid)
        info = {"level_up": False, "game_over": False}
        f = self.f0
        for a in acts:
            if a[0] == "click":
                r = w.step(GameAction.ACTION6, data={"x": int(a[1]), "y": int(a[2])})
            else:
                r = w.step(GameAction.from_id(a[1]))
            if r is None:
                info["error"] = True
                break
            if r.frame:
                f = np.array(r.frame[-1], dtype=np.int16)
            st = r.state.name if hasattr(r.state, "name") else str(r.state)
            if r.levels_completed > 0:
                info["level_up"] = True
            if st == "GAME_OVER":
                info["game_over"] = True
                break
        return f, info


def gf2_rank(A, n):
    rows = [sum(int(v) << j for j, v in enumerate(r)) for r in A]
    rank = 0
    for j in range(n):
        piv = next((i for i in range(rank, len(rows)) if rows[i] >> j & 1), None)
        if piv is None:
            continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        for i in range(len(rows)):
            if i != rank and rows[i] >> j & 1:
                rows[i] ^= rows[rank]
        rank += 1
    return rank


def interior(shape):
    m = np.zeros(shape, bool)
    m[EDGE:-EDGE, EDGE:-EDGE] = True
    return m


# ------------------------------------------------------------------ A. click algebra
def click_algebra(g, rng, max_clicks=120, n_pairs=40, n_targets=25, kmax=8):
    bg = background_color(g.f0)
    comps = list(components(g.f0, skip_colors=(bg,)))
    comps.sort(key=lambda c: (c.size, c.y0, c.x0))
    cands, seen = [], set()
    for c in comps:
        if (c.cy, c.cx) not in seen:
            seen.add((c.cy, c.cx))
            cands.append((c.cx, c.cy))
        if len(cands) >= max_clicks:
            break
    single = {}
    excluded = {"level_up": 0, "game_over": 0}
    for (x, y) in cands:
        f, info = g.play([("click", x, y)])
        if info["level_up"] or info["game_over"]:
            excluded["level_up" if info["level_up"] else "game_over"] += 1
            continue
        single[(x, y)] = f
    # --- step-counter pixels (must not be judged as mechanics) ---
    # (i) clicks on empty background: in some games they advance the counter and nothing else.
    counter = np.zeros(g.f0.shape, bool)
    bgy, bgx = np.nonzero((g.f0 == bg) & interior(g.f0.shape))
    if len(bgy):
        j = len(bgy) // 2
        for k in range(1, kmax + 1):
            fk, info = g.play([("click", bgx[j], bgy[j])] * k)
            counter |= fk != g.f0
            if info["game_over"]:
                break
    diffs = {k: (f != g.f0) for k, f in single.items()}
    eff = [k for k, d in diffs.items() if (d & ~counter).sum() > 4]
    res = {"candidates": len(cands), "effective": len(eff), "excluded": excluded}
    if not eff:
        return res
    # k-click series per button (reused below)
    series = {}
    for a in eff:
        fr = [g.f0]
        for k in range(1, kmax + 1):
            fk, info = g.play([("click", *a)] * k)
            if info["game_over"] or info["level_up"]:
                break
            fr.append(fk)
        series[a] = fr
    # (ii) with >= 3 buttons: a pixel changed after k clicks by EVERY button, for some k, is a
    # counter pixel (a step bar depends only on the number of actions). With fewer buttons this
    # cannot be told apart from a shared mechanic, so the edge band is excluded instead.
    if len(eff) >= 3:
        method = "null_clicks+intersection"
        for k in range(1, kmax + 1):
            ds = [(series[a][k] != g.f0) for a in eff if len(series[a]) > k]
            if len(ds) == len(eff):
                counter |= np.logical_and.reduce(ds)
    else:
        method = "null_clicks+edge_band"
        counter |= ~interior(g.f0.shape)
    mask = ~counter
    eff = [a for a in eff if (diffs[a] & mask).sum() > 0]
    res.update({"effective": len(eff), "counter_pixels": int(counter.sum()), "counter_method": method})
    if not eff:
        return res
    # --- repeated clicks: cyclic order (Z_m class) and guarded-affine colour counts ---
    orders, guarded = [], []
    for a in eff:
        fr = series[a]
        order = next((k for k in range(2, len(fr)) if np.array_equal(fr[k][mask], g.f0[mask])), None)
        orders.append(order)
        if len(fr) >= 4:
            cols = np.unique(np.stack(fr))
            counts = [[int(((f == c) & mask).sum()) for f in fr] for c in cols]
            changing = [cs for cs in counts if len(set(cs)) > 1]
            fits = [T.fit_guarded(list(range(len(cs))), cs) for cs in changing]
            guarded.append({"len": len(fr) - 1, "changing_colours": len(changing),
                            "explained": all(f is not None for f in fits),
                            "kinds": sorted({f["kind"] for f in fits if f})})
    res.update({"click_order": orders, "order_rate": sum(o is not None for o in orders) / len(orders),
                "involution_rate": sum(o == 2 for o in orders) / len(orders),
                "idempotent_rate": sum(len(series[a]) > 2 and np.array_equal(series[a][2][mask], series[a][1][mask])
                                       for a in eff) / len(eff),
                "guarded_counts": guarded,
                "guarded_rate": (sum(x["explained"] for x in guarded) / len(guarded)) if guarded else None})
    if len(eff) < 2:
        return res
    # --- pairs: superposition (XOR of single effects), commutation, inverse pairs ---
    pairs = [(a, b) for i, a in enumerate(eff) for b in eff[i + 1:]]
    rng.shuffle(pairs)
    pairs = pairs[:n_pairs]
    sup_ok = comm_ok = n_ok = inverse_ok = 0
    inter_pixels = tot_pixels = 0
    for a, b in pairs:
        fab, ia = g.play([("click", *a), ("click", *b)])
        fba, ib = g.play([("click", *b), ("click", *a)])
        if ia["level_up"] or ia["game_over"] or ib["level_up"] or ib["game_over"]:
            continue
        n_ok += 1
        da, db = diffs[a] & mask, diffs[b] & mask
        pred = g.f0.copy()
        only_a, only_b = da & ~db, db & ~da
        pred[only_a] = single[a][only_a]
        pred[only_b] = single[b][only_b]          # pixels in both: XOR toggle -> back to start
        sup_ok += int(np.array_equal(pred[mask], fab[mask]))
        comm_ok += int(np.array_equal(fab[mask], fba[mask]))
        inverse_ok += int(np.array_equal(fab[mask], g.f0[mask]))
        # fit_boolean on the changed-indicator per pixel over (d00, d10, d01, d11)
        touched = (da | db | ((fab != g.f0) & mask))
        ys, xs = np.nonzero(touched)
        for yy, xx in list(zip(ys, xs))[:200]:
            vals = [0, int(da[yy, xx]), int(db[yy, xx]), int(fab[yy, xx] != g.f0[yy, xx])]
            fit = T.fit_boolean([[0, 0], [1, 0], [0, 1], [1, 1]], vals, time_limit=1.0)
            tot_pixels += 1
            if fit and fit["terms"] and (0, 1) in fit["terms"]:
                inter_pixels += 1
    res.update({"pairs_tested": n_ok, "superposition_rate": sup_ok / max(1, n_ok),
                "commutation_rate": comm_ok / max(1, n_ok), "inverse_pair_rate": inverse_ok / max(1, n_ok),
                "pixels_needing_interaction_term": inter_pixels / max(1, tot_pixels)})
    # --- held-out prediction (fit_boolean's low-degree assumption): for a button triple, play
    # all 8 subsets (clicks in a fixed order), fit each touched pixel's changed-indicator on the 7
    # subsets other than {a,b,c}, predict {a,b,c}, compare with the engine. Reference predictor:
    # XOR superposition (GF(2) model). Only pixels whose prediction is pinned are scored.
    if len(eff) >= 3:
        subsets = [tuple((i >> j) & 1 for j in range(3)) for i in range(8)]
        trip_px = trip_ok = trip_xor_ok = trip_nofit = trip_unpinned = frames_ok = frames = 0
        triples = set()
        rng3 = random.Random(1)                     # own stream: does not shift the GF(2) draws
        for _ in range(200):
            triples.add(tuple(sorted(rng3.sample(range(len(eff)), 3))))
            if len(triples) >= 12:
                break
        for tri in sorted(triples):
            fr = {}
            dead = False
            for sub in subsets:
                f, info = g.play([("click", *eff[tri[j]]) for j in range(3) if sub[j]])
                if info["level_up"] or info["game_over"]:
                    dead = True
                    break
                fr[sub] = (f != g.f0) & mask
            if dead:
                continue
            touched = np.logical_or.reduce(list(fr.values()))
            ys, xs = np.nonzero(touched)
            frame_ok = True
            obs = [s_ for s_ in subsets if s_ != (1, 1, 1)]
            for yy, xx in list(zip(ys, xs))[:150]:
                vals = [int(fr[s_][yy, xx]) for s_ in obs]
                truth = int(fr[(1, 1, 1)][yy, xx])
                xor_pred = int(fr[(1, 0, 0)][yy, xx]) ^ int(fr[(0, 1, 0)][yy, xx]) ^ int(fr[(0, 0, 1)][yy, xx])
                trip_px += 1
                trip_xor_ok += int(xor_pred == truth)
                fit = T.fit_boolean([list(s_) for s_ in obs], vals, time_limit=1.0)
                if not fit or fit["terms"] is None:
                    trip_nofit += 1
                    frame_ok = False
                    continue
                if fit["alternatives"] or any(len(m_) == 3 for m_ in fit["unidentified"]):
                    trip_unpinned += 1      # (cannot happen at degree <= 2 with all 7 rows; kept as a guard)
                pred = sum(fit["terms"].values())          # every monomial is 1 at (1,1,1)
                ok = int(pred == truth)
                trip_ok += ok
                frame_ok &= bool(ok)
            frames += 1
            frames_ok += int(frame_ok)
        res["triple_prediction"] = {"triples": frames, "frames_exact": frames_ok, "pixels": trip_px,
                                    "deg2_pixel_acc": trip_ok / max(1, trip_px),
                                    "xor_pixel_acc": trip_xor_ok / max(1, trip_px),
                                    "no_fit_pixels": trip_nofit, "unpinned_pixels": trip_unpinned}
    # --- GF(2) planning to targets generated by real click subsets ---
    # Outcomes: plan reaches target on the engine (ok); tool returns None = it REJECTS the toggle
    # model (honest); tool returns a plan that fails on the engine = FALSE CONFIDENCE.
    touched_any = np.zeros_like(mask)
    for k in eff:
        touched_any |= diffs[k] & mask
    pix = list(zip(*np.nonzero(touched_any)))
    A = [[int(diffs[k][p]) for k in eff] for p in pix]
    out = {"ok": 0, "rejected": 0, "false_plan": 0, "nontrivial": 0, "nontrivial_ok": 0, "target_is_start": 0}
    tried = 0
    for _ in range(n_targets):
        S = rng.sample(eff, rng.randint(2, min(4, len(eff))))
        tgt, info = g.play([("click", *k) for k in S])
        if info["level_up"] or info["game_over"]:
            continue
        if np.array_equal(tgt[mask], g.f0[mask]):
            out["target_is_start"] += 1          # e.g. inverse buttons cancel: no planning content
            continue
        tried += 1
        b = [int(tgt[p] != g.f0[p]) for p in pix]
        outside = bool(((tgt != g.f0) & mask & ~touched_any).any())
        sol = None if outside else T.gf2_solve(A, b, time_limit=2.0)
        if sol is None:
            out["rejected"] += 1
            continue
        presses = [eff[j] for j in sol["presses"]]
        got, _ = g.play([("click", *k) for k in presses])
        nontriv = set(presses) != set(S)          # replaying the generator's own clicks is trivial
        out["nontrivial"] += int(nontriv)
        if np.array_equal(got[mask], tgt[mask]):
            out["ok"] += 1
            out["nontrivial_ok"] += int(nontriv)
        else:
            out["false_plan"] += 1
    res["gf2"] = {"targets": tried, "buttons": len(eff), "nullity": len(eff) - gf2_rank(A, len(eff)), **out}
    return res


# ------------------------------------------------------------------ B. guarded movement
def shape_sig(grid, c):
    sub = grid[c.y0:c.y1 + 1, c.x0:c.x1 + 1] == c.color
    return (c.color, sub.shape, sub.tobytes())


def movement(g, n_steps=12):
    out = {}
    inside = interior(g.f0.shape)
    for a in (1, 2, 3, 4):
        if a not in g.avail:
            continue
        frames = [g.f0]
        for k in range(1, n_steps + 1):
            f, info = g.play([("key", a)] * k)
            if info["game_over"] or info["level_up"]:
                break
            frames.append(f)
        if len(frames) < 4:
            out[a] = {"kind": "too_short"}
            continue
        bg = background_color(g.f0)
        d1 = (frames[0] != frames[1]) & inside
        if not d1.any():
            out[a] = {"kind": "no_effect"}
            continue
        # mover: a component of frame0 touching the diff whose exact shape reappears elsewhere in frame1
        c0 = [c for c in components(frames[0], skip_colors=(bg,)) if d1[c.y0:c.y1 + 1, c.x0:c.x1 + 1].any()]
        mover = None
        for c in sorted(c0, key=lambda c: -c.size):
            sig = shape_sig(frames[0], c)
            m1 = [d for d in components(frames[1], skip_colors=(bg,)) if shape_sig(frames[1], d) == sig
                  and (d.y0, d.x0) != (c.y0, c.x0)]
            if m1:
                mover = (sig, c)
                break
        if mover is None:
            out[a] = {"kind": "no_rigid_mover"}
            continue
        sig, c = mover
        ys, xs = [c.y0], [c.x0]
        for f in frames[1:]:
            cand = [d for d in components(f, skip_colors=(bg,)) if shape_sig(f, d) == sig]
            if not cand:
                break
            d = min(cand, key=lambda d: abs(d.y0 - ys[-1]) + abs(d.x0 - xs[-1]))
            ys.append(d.y0)
            xs.append(d.x0)
        t = list(range(len(ys)))
        fy, fx = T.fit_guarded(t, ys), T.fit_guarded(t, xs)
        out[a] = {"steps": len(ys) - 1, "ys": ys, "xs": xs, "y": fy, "x": fx,
                  "explained": fy is not None and fx is not None}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", default="all")
    p.add_argument("--out", default=str(ROOT / "runs" / "algebra_on_games"))
    a = p.parse_args()
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(ENV_DIR), logger=quiet())
    games = list_games() if a.games == "all" else [g for g in list_games() if g.split("-")[0] in a.games.split(",")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for gid in games:
        rng = random.Random(0)
        g = Game(arc, gid)
        r = {"available": g.avail}
        if 6 in g.avail:
            r["click"] = click_algebra(g, rng)
        if any(x in g.avail for x in (1, 2, 3, 4)):
            r["move"] = movement(g)
        results[gid] = r
        c = r.get("click", {})
        mv = r.get("move", {})
        expl = sum(1 for v in mv.values() if v.get("explained"))
        nan = float("nan")
        gr = c.get("guarded_rate")
        gf = c.get("gf2", {})
        print(f"{gid:16s} click: eff={c.get('effective', '-'):>3} sup={c.get('superposition_rate', nan):.2f} "
              f"comm={c.get('commutation_rate', nan):.2f} invol={c.get('involution_rate', nan):.2f} "
              f"invpair={c.get('inverse_pair_rate', nan):.2f} order={c.get('order_rate', nan):.2f} "
              f"guarded={nan if gr is None else gr:.2f} "
              f"gf2 ok/rej/false={gf.get('ok', '-')}/{gf.get('rejected', '-')}/{gf.get('false_plan', '-')} "
              f"of {gf.get('targets', '-')} null={gf.get('nullity', '-')} | "
              f"move explained {expl}/{len(mv)}", flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main()
