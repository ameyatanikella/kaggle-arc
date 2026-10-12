"""Offline replay of cross-game compute schedulers on logged harness runs.

usage: python scripts/sched_replay.py <benchmark.json> [--budget-per-game 170000] [--sims 200]

Model (assumptions stated, not hidden):
  * A game's progress depends only on the generated tokens it receives
    (wall-clock effects, context trims and server contention are ignored).
  * Progress is Markov at level boundaries: the time to clear level k+1 after
    reaching it is drawn from that game's logged passes that reached level k.
  * Right-censoring is respected: a logged attempt that ended unfinished after d
    tokens only says "not cleared within d". When a simulated game exceeds d, a
    new attempt is drawn *conditioned on lasting longer than d*. If no logged
    attempt lasted longer, the game makes no further progress (pessimistic).
  * Scores use the official per-level formula with the logged action counts.
Results are scheduler comparisons under this model, not predictions of LB score.
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arc3.scoring import game_score  # noqa: E402

import json  # noqa: E402

TOKENS_PER_ACTION = 900.0  # proxy to turn level tokens into an action count for priority C


def load_segments(path):
    b = json.loads(Path(path).read_text())
    games = {}
    for r in b["game_runs"]:
        hist = r.get("history") or []
        cum, toks = 0, []
        for e in hist:
            cum += int(e.get("generated_tokens") or 0)
            toks.append(cum)
        note = str(r.get("solver_note") or "")
        T = cum
        if note.startswith("tokens="):
            try:
                T = max(T, int(note.split("=")[1].split()[0]))
            except ValueError:
                pass
        apl = [int(x) for x in (r.get("actions_per_level") or [])]
        k_done = int(r.get("levels_completed") or 0)
        g = games.setdefault(r["game_id"], {"n": int(r.get("number_of_levels") or len(apl)),
                                            "base": [int(x) for x in (r.get("base_actions_per_level") or [])],
                                            "segs": {}, "passes": 0})
        g["passes"] += 1
        idx, start = 0, 0
        for k in range(g["n"]):
            if k < k_done:
                idx += apl[k]
                t_end = toks[min(idx, len(toks)) - 1] if toks and idx > 0 else start
                g["segs"].setdefault(k, []).append((max(1, t_end - start), True, apl[k]))
                start = t_end
            else:
                if r.get("state") != "won":
                    g["segs"].setdefault(k, []).append((max(1, T - start), False, 0))
                break
    return games


class GameSim:
    def __init__(self, g, rng):
        self.g, self.rng = g, rng
        self.k = 0            # levels completed
        self.tau = 0.0        # tokens on current level
        self.actions = []     # actions of completed levels
        self.stuck = False
        self.seg = self._draw(0.0)

    def _draw(self, longer_than):
        cands = [s for s in self.g["segs"].get(self.k, []) if s[0] > longer_than]
        if not cands:
            return None
        return self.rng.choice(cands)

    @property
    def done(self):
        return self.k >= self.g["n"] or self.seg is None

    def give(self, q):
        while q > 0 and not self.done:
            d, completed, a = self.seg
            need = d - self.tau
            if q < need:
                self.tau += q
                return
            q -= need
            if completed:
                self.actions.append(a)
                self.k += 1
                self.tau = 0.0
                self.seg = self._draw(0.0) if self.k < self.g["n"] else None
            else:
                self.tau = d
                self.seg = self._draw(d)   # conditioned on lasting longer than d
                if self.seg is None:
                    return

    def score(self):
        return game_score(self.g["base"], self.actions)


def franzen_priority(sim, frac_remaining):
    level = sim.k + 1
    N = min(10, max(6, sim.g["n"]))
    a, t = sim.tau / TOKENS_PER_ACTION, sim.tau
    A = level * (25.0 / (25.0 + a)) ** 2 * 55.0 / (N * (N + 1) / 2)
    remaining = N - min(N, level)
    B = {0: 0.0, 1: 5.0, 2: 7.0}.get(remaining, 8.0)
    if frac_remaining < 0.2:
        B *= frac_remaining / 0.2
    C = 0.25 * 2 ** (-(a / 115.0) ** 2) + 0.75 * 2 ** (-(t / 62000.0) ** 2)
    return (A + B) * C


def run_policy(games, policy, budget_per_game, quantum, rng, **kw):
    sims = {name: GameSim(g, rng) for name, g in games.items()}
    names = sorted(sims)
    total = budget_per_game * len(names)
    if policy == "equal":
        for n in names:
            sims[n].give(budget_per_game)
        return sum(sims[n].score() for n in names) / len(names)
    spent = 0.0
    while spent < total:
        live = [n for n in names if not sims[n].done]
        if not live:
            break
        frac = 1 - spent / total
        if policy == "franzen":
            best = max(live, key=lambda n: franzen_priority(sims[n], frac))
        elif policy == "cap":          # equal share, but abandon a level after `cap` tokens; pool redistributed
            cap = kw.get("cap", 120000)
            live2 = [n for n in live if sims[n].tau < cap] or live
            best = min(live2, key=lambda n: (sims[n].tau, n))
        elif policy == "level_first":  # always feed the game with the most levels done, then least tokens on level
            best = max(live, key=lambda n: (sims[n].k, -sims[n].tau))
        else:
            raise ValueError(policy)
        sims[best].give(quantum)
        spent += quantum
    return sum(sims[n].score() for n in names) / len(names)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("benchmark")
    p.add_argument("--budget-per-game", type=float, default=170000)
    p.add_argument("--quantum", type=float, default=4000)
    p.add_argument("--sims", type=int, default=200)
    a = p.parse_args()
    games = load_segments(a.benchmark)
    print(f"{len(games)} games, passes={sum(g['passes'] for g in games.values())}, budget/game={a.budget_per_game:.0f}")
    for pol, kw in (("equal", {}), ("franzen", {}), ("cap", {"cap": 60000}), ("cap", {"cap": 120000}),
                    ("level_first", {})):
        rng = random.Random(0)
        vals = [run_policy(games, pol, a.budget_per_game, a.quantum, rng, **kw) for _ in range(a.sims)]
        m = sum(vals) / len(vals)
        sd = (sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
        tag = pol + (f"(cap={kw['cap']})" if kw else "")
        print(f"  {tag:18s} mean={m:6.2f}  sd={sd:5.2f}")


if __name__ == "__main__":
    main()
