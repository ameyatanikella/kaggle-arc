"""Decompose B0's (Franzen v3) public-game losses from its notebook log.

Input: text output of the notebook's benchmark cell (lines starting with
'[finished]'). Output: per-run score recomputed with the official formula,
loss from incomplete levels vs inefficiency, efficiency distribution.
"""
import re, sys, json
import numpy as np

def parse(path):
    rows = []
    for l in open(path):
        if not l.startswith('[finished]'):
            continue
        m = re.match(r"\[finished\] (\S+) state=(\S+) level=(\d+)/(\d+) score=([\d.]+) actions=(\d+) tokens=(\d+) per-level=(\S+)", l)
        g, st, lv, n, sc, act, tok, pl = m.groups()
        per = [tuple(map(int, x.split('/'))) for x in pl.split(',')]
        rows.append(dict(game=g, state=st, levels=int(lv), n=int(n), reported=float(sc), actions=int(act),
                         tokens=int(tok), per=per))
    return rows

def decompose(rows):
    out = []
    for r in rows:
        n = r['n']; W = n * (n + 1) / 2
        comp = n if r['state'] == 'won' else r['levels']
        s = maxs = 0.0
        ratios = []
        for i, (a, b) in enumerate(r['per'][:comp]):
            ls = min(115.0, 100.0 * (b / a) ** 2) if a > 0 else 0.0
            s += (i + 1) * ls; maxs += (i + 1) * 100.0
            ratios.append(b / a if a else 0.0)
        score = min(s / W, maxs / W)
        out.append(dict(game=r['game'], score=score, reported=r['reported'], incomplete_loss=100 - maxs / W,
                        inefficiency_loss=maxs / W - score, ratios=ratios, tokens=r['tokens'], won=r['state'] == 'won'))
    return out

if __name__ == '__main__':
    rows = decompose(parse(sys.argv[1]))
    ratios = np.array([x for r in rows for x in r['ratios']])
    res = {
        'runs': len(rows),
        'mean_score': float(np.mean([r['score'] for r in rows])),
        'max_abs_diff_vs_reported': float(max(abs(r['score'] - r['reported']) for r in rows)),
        'mean_incomplete_loss': float(np.mean([r['incomplete_loss'] for r in rows])),
        'mean_inefficiency_loss': float(np.mean([r['inefficiency_loss'] for r in rows])),
        'completed_levels': int(len(ratios)),
        'median_human_over_agent': float(np.median(ratios)),
        'frac_levels_at_or_under_baseline': float(np.mean(ratios >= 1)),
        'frac_levels_ratio_ge_1.072_(capped)': float(np.mean(ratios >= 1.0724)),
        'won_runs': int(sum(r['won'] for r in rows)),
        'mean_tokens_per_run': float(np.mean([r['tokens'] for r in rows])),
    }
    print(json.dumps(res, indent=1))
