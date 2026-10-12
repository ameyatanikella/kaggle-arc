#!/usr/bin/env bash
# One-shot readiness check once the user has placed a token at .kaggle/access_token.
# Read-only: lists, pulls and quota queries. It pushes or submits nothing.
set -u
cd "$(dirname "$0")/.."
TOK=.kaggle/access_token
if [ ! -s "$TOK" ]; then echo "NO TOKEN at $TOK"; exit 1; fi
export KAGGLE_API_TOKEN="$(cat $TOK)"
K=.venv/bin/kaggle
echo "== whoami / quota"; $K quota 2>&1 | head -20
echo "== competition visible + entered?"; $K competitions list -s "arc-prize-2026" 2>&1 | head -8
echo "== my submissions (needs entry)"; $K competitions submissions arc-prize-2026-arc-agi-3 2>&1 | head -12
mkdir -p data/kaggle_pull/franzen
echo "== pull Franzen M2 notebook + metadata"; $K kernels pull dfranzen/arc-agi-3-milestone-2-solution -p data/kaggle_pull/franzen -m 2>&1 | tail -3
ls -la data/kaggle_pull/franzen
python3 - <<'EOF'
import json, hashlib, glob
ours = json.load(open('kaggle/b0/franzen_m2_v3_original.ipynb'))
f = [p for p in glob.glob('data/kaggle_pull/franzen/*.ipynb')]
if not f:
    print('no pulled notebook'); raise SystemExit
theirs = json.load(open(f[0]))
src = lambda nb: [''.join(c['source']) for c in nb['cells']]
print('cells ours/theirs:', len(ours['cells']), len(theirs['cells']))
print('identical cell sources:', src(ours) == src(theirs))
for i, (a, b) in enumerate(zip(src(ours), src(theirs))):
    if a != b: print('  differs: cell', i, hashlib.sha256(a.encode()).hexdigest()[:8], hashlib.sha256(b.encode()).hexdigest()[:8])
m = glob.glob('data/kaggle_pull/franzen/kernel-metadata.json')
if m: print('pulled metadata:', open(m[0]).read())
EOF
