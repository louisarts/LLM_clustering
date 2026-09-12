"""Step 6 - PAID (~$5): the ASSIGNMENT stage.

Classifies N_ASSIGN sampled documents per benchmark against that benchmark's
codebook (thinking-off, one cheap call per document). The result is the judge's
reference labelling - a weak annotation of the sample - against which every
candidate clustering is scored for free in step 07.

Prints degeneracy diagnostics per benchmark as it finishes (share of the largest
category, rate of "fits none"), so a collapsed judge is visible immediately.

Usage:
  python 06_assign_documents.py            dry run: call count + cost
  python 06_assign_documents.py --run      spends
  python 06_assign_documents.py --run --workers 12 --bench banking77

Caching: every finished document is written to data/reference_labels/<bench>.csv
(saved every 100); re-running skips what is already assigned. Only one instance
may run at a time (lock file), since concurrent runs clobber the cache.
"""
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

RQ1 = Path(__file__).resolve().parents[2]
sys.path.append(str(Path(__file__).parent))

from judge import CRITERIA, classify_document                       # noqa: E402

CB = RQ1 / 'data' / 'rq1' / 'codebooks'
LAB = RQ1 / 'data' / 'rq1' / 'reference_labels'
LOCK = RQ1 / 'results' / 'rq1' / '.assign_running.lock'
LAB.mkdir(parents=True, exist_ok=True)
LOCK.parent.mkdir(parents=True, exist_ok=True)

N_ASSIGN = 1000        # documents per benchmark in the reference sample
SEED = 1               # distinct from the discovery seed
COST_PER_CALL = 0.0006  # codebook (~1-2k tokens) + clipped doc in, a few tokens out
REPORT_EVERY = 100

RUN = '--run' in sys.argv
WORKERS = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 8
ONLY = sys.argv[sys.argv.index('--bench') + 1] if '--bench' in sys.argv else None


def gateway_spend():
    try:
        base = os.environ.get('OPENAI_BASE_URL', '').rstrip('/')
        if base.endswith('/v1'):
            base = base[:-3].rstrip('/')
        req = urllib.request.Request(
            f'{base}/key/info',
            headers={'Authorization': f"Bearer {os.environ['OPENAI_API_KEY']}"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return float(json.load(r)['info']['spend'])
    except Exception:
        return None


def diagnostics(bench, df):
    """Degeneracy alarms, computed from the assignment distribution alone."""
    assigned = df[df.category.notna()]
    if not len(assigned):
        return
    none_rate = (assigned.category == 0).mean()
    real = assigned[assigned.category > 0]
    top_share = real.category.value_counts(normalize=True).iloc[0] if len(real) else 1.0
    n_used = real.category.nunique()
    line = (f'   diagnostics: {n_used} categories used | '
            f'largest holds {top_share:.0%} of docs | "fits none": {none_rate:.0%}')
    warn = []
    if top_share > 0.5:
        warn.append('one category is swallowing the corpus')
    if none_rate > 0.2:
        warn.append('codebook misses much of the corpus')
    print(line + (f'   !! WARNING: {"; ".join(warn)}' if warn else ''), flush=True)


benches = [ONLY] if ONLY else sorted(CRITERIA)
todo = {}
for bench in benches:
    cb_path = CB / f'{bench}.json'
    if not cb_path.exists():
        print(f'!! {bench}: no codebook - run 05_build_codebook.py first; skipping')
        continue
    n_docs = len(pd.read_csv(RQ1 / 'data' / 'rq1' / 'texts' / f'{bench}.csv'))
    n_sample = min(N_ASSIGN, n_docs)
    lab_path = LAB / f'{bench}.csv'
    done = set(pd.read_csv(lab_path).doc_idx) if lab_path.exists() else set()
    pending = n_sample - len(done)
    print(f'{bench:>15}: {n_sample} reference docs, {len(done)} assigned, {pending} pending')
    if pending > 0:
        todo[bench] = pending

n_calls = sum(todo.values())
print(f'\n~{n_calls} LLM calls, ~${n_calls * COST_PER_CALL:.2f}')
if not RUN:
    print('dry run only - rerun with --run to spend.')
    sys.exit(0)

# single-instance lock (concurrent runs clobber the per-bench cache files)
if LOCK.exists():
    try:
        other = int(LOCK.read_text().strip())
    except ValueError:
        other = None
    alive = False
    if other is not None:
        try:
            os.kill(other, 0); alive = True
        except ProcessLookupError:
            alive = False
        except PermissionError:
            alive = True
    if alive:
        print(f'!! another assignment run is active (pid {other}). Kill it or delete {LOCK}.')
        sys.exit(1)
    print('   stale lock reclaimed')
LOCK.write_text(str(os.getpid()))
import atexit                                                        # noqa: E402
atexit.register(lambda: LOCK.exists() and LOCK.unlink())

t0 = time.time()
spend0 = gateway_spend()
print('gateway spend at start: '
      + (f'${spend0:.4f}' if spend0 is not None else 'unavailable') + '\n', flush=True)
completed = 0

for bench in todo:
    spec = CRITERIA[bench]
    codebook = json.loads((CB / f'{bench}.json').read_text())['categories']
    texts = pd.read_csv(RQ1 / 'data' / 'rq1' / 'texts' / f'{bench}.csv')['text'] \
        .fillna('').astype(str).tolist()
    rng = np.random.RandomState(SEED)
    sample_idx = rng.choice(len(texts), size=min(N_ASSIGN, len(texts)), replace=False)
    lab_path = LAB / f'{bench}.csv'
    rows = pd.read_csv(lab_path).to_dict('records') if lab_path.exists() else []
    done = {r['doc_idx'] for r in rows}
    pending = [int(i) for i in sample_idx if int(i) not in done]
    print(f'== {bench}: {len(pending)} docs against {len(codebook)} categories', flush=True)

    def one(i):
        return i, classify_document(texts[i], codebook, spec)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for fut in as_completed({pool.submit(one, i): i for i in pending}):
            try:
                i, cat = fut.result()
            except Exception as e:
                print(f'   !! doc failed: {type(e).__name__}: {e}', flush=True)
                continue
            rows.append({'doc_idx': i, 'category': cat})
            completed += 1
            if completed % REPORT_EVERY == 0:
                pd.DataFrame(rows).to_csv(lab_path, index=False)
                el = time.time() - t0
                line = (f'   ---- {completed}/{n_calls} | elapsed {el/60:.0f}m | '
                        f'ETA {el/completed*(n_calls-completed)/60:.0f}m')
                now = gateway_spend()
                if now is not None and spend0 is not None:
                    per = (now - spend0) / completed
                    line += f' | spent ${now-spend0:.3f} (${per:.6f}/call, proj ${per*n_calls:.2f})'
                print(line, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(lab_path, index=False)
    print(f'   {len(df)} assigned -> {lab_path.name}')
    diagnostics(bench, df)

print('\ndone - next: 07_score_partitions.py (free)')
