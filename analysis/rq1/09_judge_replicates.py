#!/usr/bin/env python
"""Four additional fully independent judge runs (runs 2-5; the original is run 1).

Each run draws a FRESH discovery sample (new codebook, model still picks granularity),
a FRESH assignment sample (1,000 docs, per-document classification, same protocol as
run 1), then scores all factorial candidates against that run's reference labelling.

Everything cached per (run, benchmark) and resumable. Budget-guarded.

Outputs:
    data/replicates/run{r}/{bench}_codebook.json
    data/replicates/run{r}/{bench}_reference.csv
    results/judge_replicates.csv     run, bench, partition_id, judge, n_docs_used
    results/replicate_meta.csv       run, bench, codebook_categories, n_reference
"""
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

RQ1C = Path(__file__).resolve().parents[2]
import os
EMB_TAG = os.environ.get('EMB_TAG', 'instructor')
def _dir(base):  # embedder-tagged data dirs; 'instructor' keeps legacy paths
    return base if EMB_TAG == 'instructor' else f'{base}_{EMB_TAG}'

sys.path.insert(0, str(RQ1C / 'scripts'))

import judge  # noqa: E402  (the clean copy: 19 criteria)
from judge import CRITERIA, build_codebook, classify_document, score_partition  # noqa: E402

RUNS = [int(x) for x in os.environ.get('RUNS', '2,3,4,5').split(',')]
N_DISCOVERY, N_ASSIGN = 300, 1000
WORKERS = 12
BUDGET_STOP = 880.0

REP = RQ1C / 'data' / 'rq1' / 'replicates'
OUT_SCORES = RQ1C / 'results' / 'rq1' / ('judge_replicates.csv' if EMB_TAG == 'instructor' else f'judge_replicates_{EMB_TAG}.csv')
OUT_META = RQ1C / 'results' / 'rq1' / 'replicate_meta.csv'


def gateway_spend():
    try:
        base = os.environ['OPENAI_BASE_URL'].rstrip('/')
        if base.endswith('/v1'):
            base = base[:-3].rstrip('/')
        req = urllib.request.Request(
            f'{base}/key/info',
            headers={'Authorization': f"Bearer {os.environ['OPENAI_API_KEY']}"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return float(json.load(r)['info']['spend'])
    except Exception:
        return None


def guard():
    s = gateway_spend()
    if s is not None and s > BUDGET_STOP:
        raise RuntimeError(f'BUDGET STOP at ${s:.2f}')


def get_codebook(run, bench, texts):
    f = REP / f'run{run}' / f'{bench}_codebook.json'
    if f.exists():
        return json.loads(f.read_text())['categories']
    guard()
    spec = CRITERIA[bench]
    for attempt in range(3):                       # truncation retries, fresh sub-sample
        rng = np.random.RandomState(1000 * run + attempt)
        idx = rng.choice(len(texts), min(N_DISCOVERY, len(texts)), replace=False)
        try:
            cats = build_codebook([texts[i] for i in idx], spec)
            break
        except ValueError as e:
            print(f'  run{run} {bench}: codebook attempt {attempt + 1} failed ({e})',
                  flush=True)
    else:
        raise RuntimeError(f'run{run} {bench}: codebook failed 3 times')
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({'bench': bench, 'run': run,
                             'criterion': spec['criterion'],
                             'categories': cats}, indent=2))
    return cats


def get_reference(run, bench, texts, codebook):
    f = REP / f'run{run}' / f'{bench}_reference.csv'
    done = {}
    if f.exists():
        d = pd.read_csv(f)
        done = dict(zip(d.doc_idx.astype(int), d.category.astype(int)))
    rng = np.random.RandomState(2000 * run + 7)
    sample = rng.choice(len(texts), min(N_ASSIGN, len(texts)), replace=False)
    pending = [int(i) for i in sample if int(i) not in done]
    if not pending:
        return done
    guard()
    spec = CRITERIA[bench]

    def one(i):
        try:
            c = classify_document(texts[i], codebook, spec)
        except Exception:
            c = None
        return i, (int(c) if c is not None else -1)

    rows = [{'doc_idx': i, 'category': c} for i, c in done.items()]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(one, i) for i in pending]
        for n, fut in enumerate(as_completed(futs), 1):
            i, c = fut.result()
            rows.append({'doc_idx': i, 'category': c})
            if n % 250 == 0:
                pd.DataFrame(rows).to_csv(f, index=False)
    pd.DataFrame(rows).to_csv(f, index=False)
    return {r['doc_idx']: r['category'] for r in rows}


def main():
    score_rows = []
    meta_rows = []
    if OUT_SCORES.exists():
        score_rows = pd.read_csv(OUT_SCORES).to_dict('records')
    done_pairs = {(r['run'], r['bench']) for r in score_rows}

    benches = sorted(CRITERIA)
    for run in RUNS:
        for bench in benches:
            if (run, bench) in done_pairs:
                continue
            texts = pd.read_csv(RQ1C / 'data' / 'rq1' / 'texts' / f'{bench}.csv')['text'] \
                .fillna('').astype(str).tolist()
            codebook = get_codebook(run, bench, texts)
            ref = get_reference(run, bench, texts, codebook)
            usable = {i: c for i, c in ref.items() if c > 0}
            ref_idx = np.array(sorted(usable))
            ref_cat = np.array([usable[i] for i in ref_idx])

            part = pd.read_csv(RQ1C / 'data' / 'rq1' / _dir('partitions') / f'{bench}.csv')
            pcols = [c for c in part.columns if c not in ('doc_idx', 'true_label')]
            for pid in pcols:
                mem = part[pid].to_numpy()
                s, n_used = score_partition(ref_idx, ref_cat, mem)
                score_rows.append({'run': run, 'bench': bench, 'partition_id': pid,
                                   'judge': s, 'n_docs_used': n_used})
            meta_rows.append({'run': run, 'bench': bench,
                              'codebook_categories': len(codebook),
                              'n_reference': len(usable)})
            pd.DataFrame(score_rows).to_csv(OUT_SCORES, index=False)
            print(f'run{run} {bench}: {len(codebook)} cats, {len(usable)} ref docs, '
                  f'{len(pcols)} candidates scored | spend {gateway_spend()}', flush=True)

    if meta_rows:
        old = pd.read_csv(OUT_META).to_dict('records') if OUT_META.exists() else []
        pd.DataFrame(old + meta_rows).drop_duplicates(['run', 'bench']) \
          .to_csv(OUT_META, index=False)
    print('replicates complete')


if __name__ == '__main__':
    main()
