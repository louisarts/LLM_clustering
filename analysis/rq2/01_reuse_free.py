#!/usr/bin/env python
"""RQ2-clean stage 1: the two zero-cost methods, from reused RQ1-clean artifacts.

  codebook     k = category count of the judge's discovery-stage codebook. The ten
               "seeds" ARE the ten independent RQ1-clean judge runs (run 1 original +
               runs 2-10 replicates), each a fresh 300-doc sample through
               build_codebook verbatim, so this method is RQ1's codebook
               methodology by construction, not a reimplementation.
  judge_sweep  hybrid: MiniBatchKMeans at the shared candidate-k grid on the
               INSTRUCTOR (generic instruction) embeddings, each partition scored
               as AMI against that run's judge reference labelling; k = argmax.
               Local compute only.

Shared grid (all methods that select among k values): 25 log-spaced points on
[2, 200], rounded (23 unique). Gold-blind, identical for every corpus. Seeds 0-9
map to judge runs 1-10.

Appends to data/llm_k_runs.csv (idempotent per bench/method/seed).
"""
import json
import os
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score as ami

RQ2C = Path(__file__).resolve().parents[2]
ROOT = RQ2C.parent
RQ1C = ROOT / 'research_question_1_clean'
OUT = RQ2C / 'data' / 'rq2' / 'llm_k_runs.csv'

CRITERIA = json.loads((ROOT / 'research_question_3_clean' / 'criteria.json').read_text())
BENCHES = sorted(CRITERIA)
# 25 log-spaced points on [2, 200], rounded to integers (23 unique)
K_GRID = [2, 3, 4, 5, 6, 8, 9, 11, 14, 17, 20, 24, 29, 36, 43, 52, 63, 77,
          93, 112, 136, 165, 200]


def codebook_path(bench, run):
    if run == 1:
        return RQ1C / 'data' / 'rq2' / 'codebooks' / f'{bench}.json'
    return RQ1C / 'data' / 'rq2' / 'replicates' / f'run{run}' / f'{bench}_codebook.json'


def reference_path(bench, run):
    if run == 1:
        return RQ1C / 'data' / 'rq2' / 'reference_labels' / f'{bench}.csv'
    return RQ1C / 'data' / 'rq2' / 'replicates' / f'run{run}' / f'{bench}_reference.csv'


def load_reference(bench, run):
    lab = pd.read_csv(reference_path(bench, run)).dropna(subset=['category'])
    lab = lab[lab.category > 0]
    return dict(zip(lab.doc_idx.astype(int), lab.category.astype(int)))


def load_embeddings(bench):
    X = np.load(RQ1C / 'data' / 'rq2' / 'embeddings' / f'{bench}_instructor_gen.npy') \
        .astype(np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def k_judge_sweep(X, ref, seed):
    ref_idx = np.array(sorted(ref))
    ref_cat = np.array([ref[i] for i in ref_idx])
    best_k, best_s = None, -1
    for k in K_GRID:
        lab = MiniBatchKMeans(k, n_init=3, random_state=seed,
                              batch_size=1024).fit_predict(X)
        s = ami(ref_cat, lab[ref_idx])
        if s > best_s:
            best_s, best_k = s, k
    return best_k


rows = pd.read_csv(OUT).to_dict('records') if OUT.exists() else []
done = {(r['bench'], r['method'], r['seed']) for r in rows}

for bench in BENCHES:
    X = None
    for run in range(1, 11):
        seed = run - 1
        if (bench, 'codebook', seed) not in done:
            cbk = json.loads(codebook_path(bench, run).read_text())
            rows.append({'bench': bench, 'method': 'codebook', 'seed': seed,
                         'k': len(cbk['categories'])})
        if (bench, 'judge_sweep', seed) not in done:
            if X is None:
                X = load_embeddings(bench)
            rows.append({'bench': bench, 'method': 'judge_sweep', 'seed': seed,
                         'k': int(k_judge_sweep(X, load_reference(bench, run),
                                                seed))})
    pd.DataFrame(rows).to_csv(OUT, index=False)
    got = [r for r in rows if r['bench'] == bench]
    cb = sorted(r['k'] for r in got if r['method'] == 'codebook')
    js = sorted(r['k'] for r in got if r['method'] == 'judge_sweep')
    print(f'{bench}: codebook k {cb} | judge_sweep k {js}', flush=True)

print(f'\n{len(rows)} rows in data/llm_k_runs.csv')
