#!/usr/bin/env python
"""Seed-varied silhouette-argmax baseline: the RQ2 baseline re-run at 10 k-means
seeds per corpus, so its run-to-run spread is measured the same way as the LLM
methods' instead of being fixed to a single seed. Same grid, embeddings and
silhouette configuration as the deterministic baseline in the analysis notebook;
the seed drives both the MiniBatchKMeans initialisation and the silhouette
subsample.

Output: data/silhouette_baseline_seeds.csv   bench, seed, k_sil, gold
Pure local compute.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

RQ2C = Path(__file__).resolve().parents[2]
RQ1C = RQ2C.parent / 'research_question_1_clean'
GRID = [2, 3, 4, 5, 6, 8, 9, 11, 14, 17, 20, 24, 29, 36, 43, 52, 63, 77,
        93, 112, 136, 165, 200]
SEEDS = range(10)
OUT = RQ2C / 'data' / 'rq2' / 'silhouette_baseline_seeds.csv'

meta = pd.read_csv(RQ2C / 'results' / 'rq2' / 'suite_meta.csv').set_index('bench')
rows = pd.read_csv(OUT).to_dict('records') if OUT.exists() else []
done = {(r['bench'], r['seed']) for r in rows}

for b in sorted(meta.index):
    X = np.load(RQ1C / 'data' / 'rq2' / 'embeddings' / f'{b}_instructor_gen.npy').astype('float32')
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    for seed in SEEDS:
        if (b, seed) in done:
            continue
        best_k, best_s = None, -2
        for k in GRID:
            lab = MiniBatchKMeans(k, n_init=3, random_state=seed,
                                  batch_size=1024).fit_predict(X)
            s = silhouette_score(X, lab, sample_size=2000, random_state=seed)
            if s > best_s:
                best_s, best_k = s, k
        rows.append({'bench': b, 'seed': seed, 'k_sil': best_k,
                     'gold': int(meta.loc[b, 'gold_k'])})
        pd.DataFrame(rows).to_csv(OUT, index=False)
    ks = sorted(r['k_sil'] for r in rows if r['bench'] == b)
    print(f'{b}: k_sil over seeds {ks}', flush=True)
print('silhouette seed baseline complete')
