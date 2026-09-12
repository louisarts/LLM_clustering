#!/usr/bin/env python
"""Stage 8: assemble the analysis-ready tables the clean notebook reads. Free, local.

Adds to the master table the two quantities the original notebook computed inline:

    ami        AMI between each candidate and gold, over the documents it covers
    calinski   Calinski-Harabasz of the candidate on the INSTRUCTOR embeddings

and computes, per benchmark, the free single-run judge-of-gold: the judge score of the
true partition (AMI between the reference labelling and gold over the reference
sample), plus its percentile among the candidates' judge scores.

Writes results/analysis_table.csv (one row per candidate clustering) and
results/bench_meta.csv (one row per benchmark).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score as ami_score
from sklearn.metrics import calinski_harabasz_score

RQ1C = Path(__file__).resolve().parents[2]
import os
EMB_TAG = os.environ.get('EMB_TAG', 'instructor')
def _dir(base):  # embedder-tagged data dirs; 'instructor' keeps legacy paths
    return base if EMB_TAG == 'instructor' else f'{base}_{EMB_TAG}'

def load_embeddings(rq1, bench):
    """Embeddings for this benchmark: .npy (preferred, memory-light) or legacy CSV."""
    p = rq1 / 'data' / 'embeddings' / f'{bench}_{EMB_TAG}.npy'
    if p.exists():
        import numpy as _np
        return _np.load(p).astype('float32')
    import pandas as _pd
    return _pd.read_csv(rq1 / 'data' / 'embeddings' / f'{bench}_{EMB_TAG}.csv') \
        .drop(columns='doc_idx').to_numpy(dtype='float32')


master = pd.read_csv(RQ1C / 'data' / 'rq1' / ('master_table.csv' if EMB_TAG == 'instructor' else f'master_table_{EMB_TAG}.csv'))
master = master[master.source == 'rq1_factorial'].copy()

rows_meta = []
ami_col, ch_col = {}, {}

for bench, sub in master.groupby('bench'):
    part = pd.read_csv(RQ1C / 'data' / 'rq1' / _dir('partitions') / f'{bench}.csv')
    gold = part['true_label'].to_numpy()
    emb = load_embeddings(RQ1C, bench)

    for _, r in sub.iterrows():
        mem = part[r.partition_id].to_numpy()
        ok = mem >= 0
        ami_col[(bench, r.partition_id)] = (
            ami_score(gold[ok], mem[ok]) if ok.sum() >= 50 else np.nan)
        try:
            ch_col[(bench, r.partition_id)] = (
                calinski_harabasz_score(emb[ok], mem[ok])
                if len(np.unique(mem[ok])) >= 2 else np.nan)
        except Exception:
            ch_col[(bench, r.partition_id)] = np.nan

    # judge-of-gold (single run, free): reference labelling vs gold on the sample
    ref = pd.read_csv(RQ1C / 'data' / 'rq1' / 'reference_labels' / f'{bench}.csv')
    ref = ref[ref.category > 0]
    jg = ami_score(gold[ref.doc_idx.to_numpy()], ref.category.to_numpy())
    cand_scores = sub.judge.dropna()
    pctile = float((cand_scores < jg).mean() * 100) if len(cand_scores) else np.nan

    cbk = json.loads((RQ1C / 'data' / 'rq1' / 'codebooks' / f'{bench}.json').read_text())
    n_cats = len(cbk['categories'])
    k_gold = int(pd.Series(gold).nunique())
    rows_meta.append({'bench': bench, 'k_gold': k_gold, 'codebook_categories': n_cats,
                      'granularity_ratio': round(n_cats / k_gold, 3),
                      'n_reference': len(ref), 'judge_of_gold': round(jg, 4),
                      'gold_percentile': round(pctile, 1),
                      'n_candidates': len(sub)})
    print(f'{bench}: {len(sub)} candidates | codebook {n_cats} vs k={k_gold} '
          f'| judge(gold)={jg:.3f} at pct {pctile:.0f}', flush=True)

master['ami'] = [ami_col[(b, p)] for b, p in zip(master.bench, master.partition_id)]
master['calinski'] = [ch_col[(b, p)] for b, p in zip(master.bench, master.partition_id)]
master.to_csv(RQ1C / 'results' / 'rq1' / ('analysis_table.csv' if EMB_TAG == 'instructor' else f'analysis_table_{EMB_TAG}.csv'), index=False)
pd.DataFrame(rows_meta).to_csv(RQ1C / 'results' / 'rq1' / ('bench_meta.csv' if EMB_TAG == 'instructor' else f'bench_meta_{EMB_TAG}.csv'), index=False)
print(f'\nwrote analysis_table.csv ({len(master)} rows) and bench_meta.csv '
      f'({len(rows_meta)} benchmarks)')
