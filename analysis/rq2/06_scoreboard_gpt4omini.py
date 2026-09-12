#!/usr/bin/env python
"""RQ2 gpt-4o-mini bed: the scoreboard statistics, mirroring the analysis
notebook's construction exactly (per-corpus median over the 10 seeds, log2
errors, population-moment CCC, spread = range/median medianed over corpora,
Wilcoxon on |eps| vs the 10-seed silhouette baseline, calibration slopes).

Reads  data/llm_k_runs_{tag}.csv + data/llm_k_runs_{tag}_free.csv
Writes results/table_T2_{tag}.csv and prints the comparison to Flash.
Tag defaults to gpt4omini; pass another bed tag (e.g. gpt35) as argv[1].
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

TAG = sys.argv[1] if len(sys.argv) > 1 else 'gpt4omini'
RQ2C = Path(__file__).resolve().parents[2]
meta = pd.read_csv(RQ2C / 'results' / 'rq2' / 'suite_meta.csv').set_index('bench')

runs = pd.concat([pd.read_csv(RQ2C / 'data' / 'rq2' / f'llm_k_runs_{TAG}.csv'),
                  pd.read_csv(RQ2C / 'data' / 'rq2' / f'llm_k_runs_{TAG}_free.csv')],
                 ignore_index=True)
runs['gold_k'] = runs.bench.map(meta.gold_k)

SILS = pd.read_csv(RQ2C / 'data' / 'rq2' / 'silhouette_baseline_seeds.csv')
sil_med = SILS.groupby('bench').k_sil.median().loc[sorted(meta.index)]
sil_e = np.log2(sil_med.values / meta.gold_k.loc[sil_med.index].values)

T15 = np.log2(1.5)


def ccc(x, y):
    mx, my = x.mean(), y.mean()
    return (2 * np.cov(x, y)[0, 1] * (len(x) - 1) / len(x)
            / (x.var() + y.var() + (mx - my) ** 2))


rows = []
for m, g in runs.groupby('method'):
    med = g.groupby('bench').agg(km=('k', 'median'), gold=('gold_k', 'first'),
                                 n=('k', 'size'))
    e = np.log2(med.km / med.gold)
    x = np.log2(med.gold.astype(float))
    y = np.log2(med.km.astype(float))
    slope = np.polyfit(x, y, 1)[0]
    sp = g.groupby('bench').k.agg(
        lambda d: (d.max() - d.min()) / d.median()).median()
    p = wilcoxon(np.abs(e.values), np.abs(sil_e)).pvalue
    rows.append({'method': m,
                 'w15': int((e.abs() <= T15).sum()),
                 'w10': int(((med.km / med.gold - 1).abs() <= 0.10).sum()),
                 'med_abs_e': round(float(e.abs().median()), 2),
                 'signed': round(float(e.median()), 2),
                 'ccc': round(float(ccc(x.values, y.values)), 2),
                 'spread': round(float(sp), 2),
                 'slope': round(float(slope), 2),
                 'wilcoxon_p': round(float(p), 3),
                 'min_seeds': int(med.n.min())})

T = pd.DataFrame(rows).set_index('method')
ORDER = ['codebook', 'direct', 'probe', 'pairwise', 'sequential', 'simpson',
         'judge_sweep', 'merge_conv']
T = T.loc[[m for m in ORDER if m in T.index]]
T.to_csv(RQ2C / 'results' / 'rq2' / f'table_T2_{TAG}.csv')
print(f'{TAG} bed:')
print(T.to_string())

F = pd.read_csv(RQ2C / 'results' / 'rq2' / 'table_T2.csv')
print('\nFlash bed (published table):')
print(F.to_string(index=False))
