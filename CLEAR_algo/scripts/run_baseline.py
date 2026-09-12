#!/usr/bin/env python
"""Same-encoder baseline: plain k-means on the identical embeddings the algorithms use.

    python scripts/run_baseline.py                    # unmatched (gemini-embedding-001)

This is the comparison that separates the algorithm's contribution from the encoder's. The
headline "beats published SOTA" numbers are confounded, because gemini-embedding-001 is
stronger than any competitor's encoder; the uplift over plain k-means on that same encoder is
not, and it is the honest measure of what CritClust adds.

No LLM calls - the embeddings are cached, so this is local compute only and costs nothing.
Writes results/baseline_<setting>.csv and prints the uplift table against both algorithms.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from critclust import config
from critclust.data import load_bench, score


def run_baseline(setting='unmatched', benches=None, seeds=None):
    benches = benches or config.benchmarks_for(setting)
    seeds = seeds or config.SEEDS
    rows = []
    for bench in benches:
        docs, gold, k = load_bench(bench)
        if setting == 'unmatched':
            X = np.load(config.GEMINI_EMB / f'{bench}_raw.npy')
        else:
            embedder = config.MATCHED_CFG[bench][0]
            legacy = config.MATCHED_EMB / f'A_{bench}_raw.npy'
            own = config.EMB_CACHE / f'{embedder.replace("/", "_")}_{bench}_raw.npy'
            e5 = config.E5_CACHE / f'e5_{bench}_raw.npy'
            X = np.load(legacy if (legacy.exists() and embedder != 'e5-large-v2')
                        else (e5 if e5.exists() else own))
        for seed in seeds:
            labels = KMeans(k, n_init=3, random_state=seed).fit_predict(X)
            nmi_val, acc_val = score(gold, labels)
            rows.append({'bench': bench, 'k': k, 'seed': seed,
                         'nmi': nmi_val, 'acc': acc_val})
        last = [r for r in rows if r['bench'] == bench]
        print(f'{bench:18} k={k:4}  nmi {np.mean([r["nmi"] for r in last]):5.2f}  '
              f'acc {np.mean([r["acc"] for r in last]):5.2f}', flush=True)
    return pd.DataFrame(rows)


def summarise(df):
    return (df.groupby('bench')
              .agg(k=('k', 'first'),
                   nmi_mean=('nmi', 'mean'), nmi_std=('nmi', 'std'),
                   acc_mean=('acc', 'mean'), acc_std=('acc', 'std'))
              .round(2).reset_index())


def uplift(baseline_summary, setting):
    """Per-benchmark uplift of each algorithm over the baseline on the same encoder."""
    from critclust import runners
    out = baseline_summary[['bench', 'k', 'nmi_mean', 'acc_mean']].rename(
        columns={'nmi_mean': 'km_nmi', 'acc_mean': 'km_acc'})
    for algo in ('A', 'B'):
        got = runners.load_if_available(algo, setting)
        if got is None:
            print(f'CritClust_{algo} / {setting} not available, skipped')
            continue
        s = got[1][['bench', 'nmi_mean', 'acc_mean']].rename(
            columns={'nmi_mean': f'{algo}_nmi', 'acc_mean': f'{algo}_acc'})
        out = out.merge(s, on='bench', how='left')
        out[f'{algo}_d_nmi'] = (out[f'{algo}_nmi'] - out.km_nmi).round(2)
        out[f'{algo}_d_acc'] = (out[f'{algo}_acc'] - out.km_acc).round(2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--setting', default='unmatched', choices=['unmatched', 'matched'])
    args = ap.parse_args()

    df = run_baseline(args.setting)
    df.to_csv(config.RESULTS / f'baseline_{args.setting}_runs.csv', index=False)
    summary = summarise(df)
    summary.to_csv(config.RESULTS / f'baseline_{args.setting}_summary.csv', index=False)

    table = uplift(summary, args.setting)
    table.to_csv(config.RESULTS / f'uplift_{args.setting}.csv', index=False)
    print('\n' + table.to_string(index=False))

    for algo in ('A', 'B'):
        if f'{algo}_d_nmi' not in table:
            continue
        dn, da = table[f'{algo}_d_nmi'].dropna(), table[f'{algo}_d_acc'].dropna()
        print(f'\nCritClust_{algo} vs same-encoder k-means: '
              f'NMI up on {int((dn > 0).sum())}/{len(dn)} (mean {dn.mean():+.2f}), '
              f'ACC up on {int((da > 0).sum())}/{len(da)} (mean {da.mean():+.2f})')


if __name__ == '__main__':
    main()
