#!/usr/bin/env python
"""Selector comparison for the aligned CritClust_A study (unmatched setting). Local only.

For every (benchmark, seed): regenerate the five pool_A candidates (deterministic from
cached embeddings + seed), score each against gold (NMI, ACC) and against the cached
per-seed aligned judge reference (AMI), and compute each candidate's silhouette on the
raw document embeddings (one shared space so values are comparable; sample 2000).

Correctness gate: the recomputed judge scores are checked against the judge_* columns
recorded in critclust_A_unmatched_aligned_runs.csv, and the deployed tie-break winner
must reproduce. Mismatches are counted and reported; the analysis is only trustworthy
if they are ~zero.

Writes results/selector_comparison_unmatched.csv, one row per (bench, seed, candidate):
    gold_nmi, gold_acc, judge_ami, silhouette, recorded_winner, match flags.
"""
import os
import sys
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '4')

import numpy as np
import pandas as pd

CLEAN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN))

from critclust import config                                    # noqa: E402
from critclust.data import load_bench, score                    # noqa: E402
from critclust.embeddings import spaces                         # noqa: E402
from critclust.generation import rewrites, exemplars            # noqa: E402
from critclust.judges import judge_A                            # noqa: E402
from critclust.pools import pool_A, pool_C                      # noqa: E402
from sklearn.metrics import silhouette_score                    # noqa: E402

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--algo', choices=['A', 'C'], default='A')
ALGO = _ap.parse_args().algo

SETTING = 'unmatched'
OUT = config.RESULTS / (f'selector_comparison_{SETTING}.csv' if ALGO == 'A'
                        else f'selector_comparison_C_{SETTING}.csv')
RUNS = pd.read_csv(config.RESULTS / f'critclust_{ALGO}_{SETTING}_aligned_runs.csv')
CRIT = config.criteria()

rows = []
if OUT.exists():
    rows = pd.read_csv(OUT).to_dict('records')
done = {(r['bench'], r['seed']) for r in rows}

for bench in config.BENCH19:
    sub = RUNS[RUNS.bench == bench]
    if sub.empty or all((bench, s) in done for s in sub.seed):
        continue
    docs, gold, k = load_bench(bench)
    criterion = CRIT[bench]
    rewrite_texts = rewrites(bench, SETTING, docs)          # cached, no LLM
    cb = __import__('json').loads(
        (config.CODEBOOKS / f'A_{bench}_{SETTING}_codebook.json').read_text())
    exemplar_map = exemplars(bench, SETTING, cb)            # cached, no LLM
    Xr, Xw, Xc, P = spaces(bench, SETTING, docs, rewrite_texts, exemplar_map)

    for _, run in sub.iterrows():
        seed = int(run.seed)
        if (bench, seed) in done:
            continue
        ref = pd.read_csv(config.CODEBOOKS / f'A_{bench}_{SETTING}_s{seed}_ref1000.csv')
        fm = {int(r.doc_idx): int(r.category) for r in ref.itertuples()
              if r.category and r.category > 0}
        score_fn, select_fn = judge_A(fm, len(docs))
        if ALGO == 'C':
            cands, cand_spaces = pool_C(Xr, Xw, Xc, P, k, seed)
        else:
            cands, cand_spaces = pool_A(Xr, Xc, P, k, seed)
        winner_re, judge_scores = select_fn(cands)

        # correctness gate vs the recorded run
        js_ok = all(abs(judge_scores[c] - run[f'judge_{c}']) < 5e-4
                    for c in cands if f'judge_{c}' in run and pd.notna(run[f'judge_{c}']))
        win_ok = (winner_re == run.winner)

        rng = np.random.RandomState(seed)
        for cname, lab in cands.items():
            nmi_v, acc_v = score(gold, lab)
            sil = float(silhouette_score(Xr, lab, sample_size=min(2000, len(docs)),
                                         random_state=seed))
            rows.append({'bench': bench, 'seed': seed, 'candidate': cname,
                         'gold_nmi': nmi_v, 'gold_acc': acc_v,
                         'judge_ami': round(judge_scores[cname], 4),
                         'silhouette': round(sil, 4),
                         'recorded_winner': run.winner, 'regen_winner': winner_re,
                         'judge_scores_match': js_ok, 'winner_match': win_ok})
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f'{bench} s{seed}: winner {winner_re} '
              f'(recorded {run.winner}, scores_match={js_ok})', flush=True)

d = pd.DataFrame(rows)
n_pairs = d.groupby(['bench', 'seed']).ngroups
print(f'\n{len(d)} candidate rows over {n_pairs} (bench, seed) pairs')
print(f"validation: judge scores match on "
      f"{d.groupby(['bench', 'seed']).judge_scores_match.first().mean():.0%} of pairs, "
      f"winner reproduces on {d.groupby(['bench', 'seed']).winner_match.first().mean():.0%}")
