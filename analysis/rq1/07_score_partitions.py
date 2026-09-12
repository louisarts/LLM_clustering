"""Step 7 - FREE: score every clustering against the judge's reference labelling.

For each of the 608 candidate clusterings, the judge score is the AMI between the
clustering's labels and the LLM's reference assignment, over the sampled documents
the clustering covers. AMI is chance-corrected for granularity, so scores are
comparable across candidates with different numbers of clusters - no same-k
restriction and no trust gate are built into the score.

Also writes results/reference_quality.csv - the validation-only comparison of the
LLM reference against GOLD labels (AMI/ARI, plus the degeneracy diagnostics).
Gold is used here ONLY to evaluate the judge itself for the thesis; scoring a
clustering never touches it.

Idempotent, no LLM calls. Run after 06.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

RQ1 = Path(__file__).resolve().parents[2]
import os
EMB_TAG = os.environ.get('EMB_TAG', 'instructor')
def _dir(base):  # embedder-tagged data dirs; 'instructor' keeps legacy paths
    return base if EMB_TAG == 'instructor' else f'{base}_{EMB_TAG}'

sys.path.append(str(Path(__file__).parent))

from judge import score_partition                                    # noqa: E402

MASTER = RQ1 / 'data' / 'rq1' / ('master_table.csv' if EMB_TAG == 'instructor' else f'master_table_{EMB_TAG}.csv')
QUAL = RQ1 / 'results' / 'rq1' / 'reference_quality.csv'
QUAL.parent.mkdir(parents=True, exist_ok=True)

master = pd.read_csv(MASTER)
# drop every previous judge column so a re-run can never leave stale scores behind
master = master.drop(columns=[c for c in ('judge', 'judge_phi', 'n_pairs_used',
                                          'n_docs_used', 'n_judge_seeds')
                              if c in master.columns])
master['judge'] = np.nan
master['n_docs_used'] = 0

quality = []
for bench in sorted(master.bench.unique()):
    lab_path = RQ1 / 'data' / 'rq1' / 'reference_labels' / f'{bench}.csv'
    if not lab_path.exists():
        print(f'!! {bench}: no reference labels yet (run 05 + 06), skipping')
        continue
    lab = pd.read_csv(lab_path)
    n_asked = len(lab)
    lab = lab.dropna(subset=['category'])
    none_rate = (lab.category == 0).mean() if len(lab) else np.nan
    ref = lab[lab.category > 0]                     # usable reference docs
    ref_idx = ref.doc_idx.to_numpy(int)
    ref_cat = ref.category.to_numpy(int)

    part = pd.read_csv(RQ1 / 'data' / 'rq1' / _dir('partitions') / f'{bench}.csv')
    gold = part['true_label'].to_numpy()
    codebook = json.loads((RQ1 / 'data' / 'rq1' / 'codebooks' / f'{bench}.json').read_text())

    # --- validation only: how good an annotator is the judge on this corpus? ---
    counts = pd.Series(ref_cat).value_counts(normalize=True)
    quality.append({
        'bench': bench,
        'ami_vs_gold': adjusted_mutual_info_score(gold[ref_idx], ref_cat),
        'ari_vs_gold': adjusted_rand_score(gold[ref_idx], ref_cat),
        'n_codebook': len(codebook['categories']),
        'n_categories_used': int(pd.Series(ref_cat).nunique()),
        'k_gold': int(len(np.unique(gold))),
        'n_reference_docs': len(ref),
        'none_rate': round(float(none_rate), 3),
        'top_category_share': round(float(counts.iloc[0]), 3) if len(counts) else np.nan,
    })

    # --- the judge score for every candidate clustering (free) -----------------
    sub = master.bench == bench
    for pid in master.loc[sub, 'partition_id']:
        if pid not in part.columns:
            continue
        ami, n_used = score_partition(ref_idx, ref_cat, part[pid].to_numpy())
        m = sub & (master.partition_id == pid)
        master.loc[m, 'judge'] = ami
        master.loc[m, 'n_docs_used'] = n_used
    print(f'== {bench}: scored {int(sub.sum())} clusterings on {len(ref)} reference docs')

master.to_csv(MASTER, index=False)
q = pd.DataFrame(quality)
q.to_csv(QUAL, index=False)
print(f'\nmaster table -> {MASTER}  ({master.judge.notna().sum()}/{len(master)} scored)')
print(f'reference quality (VALIDATION ONLY, uses gold) -> {QUAL}\n')
print(q.round(3).to_string(index=False))
print('\nread n_codebook vs k_gold as the judge\'s granularity choice; ami_vs_gold as')
print('how good an annotator it is; top_category_share / none_rate as degeneracy flags.')
