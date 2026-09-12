#!/usr/bin/env python
"""RQ2-clean stage 0: suite metadata for the 19 RQ1-clean benchmarks. Free, local.

One row per benchmark: gold k (never shown to any method), corpus size, and the
criterion sentence (verbatim from research_question_3_clean/criteria.json — the
single criteria source shared with RQ1-clean and RQ3-clean).

Writes results/suite_meta.csv.
"""
import json
from pathlib import Path

import pandas as pd

RQ2C = Path(__file__).resolve().parents[2]
ROOT = RQ2C.parent
RQ1C = ROOT / 'research_question_1_clean'

CRITERIA = json.loads((ROOT / 'research_question_3_clean' / 'criteria.json').read_text())

rows = []
for bench in sorted(CRITERIA):
    texts = pd.read_csv(RQ1C / 'data' / 'rq2' / 'texts' / f'{bench}.csv')
    part = pd.read_csv(RQ1C / 'data' / 'rq2' / 'partitions' / f'{bench}.csv',
                       usecols=['true_label'])
    rows.append({'bench': bench, 'gold_k': int(part.true_label.nunique()),
                 'n_docs': len(texts), 'criterion': CRITERIA[bench]})
    print(f"{bench}: gold_k={rows[-1]['gold_k']} n={rows[-1]['n_docs']}")

pd.DataFrame(rows).to_csv(RQ2C / 'results' / 'rq2' / 'suite_meta.csv', index=False)
print(f'\nwrote results/suite_meta.csv ({len(rows)} benchmarks)')
