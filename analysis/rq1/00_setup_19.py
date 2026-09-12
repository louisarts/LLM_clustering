#!/usr/bin/env python
"""Stage 0 of the clean 19-benchmark RQ1 rerun: texts + INSTRUCTOR embeddings.

Writes, for each of the 19 literature benchmarks, the two files the original RQ1
pipeline expects:

    data/texts/{bench}.csv                 doc_idx, text, true_label
    data/embeddings/{bench}_instructor.csv doc_idx + 768 columns

Corpora and gold labels come from the rq3_clean loader (identical documents to the RQ3
study). Embeddings are instructor-large for every benchmark, reusing any genuinely
instructor-large cache from the RQ3 studies and computing the rest locally on CPU.
No LLM calls; free.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RQ1C = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research_question_3_clean'))

from critclust import config as c3                      # noqa: E402
from critclust.data import load_bench                   # noqa: E402
from critclust.embeddings import embed                  # noqa: E402

# caches that really are instructor-large
def instructor_cache(bench):
    cands = [c3.EMB_CACHE / f'instructor-large_{bench}_raw.npy']
    if c3.MATCHED_CFG.get(bench, ('',))[0] == 'instructor-large':
        cands.append(c3.MATCHED_EMB / f'A_{bench}_raw.npy')
    for p in cands:
        if p.exists():
            return p
    return None


def main():
    for bench in c3.BENCH19:
        tpath = RQ1C / 'data' / 'rq1' / 'texts' / f'{bench}.csv'
        epath = RQ1C / 'data' / 'rq1' / 'embeddings' / f'{bench}_instructor.csv'
        if tpath.exists() and epath.exists():
            print(f'{bench}: done already')
            continue

        docs, gold, k = load_bench(bench)
        pd.DataFrame({'doc_idx': np.arange(len(docs)), 'text': docs,
                      'true_label': gold}).to_csv(tpath, index=False)

        cache = instructor_cache(bench)
        if cache is not None:
            X = np.load(cache)
            src = cache.name
        else:
            own = RQ1C / 'data' / 'rq1' / 'embeddings' / f'{bench}_instructor.npy'
            X = embed('instructor-large', docs, own)
            src = 'computed'
        assert len(X) == len(docs), f'{bench}: embedding/text length mismatch'

        df = pd.DataFrame(X.astype(np.float32))
        df.insert(0, 'doc_idx', np.arange(len(docs)))
        df.to_csv(epath, index=False)
        print(f'{bench}: n={len(docs)} k={k} dim={X.shape[1]} ({src})', flush=True)


if __name__ == '__main__':
    main()
