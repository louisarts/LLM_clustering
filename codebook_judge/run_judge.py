"""Score any clustering of any corpus with the codebook judge - no labels needed.

Input: a CSV with one row per document, holding the document text and the cluster
it was assigned to. Plus a one-sentence criterion saying what the clustering is
supposed to be organised by.

    python run_judge.py my_clustering.csv \
        --criterion "which product the customer complaint is about" \
        --text-col text --cluster-col cluster

The judge (thesis §3.1) builds a codebook from a document sample, labels a second
sample against it, and scores your clustering by Adjusted Mutual Information with
that reference labelling. Output: the score printed, plus the codebook and the
reference labelling saved next to the input for inspection.

Needs a .env in the repo root with OPENAI_BASE_URL / OPENAI_API_KEY for the LLM
gateway. Cost: one discovery call plus one call per reference document
(default 300 + 1,000 documents).
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
import LLM_call                                                      # noqa: E402
import judge                                                         # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('csv', help='CSV with the clustered corpus')
    ap.add_argument('--criterion', required=True,
                    help='one sentence: what the clustering organises by')
    ap.add_argument('--text-col', default='text')
    ap.add_argument('--cluster-col', default='cluster')
    ap.add_argument('--llm', default=LLM_call.MODEL,
                    help=f'judge LLM (default {LLM_call.MODEL})')
    ap.add_argument('--discovery', type=int, default=300,
                    help='documents sampled for codebook discovery')
    ap.add_argument('--reference', type=int, default=1000,
                    help='documents labelled for the reference')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--out', default=None,
                    help='output directory (default <csv stem>_judge/)')
    args = ap.parse_args()

    LLM_call.MODEL = args.llm
    judge.MODEL = args.llm
    spec = {'criterion': args.criterion}

    df = pd.read_csv(args.csv)
    for col in (args.text_col, args.cluster_col):
        if col not in df.columns:
            sys.exit(f'column {col!r} not in {args.csv} (has: {list(df.columns)})')
    docs = df[args.text_col].fillna('').astype(str).tolist()
    clusters = pd.factorize(df[args.cluster_col])[0]
    n = len(docs)
    out_dir = Path(args.out) if args.out else Path(args.csv).with_suffix('') \
        .parent / (Path(args.csv).stem + '_judge')
    out_dir.mkdir(parents=True, exist_ok=True)

    # stage 1 - codebook discovery from a document sample (reasoning on)
    rng = np.random.RandomState(args.seed)
    disc_ids = rng.choice(n, min(args.discovery, n), replace=False)
    print(f'discovering codebook from {len(disc_ids)} documents ...')
    codebook = judge.build_codebook([docs[i] for i in disc_ids], spec,
                                    raw_out=out_dir / 'codebook_raw_reply.txt')
    (out_dir / 'codebook.json').write_text(json.dumps(codebook, indent=1))
    print(f'  {len(codebook)} categories -> {out_dir / "codebook.json"}')

    # stage 2 - reference labelling of a disjoint sample (reasoning off, threaded)
    rest = np.setdiff1d(np.arange(n), disc_ids)
    ref_ids = rng.choice(rest, min(args.reference, len(rest)), replace=False)
    print(f'labelling {len(ref_ids)} reference documents ...')

    def one(i):
        return i, judge.classify_document(docs[i], codebook, spec)

    ref = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(one, int(i)) for i in ref_ids]):
            i, cat = fut.result()
            if cat is not None:
                ref[i] = cat
    pd.DataFrame([{'doc_idx': i, 'category': c} for i, c in sorted(ref.items())]) \
        .to_csv(out_dir / 'reference.csv', index=False)

    # degeneracy diagnostics (thesis §3.1 thresholds)
    fits_none = sum(1 for c in ref.values() if c == 0)
    kept = {i: c for i, c in ref.items() if c > 0}
    if fits_none > 0.20 * len(ref):
        print(f'  WARNING: {fits_none}/{len(ref)} documents fit no category - the '
              f'criterion may not describe this corpus')
    top = max(np.bincount(list(kept.values()))) if kept else 0
    if kept and top > 0.50 * len(kept):
        print('  WARNING: one category holds over half the reference - degenerate codebook')

    # stage 3 - AMI between the reference and your clustering (no LLM cost)
    ids = sorted(kept)
    score = adjusted_mutual_info_score([kept[i] for i in ids], clusters[ids])
    print(f'\njudge score (AMI vs reference, {len(ids)} documents): {score:.4f}')
    print(f'artifacts in {out_dir}/')
    return score


if __name__ == '__main__':
    main()
