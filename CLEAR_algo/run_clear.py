"""Cluster any corpus with CLEAR - criterion in, named clusters out.

Input: a CSV with one row per document, a one-sentence criterion saying what to
organise the corpus by, and the number of clusters k.

    python run_clear.py my_corpus.csv \
        --criterion "which product the customer complaint is about" \
        --k 12 --out my_clusters.csv

Pipeline (thesis §3.3): rewrite each document into a criterion-only phrase, embed
raw/rewritten/averaged spaces, build eight candidate clusterings, let the codebook
judge select the best one, then repair the boundary documents. Output: a CSV with
each document's cluster id and LLM-chosen cluster name, plus the judge score.

Every LLM artifact is cached under artifacts/ by corpus name, so re-runs are
cheap. Needs a .env in the repo root with OPENAI_BASE_URL / OPENAI_API_KEY.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from critclust import config                                          # noqa: E402
from critclust.codebooks import codebook_A                            # noqa: E402
from critclust.embeddings import embed                                # noqa: E402
from critclust.generation import rewrites, exemplars                  # noqa: E402
from critclust.judges import judge_A                                  # noqa: E402
from critclust.llm import bind_llm, classify_batch                    # noqa: E402
from critclust.pools import pool_C                                    # noqa: E402
from critclust.repair import boundary_repair                          # noqa: E402


def build_spaces(name, embedder, docs, rewrite_texts, exemplar_map):
    """Raw, rewrite and averaged embedding spaces plus exemplar prototypes, cached."""
    tag = f'{embedder.replace("/", "_")}_{name}'
    Xr = embed(embedder, docs, config.EMB_CACHE / f'{tag}_raw.npy')
    Xw = embed(embedder, rewrite_texts, config.EMB_CACHE / f'{tag}_rew.npy')
    P = np.stack([embed(embedder, texts, config.EMB_CACHE / f'{tag}_ex{c}.npy').mean(0)
                  for c, texts in exemplar_map.items() if texts])
    Xc = (Xr + Xw) / 2
    Xc /= np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    return Xr, Xw, Xc, P


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('csv', help='CSV with the corpus to cluster')
    ap.add_argument('--criterion', required=True,
                    help='one sentence: what to organise the corpus by')
    ap.add_argument('--k', type=int, required=True, help='number of clusters')
    ap.add_argument('--text-col', default='text')
    ap.add_argument('--name', default=None,
                    help='cache name for this corpus (default: CSV stem)')
    ap.add_argument('--llm', default=config.UNMATCHED_LLM)
    ap.add_argument('--embedder', default=config.UNMATCHED_EMBEDDER)
    ap.add_argument('--judge-size', type=int, default=1000,
                    help='documents in the judge reference labelling')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None,
                    help='output CSV (default <csv stem>_clusters.csv)')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.text_col not in df.columns:
        sys.exit(f'column {args.text_col!r} not in {args.csv} (has: {list(df.columns)})')
    docs = df[args.text_col].fillna('').astype(str).tolist()
    n, k = len(docs), args.k
    name = args.name or 'app_' + Path(args.csv).stem.replace(' ', '_')
    spec = {'criterion': args.criterion}
    out_csv = Path(args.out) if args.out else \
        Path(args.csv).parent / (Path(args.csv).stem + '_clusters.csv')
    print(f'CLEAR on {n} documents, k={k}, criterion: {args.criterion!r}')

    ask = bind_llm(args.llm)

    # stage 0a - codebook discovery (cached by corpus name)
    cb_file = config.CODEBOOKS / f'{name}_codebook.json'
    if cb_file.exists():
        codebook = json.loads(cb_file.read_text())
    else:
        codebook, method = codebook_A(docs, args.criterion, ask=ask)
        cb_file.write_text(json.dumps(codebook, indent=1))
    print(f'codebook: {len(codebook)} categories')

    # stage 0b - criterion rewrites, exemplars, embedding spaces (all cached)
    rewrite_texts = rewrites(name, 'unmatched', docs, args.criterion, ask)
    exemplar_map = exemplars(name, 'unmatched', codebook, args.criterion, ask)
    Xr, Xw, Xc, P = build_spaces(name, args.embedder, docs, rewrite_texts, exemplar_map)
    print(f'embedded: {Xr.shape[0]} docs x {Xr.shape[1]} dims')

    # stage 2 prerequisite - the judge's reference labelling (cached per seed)
    ref_file = config.CODEBOOKS / f'{name}_s{args.seed}_reference.csv'
    if ref_file.exists():
        d = pd.read_csv(ref_file)
        ref = {int(r.doc_idx): int(r.category) for r in d.itertuples()}
    else:
        rng = np.random.RandomState(9000 + args.seed)
        ids = sorted(int(i) for i in
                     rng.choice(n, min(args.judge_size, n), replace=False))
        ref, _ = classify_batch(ids, docs, codebook, spec, ask,
                                label=f'{name} reference')
        pd.DataFrame([{'doc_idx': i, 'category': c} for i, c in sorted(ref.items())]) \
            .to_csv(ref_file, index=False)
    fm = {i: c for i, c in ref.items() if c > 0}
    print(f'judge reference: {len(fm)}/{len(ref)} usable documents')

    # stage 1 + 2 - eight candidates, judge selects
    score_fn, select_fn = judge_A(fm, n)
    cands, cand_spaces = pool_C(Xr, Xw, Xc, P, k, args.seed)
    winner, judge_scores = select_fn(cands)
    print('judge scores: ' + ', '.join(f'{c} {s:.3f}'
                                       for c, s in sorted(judge_scores.items(),
                                                          key=lambda x: -x[1])))
    print(f'selected candidate: {winner}')

    # stage 3 - boundary repair (judge-gated, up to three rounds)
    tag = f'{name}_s{args.seed}'
    labels, rounds_kept, n_calls = boundary_repair(
        docs, cands[winner], cand_spaces[winner], k, args.criterion, spec,
        score_fn, ask, tag=tag)
    print(f'repair: kept {rounds_kept} round(s), {n_calls} LLM calls')

    # cluster names come from the repair stage's naming cache
    names = {}
    for f in sorted(config.REPAIR_CACHE.glob(f'{tag}*names.json')):
        names = json.loads(f.read_text())
    out = df.copy()
    out['cluster'] = labels
    out['cluster_name'] = [names.get(str(c), f'cluster {c}') for c in labels]
    out.to_csv(out_csv, index=False)

    print(f'\nfinal judge score (AMI vs reference): {score_fn(labels):.4f}')
    print(f'clusters written to {out_csv}')
    for c, cnt in out.cluster.value_counts().sort_index().items():
        print(f'  {c:>3}  {cnt:>5} docs  {names.get(str(c), "")}')


if __name__ == '__main__':
    main()
