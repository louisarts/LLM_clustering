"""Step 3 (free): geometric metrics + document lengths.

Two geometric scores per clustering, both computed in the shared INSTRUCTOR
embedding space on the documents the clustering covers:

  silhouette        cosine silhouette (global compactness-separation)
  knn_consistency   chance-corrected kNN neighbourhood consistency: the fraction
                    of each document's 15 nearest neighbours that share its
                    cluster, minus the fraction expected under a random
                    assignment with the same cluster sizes, rescaled to [.,1].
                    Local rather than global geometry: shape-agnostic, so it does
                    not penalise elongated or manifold-like clusters the way
                    silhouette does. Chance correction removes the mechanical
                    advantage of coarse partitions (one giant cluster would
                    otherwise score 1 trivially).
  geo_silhouette    silhouette computed on GEODESIC distances: shortest paths
                    along the 15-NN cosine graph instead of straight-line cosine.
                    Keeps silhouette's balanced cohesion-and-separation reading
                    while removing its round-blob assumption - two documents at
                    opposite ends of an elongated cluster are close along the
                    manifold. Estimated at 400 fixed-seed landmark documents
                    (exact geodesics from each landmark to every document);
                    unreachable pairs are capped at 1.2x the largest finite
                    distance.

Idempotent: values already present in the master table are kept; only missing
ones are computed. Also writes data/doc_lengths.csv. No LLM calls.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import NearestNeighbors

RQ1 = Path(__file__).resolve().parents[2]
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

MASTER = RQ1 / 'data' / 'rq1' / ('master_table.csv' if EMB_TAG == 'instructor' else f'master_table_{EMB_TAG}.csv')
K_NEIGH = 15
N_LANDMARKS = 400

master = pd.read_csv(MASTER)
for col in ('silhouette', 'knn_consistency', 'geo_silhouette'):
    if col not in master.columns:
        master[col] = np.nan

out, len_rows = [], []
for bench, sub in master.groupby('bench'):
    part = pd.read_csv(RQ1 / 'data' / 'rq1' / _dir('partitions') / f'{bench}.csv')
    emb = load_embeddings(RQ1, bench)
    assert len(part) == len(emb), f'{bench}: partitions vs embeddings length mismatch'

    texts = pd.read_csv(RQ1 / 'data' / 'rq1' / 'texts' / f'{bench}.csv')['text'].fillna('')
    len_rows.append({'bench': bench, 'median_chars': int(texts.str.len().median())})

    # 15-NN graph, built once per benchmark, reused for every clustering
    need_knn = sub.knn_consistency.isna().any()
    need_geo = sub.geo_silhouette.isna().any()
    if need_knn or need_geo:
        nn = NearestNeighbors(n_neighbors=K_NEIGH + 1, metric='cosine').fit(emb)
        dist, idx = nn.kneighbors(emb)
        dist, idx = dist[:, 1:], idx[:, 1:]                # drop self
    if need_geo:
        # geodesic distances from fixed landmarks along the (undirected) kNN graph
        n = len(emb)
        G = csr_matrix((dist.ravel(),
                        (np.repeat(np.arange(n), K_NEIGH), idx.ravel())), shape=(n, n))
        land = np.sort(np.random.RandomState(0).choice(n, size=min(N_LANDMARKS, n),
                                                       replace=False))
        D = dijkstra(G, directed=False, indices=land)      # (landmarks, n)
        finite = np.isfinite(D)
        D = np.where(finite, D, D[finite].max() * 1.2)

    sub = sub.copy()
    for i, row in sub.iterrows():
        pid = row.partition_id
        if pid not in part.columns:
            continue
        mem = part[pid].to_numpy()
        covered = mem >= 0
        labels = mem[covered]
        if len(np.unique(labels)) < 2:
            continue

        if pd.isna(row.silhouette):
            sub.loc[i, 'silhouette'] = silhouette_score(emb[covered], labels,
                                                        metric='cosine')
        if pd.isna(row.knn_consistency):
            mem_nb = mem[idx]                              # (n, 15) neighbour labels
            valid = covered[:, None] & (mem_nb >= 0)
            same = (mem_nb == mem[:, None]) & valid
            raw = same.sum() / valid.sum()
            p = np.bincount(labels) / len(labels)          # chance level for this
            expected = float((p ** 2).sum())               # partition's cluster sizes
            sub.loc[i, 'knn_consistency'] = (raw - expected) / (1 - expected)
        if pd.isna(row.geo_silhouette):
            # silhouette at the landmarks, using geodesic distances
            labs = np.unique(labels)
            lut = {l: j for j, l in enumerate(labs)}
            cols = np.array([lut[l] for l in mem[covered]])
            S = csr_matrix((np.ones(covered.sum()), (np.where(covered)[0], cols)),
                           shape=(len(mem), len(labs)))
            Msum = D @ S                                   # (landmarks, clusters) sums
            counts = np.bincount(cols, minlength=len(labs))
            svals = []
            for t in range(len(land)):
                if not covered[land[t]]:
                    continue
                ci = lut[mem[land[t]]]
                if counts[ci] < 2:
                    continue
                a = Msum[t, ci] / (counts[ci] - 1)         # exclude self (distance 0)
                means = Msum[t] / counts
                b = np.delete(means, ci).min()
                svals.append((b - a) / max(a, b))
            if svals:
                sub.loc[i, 'geo_silhouette'] = float(np.mean(svals))
    out.append(sub)
    done = sub[['silhouette', 'knn_consistency', 'geo_silhouette']].notna().all(axis=1).sum()
    print(f'== {bench}: {done}/{len(sub)} clusterings have both metrics')

master = pd.concat(out, ignore_index=True)
master.to_csv(MASTER, index=False)
pd.DataFrame(len_rows).to_csv(RQ1 / 'data' / 'rq1' / 'doc_lengths.csv', index=False)
print(f'\nsilhouette: {master.silhouette.notna().sum()}/{len(master)} | '
      f'knn_consistency: {master.knn_consistency.notna().sum()}/{len(master)} | '
      f'geo_silhouette: {master.geo_silhouette.notna().sum()}/{len(master)} -> {MASTER}')
