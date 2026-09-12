"""RQ1 step 2 (free): the designed factorial of clusterings.

One representative per family of geometric bias, all on the SAME fixed input
embedding (INSTRUCTOR, from script 01):

  algorithm  family     k        seeds  notes
  ---------  ---------  -------  -----  ------------------------------------------
  kmeans     centroid   grid     0,1,2  house recipe (n_init=1)
  gmm        mixture    grid     0,1,2  house recipe: shared UMAP-7 (seed 42) +
                                        diag covariance + k-means init
  spectral   spectral   grid     0,1,2  15-NN affinity graph
  ward       hierarchy  grid     -      deterministic; one linkage cut at every k;
                                        fitted on the first 10,000 docs (memory),
                                        docs beyond that get membership -1
  leiden     graph      ~grid    0,1,2  15-NN cosine graph, RBConfiguration;
                                        resolution bisected per (k, seed) to land
                                        the achieved k nearest the target

k grid per benchmark: k_true x {1/4, 1/2, 1, 2, 4}, floored at 2, deduplicated.
That is <= 65 partitions per benchmark, every cell filled on purpose:
same-k cells get 10 partitions (3 kmeans + 3 gmm + 3 spectral + 1 ward),
cross-k comparisons get 5 deliberate granularities.

Outputs:
  data/partitions/<bench>.csv   doc_idx, true_label, one column per partition
  data/master_table.csv         one row per partition (judge columns empty;
                                script 05 fills them, script 03 adds silhouette)

Idempotent: benchmarks whose partition file already exists are skipped;
master-table rows are upserted per benchmark. No LLM calls.
Runtime: roughly 20-60 min per benchmark (UMAP + spectral + ward dominate).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
RQ1 = Path(__file__).resolve().parents[2]   # clean 19-benchmark rerun
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

sys.path.append(str(Path(__file__).resolve().parent))  # kmeans/spectral/gmm helpers sit next to this file

from kmeans import cluster_kmeans                       # noqa: E402
from spectral import cluster_spectral                   # noqa: E402
from gmm import umap_reduce                             # noqa: E402
from sklearn.cluster import KMeans                      # noqa: E402
from sklearn.mixture import GaussianMixture             # noqa: E402
import igraph as ig                                     # noqa: E402
import leidenalg as la                                  # noqa: E402

MASTER = RQ1 / 'data' / 'rq1' / f"master_table_{EMB_TAG}.csv" if EMB_TAG != 'instructor' else RQ1 / 'data' / 'rq1' / 'master_table.csv'
K_MULTIPLIERS = [0.25, 0.5, 1, 2, 4]
SEEDS = [0, 1, 2]
WARD_CAP = 10000          # exp5 precedent: Ward linkage is O(n^2) memory
LEIDEN_BISECT_STEPS = 12

FAMILY = {'kmeans': 'centroid', 'gmm': 'mixture', 'spectral': 'spectral',
          'ward': 'hierarchy', 'leiden': 'graph'}


def gmm_house(umap_emb, k, seed):
    """The house GMM recipe (gmm.cluster_gmm) on a precomputed UMAP reduction."""
    with threadpool_limits(limits=1, user_api='blas'), \
         threadpool_limits(limits=1, user_api='openmp'):
        init = KMeans(n_clusters=k, n_init=1, random_state=seed) \
            .fit(umap_emb).cluster_centers_.astype(np.float64)
        return GaussianMixture(n_components=k, covariance_type='diag',
                               means_init=init, reg_covar=1e-3,
                               random_state=seed).fit(umap_emb).predict(umap_emb)


def build_knn_graph(emb, n_neighbors=15):
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='cosine').fit(emb)
    dist, idx = nn.kneighbors(emb)
    src = np.repeat(np.arange(len(emb)), n_neighbors)
    g = ig.Graph(n=len(emb),
                 edges=list(zip(src.tolist(), idx[:, 1:].ravel().tolist())),
                 edge_attrs={'weight': (1.0 - dist[:, 1:].ravel()).tolist()},
                 directed=False)
    g.simplify(combine_edges='max')
    return g


def leiden_at(g, res, seed):
    part = la.find_partition(g, la.RBConfigurationVertexPartition, weights='weight',
                             resolution_parameter=res, seed=seed, n_iterations=2)
    return np.array(part.membership)


def leiden_target_k(g, k_target, seed):
    """Bisect the resolution (k is monotone in it) to land nearest k_target."""
    lo, hi = 1e-3, 200.0
    best, best_gap = None, np.inf
    for _ in range(LEIDEN_BISECT_STEPS):
        mid = np.sqrt(lo * hi)                     # geometric: res spans decades
        mem = leiden_at(g, mid, seed)
        k = len(np.unique(mem))
        if abs(k - k_target) < best_gap:
            best, best_gap = mem, abs(k - k_target)
        if k < k_target:
            lo = mid
        elif k > k_target:
            hi = mid
        else:
            break
    return best


for tpath in sorted((RQ1 / 'data' / 'rq1' / 'texts').glob('*.csv')):
    bench = tpath.stem
    part_path = RQ1 / 'data' / 'rq1' / _dir('partitions') / f'{bench}.csv'
    if part_path.exists():
        print(f'== {bench}: partitions exist, skipping')
        continue
    print(f'== {bench}')
    tdf = pd.read_csv(tpath)
    texts = tdf['text'].fillna('').astype(str).tolist()
    gold = tdf['true_label'].to_numpy()
    emb = load_embeddings(RQ1, bench)
    assert len(emb) == len(texts), f'{bench}: embeddings vs texts mismatch'
    n, k_true = len(texts), len(np.unique(gold))
    k_grid = sorted({max(2, int(round(k_true * m))) for m in K_MULTIPLIERS})
    print(f'   n={n}, k_true={k_true}, k grid = {k_grid}')

    parts = {}   # partition_id -> (membership, method, k_target, seed)

    for k in k_grid:
        for s in SEEDS:
            parts[f'kmeans_k{k}_s{s}'] = (
                np.asarray(cluster_kmeans(texts, k, embeddings=emb, seed=s)),
                'kmeans', k, s)
    print('   kmeans done')

    umap_emb = umap_reduce(emb)                       # house seed 42, shared
    for k in k_grid:
        for s in SEEDS:
            parts[f'gmm_k{k}_s{s}'] = (gmm_house(umap_emb, k, s), 'gmm', k, s)
    print('   gmm done')

    for k in k_grid:
        for s in SEEDS:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')       # kNN graph connectivity warning
                parts[f'spectral_k{k}_s{s}'] = (
                    np.asarray(cluster_spectral(texts, k, embeddings=emb, seed=s)),
                    'spectral', k, s)
    print('   spectral done')

    n_ward = min(n, WARD_CAP)
    Z = linkage(emb[:n_ward].astype(np.float64), method='ward')
    for k in k_grid:
        mem = np.full(n, -1)
        mem[:n_ward] = fcluster(Z, t=k, criterion='maxclust')
        parts[f'ward_k{k}'] = (mem, 'ward', k, np.nan)
    print(f'   ward done (fitted on {n_ward}/{n} docs)')

    g = build_knn_graph(emb)
    for k in k_grid:
        for s in SEEDS:
            mem = leiden_target_k(g, k, s)
            parts[f'leiden_k{k}_s{s}'] = (mem, 'leiden', k, s)
    print('   leiden done')

    out = pd.DataFrame({'doc_idx': np.arange(n), 'true_label': gold})
    rows = []
    for pid, (mem, method, k_target, seed) in parts.items():
        out[pid] = mem
        covered = mem >= 0
        rows.append({'bench': bench, 'partition_id': pid, 'source': 'rq1_factorial',
                     'method': method, 'family': FAMILY[method],
                     'k_target': k_target, 'k': int(len(np.unique(mem[covered]))),
                     'seed': seed, 'n_judge_seeds': 0,
                     'judge_intruder': np.nan, 'judge_separability': np.nan,
                     'judge_combined': np.nan,
                     'ari': adjusted_rand_score(gold[covered], mem[covered])})
    out.to_csv(part_path, index=False)

    master = pd.read_csv(MASTER) if MASTER.exists() else pd.DataFrame()
    if len(master):
        master = master[master.bench != bench]
    master = pd.concat([master, pd.DataFrame(rows)], ignore_index=True)
    master.to_csv(MASTER, index=False)
    print(f'   {len(parts)} partitions -> {part_path.name}; master upserted')

print('\ndone - next: 03_compute_metrics.py, then 04_build_ladder.py')
