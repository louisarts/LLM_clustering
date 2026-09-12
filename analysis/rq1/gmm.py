"""Gaussian-mixture clustering of text documents.

cluster_gmm(documents, k) embeds the documents, reduces them with UMAP, then
hard-assigns them to k clusters with a Gaussian mixture (k-means-initialised),
returning one integer label per document.
"""

# Import embedder first: it sets the threading env vars before sklearn loads.
from embedder import embed_miniLM, EMBEDDER

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import umap
from threadpoolctl import threadpool_limits

N_UMAP_COMPONENTS = 7


def umap_reduce(embeddings, n_components=N_UMAP_COMPONENTS):
    """Reduce embeddings to n_components dimensions with UMAP."""
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.0,
        low_memory=True,
        random_state=42,
        n_jobs=1,
    )
    return reducer.fit_transform(embeddings).astype(np.float64)


def cluster_gmm(documents, k, embedder=EMBEDDER, n_components=N_UMAP_COMPONENTS,
                embeddings=None, seed=None):
    """Embed, UMAP-reduce, then hard-assign a k-means-initialised Gaussian mixture.

    Pass precomputed `embeddings` to skip the embedding step (UMAP still runs,
    as it needs the full embedding matrix). `seed` fixes the k-means init and the
    mixture fit for reproducible runs. Returns an array of integer cluster
    labels, one per document.
    """
    if embeddings is None:
        embeddings = embed_miniLM(documents, embedder)
    with threadpool_limits(limits=1, user_api="blas"), threadpool_limits(limits=1, user_api="openmp"):
        embeddings_umap = umap_reduce(embeddings, n_components)
        means_init = KMeans(n_clusters=k, n_init=1, random_state=seed).fit(embeddings_umap).cluster_centers_.astype(np.float64)
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            means_init=means_init,
            reg_covar=1e-3,
            random_state=seed,
        )
        gmm.fit(embeddings_umap)
        return gmm.predict(embeddings_umap)
