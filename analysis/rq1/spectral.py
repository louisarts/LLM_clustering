"""Spectral clustering of text documents.

cluster_spectral(documents, k) embeds the documents and hard-assigns them to k
clusters via spectral clustering on a k-nearest-neighbours affinity graph, in the
same L2-normalised embedding space as k-means (not the UMAP space GMM uses).
The sparse k-NN affinity keeps it tractable on large benchmarks.
"""

# Import embedder first: it sets the threading env vars before sklearn loads.
from embedder import embed_miniLM, EMBEDDER

from sklearn.cluster import SpectralClustering
from threadpoolctl import threadpool_limits


def cluster_spectral(documents, k, embedder=EMBEDDER, embeddings=None, seed=None, n_neighbors=15):
    """Embed documents and hard-cluster them into k groups with spectral clustering.

    Pass precomputed `embeddings` to skip the embedding step. `seed` fixes the
    eigen-solver and final k-means for reproducible runs. Returns an array of
    integer cluster labels, one per document.
    """
    if embeddings is None:
        embeddings = embed_miniLM(documents, embedder)
    model = SpectralClustering(
        n_clusters=k,
        affinity="nearest_neighbors",
        n_neighbors=n_neighbors,
        assign_labels="kmeans",
        random_state=seed,
    )
    with threadpool_limits(limits=1, user_api="blas"), threadpool_limits(limits=1, user_api="openmp"):
        return model.fit_predict(embeddings)
