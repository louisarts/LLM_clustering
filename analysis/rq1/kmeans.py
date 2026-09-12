"""K-means clustering of text documents.

cluster_kmeans(documents, k) embeds the documents and hard-assigns them to k
clusters, returning one integer label per document.
"""

# Import embedder first: it sets the threading env vars before sklearn loads.
from embedder import embed_miniLM, EMBEDDER

from sklearn.cluster import KMeans


def cluster_kmeans(documents, k, embedder=EMBEDDER, embeddings=None, seed=None):
    """Embed documents and hard-cluster them into k groups with k-means.

    Pass precomputed `embeddings` to skip the embedding step (e.g. when
    clustering the same documents many times). `seed` fixes the k-means
    initialisation for reproducible runs. Returns an array of integer cluster
    labels, one per document.
    """
    if embeddings is None:
        embeddings = embed_miniLM(documents, embedder)
    return KMeans(n_clusters=k, n_init=1, random_state=seed).fit(embeddings).labels_
