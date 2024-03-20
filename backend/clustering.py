"""
    Clustering functions
"""

# pylint: disable=C0301,C0103,C0303,C0411,W1203

import numpy as np
import umap
from typing import Optional
from sklearn.mixture import GaussianMixture

def reduce_cluster_embeddings(embeddings: np.ndarray, dim: int, n_neighbors : Optional[int] = None, metric: str = "cosine") -> np.ndarray:
    """
        Cluster embeddings and reduce dimension
    """
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    
def get_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 50, random_state: int = 1234):
    """
        Get optimal number of clusters
    """
    max_clusters = min(max_clusters, len(embeddings))
    bics = [GaussianMixture(n_components=n, random_state=random_state).fit(embeddings).bic(embeddings)
            for n in range(1, max_clusters)]
    return np.argmin(bics) + 1

def gmm_clustering(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    """
        GMM clustering
    """
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state).fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters    