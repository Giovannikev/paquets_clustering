from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage
import numpy as np


def kmeans_clustering(X, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X)
    return labels, model

def hierarchical_clustering(X, method="complete", sample_size=1000):
    """
    On échantillonne les données pour éviter un dendrogramme énorme
    """
    if X.shape[0] > sample_size:
        idx = np.random.choice(X.shape[0], sample_size, replace=False)
        X = X[idx]

    Z = linkage(X, method=method)
    return Z

def dbscan_clustering(X, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return labels, model