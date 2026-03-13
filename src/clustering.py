from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage

def kmeans_clustering(X, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X)
    return labels, model

def hierarchical_clustering(X, method="complete"):
    Z = linkage(X, method=method)
    return Z

def dbscan_clustering(X, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return labels, model