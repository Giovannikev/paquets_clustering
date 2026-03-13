from src.data_loader import load_dataset
from src.preprocessing import normalize_data, apply_pca
from src.clustering import (
    kmeans_clustering,
    hierarchical_clustering,
    dbscan_clustering
)
from src.vizualisation import plot_clusters, plot_dendrogram
from src.anomaly_detection import detect_anomalies_dbscan


def main():

    data = load_dataset("data/paquets.csv")

    print("Aperçu des données")
    print(data.head())

    X_scaled, scaler = normalize_data(data)

    X_pca, pca = apply_pca(X_scaled)

    # KMEANS
    k_labels, _ = kmeans_clustering(X_scaled)
    plot_clusters(X_pca, k_labels, "K-Means Clustering")

    # HIERARCHICAL
    Z = hierarchical_clustering(X_scaled)
    plot_dendrogram(Z)

    # DBSCAN
    d_labels, _ = dbscan_clustering(X_scaled)

    anomalies = detect_anomalies_dbscan(d_labels)

    print("Indices des anomalies détectées :")
    print(anomalies)


if __name__ == "__main__":
    main()