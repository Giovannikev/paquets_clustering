import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import os


def plot_clusters(X_pca, labels, title):
    os.makedirs("output", exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    filename = title.lower().replace(" ", "_")
    plt.savefig(f"output/{filename}.png", dpi=300)
    plt.show()

def plot_dendrogram(Z):
    os.makedirs("output", exist_ok=True)
    plt.figure(figsize=(10, 6))
    dendrogram(Z)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    plt.savefig("output/dendrogram.png", dpi=300)
    plt.show()