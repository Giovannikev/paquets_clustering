import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import os


OUTPUT_DIR = "output"


def plot_clusters(X_pca, labels, title):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=20)

    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")

    filename = title.lower().replace(" ", "_")

    plt.savefig(f"{OUTPUT_DIR}/{filename}.png", dpi=300)
    plt.show()
    plt.close()


def plot_dendrogram(Z):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(10, 6))

    dendrogram(
        Z,
        truncate_mode="lastp",   # réduit l'affichage
        p=30                     # nombre de clusters affichés
    )

    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Clusters")
    plt.ylabel("Distance")

    plt.savefig(f"{OUTPUT_DIR}/dendrogram.png", dpi=300)

    plt.show()
    plt.close()