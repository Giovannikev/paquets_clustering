import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def normalize_data(data: pd.DataFrame):
    # Normalise les données avec StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    return X_scaled, scaler

def apply_pca(X_scaled, n_components=2):
    # Réduction de dimension avec PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, pca

def compute_correlation(data: pd.DataFrame):
    # Calcule la matrice de corrélation
    return data.corr()