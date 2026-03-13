# Chargement des données
import pandas as pd
def load_dataset(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    # garder uniquement les colonnes numériques
    data = data.select_dtypes(include=["number"])
    return data