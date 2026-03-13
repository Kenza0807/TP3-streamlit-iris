import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def suggest_target(df: pd.DataFrame) -> str:
    """
    Devine la colonne cible la plus probable.
    Priorité : SalePrice > price > target > label > dernière colonne.
    """
    candidates = ["SalePrice", "price", "Price", "target", "label", "species", "class"]
    for c in candidates:
        if c in df.columns:
            return c
    return df.columns[-1]


def prepare_features(df: pd.DataFrame, target: str):
    """
    Retourne X (features) et y (target) prêts pour l'entraînement.
    - Encode les colonnes catégorielles
    - Impute les valeurs manquantes
    - target : nom de la colonne cible (choisi par l'utilisateur)
    """
    if target not in df.columns:
        raise ValueError(
            f"La colonne '{target}' est introuvable. "
            f"Colonnes disponibles : {list(df.columns)}"
        )

    df = df.copy()

    # Supprimer colonnes non pertinentes (ID, index)
    drop_cols = ["Id", "id", "ID", "index"]
    df.drop(columns=[c for c in drop_cols if c in df.columns and c != target], inplace=True)

    # Supprimer les lignes où la cible est NaN
    df = df.dropna(subset=[target])

    # Encoder les colonnes catégorielles
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    df.fillna(df.median(numeric_only=True), inplace=True)

    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def get_numeric_features(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).columns.tolist()
