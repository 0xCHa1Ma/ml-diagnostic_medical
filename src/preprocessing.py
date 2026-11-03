"""Prétraitement commun du projet : Breast Cancer Wisconsin (Diagnostic).

Toutes les étudiantes utilisent ces fonctions, afin que chaque méthode travaille
sur des données identiques. Règle anti-fuite : en validation croisée, le scaler
doit être ajusté sur les données d'entraînement uniquement (fit_scaler) puis
appliqué au test (apply_scaler).
"""
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "data.csv"

ID_COL = "id"
DIAG_COL = "diagnosis"
TARGET_COL = "target"
LABEL_MAP = {"B": 0, "M": 1}  # M (malin) = classe positive


def load_data(path=RAW_PATH) -> pd.DataFrame:
    """Charge le CSV brut, retire la colonne parasite et code le diagnostic."""
    df = pd.read_csv(path)

    # Colonne parasite : supprimée seulement si elle est entièrement vide
    parasites = [c for c in df.columns if str(c).startswith("Unnamed")]
    for c in parasites:
        if not df[c].isna().all():
            raise ValueError(f"La colonne {c} contient des données : ne pas la supprimer.")
    df = df.drop(columns=parasites)

    # Diagnostic : B = 0 (bénin), M = 1 (malin)
    df[TARGET_COL] = df[DIAG_COL].map(LABEL_MAP)
    if df[TARGET_COL].isna().any():
        raise ValueError("Valeurs de diagnostic inattendues (attendu : B ou M).")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Les 30 variables numériques (hors identifiant et diagnostic)."""
    return [c for c in df.columns if c not in (ID_COL, DIAG_COL, TARGET_COL)]


def get_X_y(df: pd.DataFrame):
    """Renvoie X (30 variables, non standardisées) et y (0 = bénin, 1 = malin)."""
    return df[get_feature_columns(df)].copy(), df[TARGET_COL].copy()


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Ajuste le Z-score sur les données d'entraînement uniquement."""
    return StandardScaler().fit(X_train)


def apply_scaler(scaler: StandardScaler, X: pd.DataFrame) -> pd.DataFrame:
    """Applique un scaler déjà ajusté (entraînement ou test)."""
    return pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)


def standardize_all(X: pd.DataFrame):
    """Z-score sur tout le jeu : réservé à l'exploration (statistiques, ACP, clustering)."""
    scaler = fit_scaler(X)
    return apply_scaler(scaler, X), scaler


if __name__ == "__main__":
    df = load_data()
    X, y = get_X_y(df)
    Xs, _ = standardize_all(X)
    print("Dimensions :", df.shape, "| variables :", X.shape[1])
    print("Classes (0 = B, 1 = M) :", y.value_counts().to_dict())
    print("Moyenne max |z| :", Xs.mean().abs().max())