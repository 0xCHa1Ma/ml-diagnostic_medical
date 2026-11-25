"""
Prétraitement du jeu Breast Cancer Wisconsin (Diagnostic).

Ce module est la source unique de vérité pour le prétraitement du groupe :
toutes les analyses (ACP, ADL, clustering, dashboard) doivent partir de
`load_clean_data()` et, pour toute évaluation, de `build_preprocessor()`.

Décisions documentées dans notebooks/01_preprocessing_eda.ipynb.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "data.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "clean.csv"

TARGET = "diagnosis"
ID_COL = "id"
PARASITE_COL = "Unnamed: 32"
LABEL_MAP = {"B": 0, "M": 1}  # M = classe positive (classe d'intérêt médical)

MEASURES = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave points", "symmetry",
    "fractal_dimension",
]
STATS = ["mean", "se", "worst"]
FEATURES = [f"{m}_{s}" for s in STATS for m in MEASURES]

MEASURES_FR = {
    "radius": "rayon", "texture": "texture", "perimeter": "périmètre",
    "area": "aire", "smoothness": "régularité", "compactness": "compacité",
    "concavity": "concavité", "concave points": "points concaves",
    "symmetry": "symétrie", "fractal_dimension": "dimension fractale",
}


# ---------------------------------------------------------------------------
# Chargement et nettoyage
# ---------------------------------------------------------------------------
def load_raw(path: str | Path = RAW_PATH) -> pd.DataFrame:
    """Charge le fichier brut tel quel (colonne parasite incluse)."""
    return pd.read_csv(path)


def audit(df: pd.DataFrame) -> dict:
    """Résumé chiffré de la qualité des données brutes."""
    feats = [c for c in df.columns if c in FEATURES]
    return {
        "n_lignes": len(df),
        "n_colonnes": df.shape[1],
        "colonnes_100pct_vides": df.columns[df.isna().all()].tolist(),
        "nan_hors_colonnes_vides": int(df.drop(columns=df.columns[df.isna().all()]).isna().sum().sum()),
        "id_dupliques": int(df[ID_COL].duplicated().sum()),
        "lignes_dupliquees_hors_id": int(df.drop(columns=[ID_COL]).duplicated().sum()),
        "valeurs_negatives": int((df[feats] < 0).sum().sum()),
        "lignes_avec_zero": int((df[feats] == 0).any(axis=1).sum()),
    }


def worst_equals_mean_rows(df: pd.DataFrame) -> pd.Index:
    """Lignes où les 10 variables `_worst` sont strictement égales aux `_mean`.

    Incohérent avec la définition (worst = moyenne des 3 plus grandes valeurs)
    dès que l'erreur standard `_se` est non nulle.
    """
    same = np.column_stack(
        [np.isclose(df[f"{m}_worst"], df[f"{m}_mean"]) for m in MEASURES]
    ).all(axis=1)
    return df.index[same]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Applique les décisions de nettoyage justifiées dans le notebook 01.

    1. Supprime la colonne parasite (100 % vide, créée par une virgule finale
       dans l'en-tête du CSV) : aucune information, imputation impossible.
    2. Place `id` en index : identifiant de traçabilité, pas une mesure.
    3. Encode la cible binaire : B -> 0, M -> 1 (colonne `y`).
    4. Ajoute un drapeau `flag_worst_eq_mean` pour les 2 lignes incohérentes
       (conservées : décision et test de robustesse dans les notebooks).
    5. Conserve les zéros de concavité (zéros structurels, pas des manquants).
    """
    out = df.drop(columns=[c for c in df.columns if df[c].isna().all()])
    assert out[ID_COL].is_unique, "id non unique : vérifier la source"
    out = out.set_index(ID_COL)
    out["y"] = out[TARGET].map(LABEL_MAP).astype(int)
    out["flag_worst_eq_mean"] = False
    out.loc[worst_equals_mean_rows(out), "flag_worst_eq_mean"] = True
    return out[[TARGET, "y", "flag_worst_eq_mean"] + FEATURES]


def load_clean_data(path: str | Path = RAW_PATH):
    """Retourne (df_propre, X, y) prêts pour l'analyse."""
    df = clean(load_raw(path))
    return df, df[FEATURES].copy(), df["y"].copy()


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------
def build_preprocessor() -> Pipeline:
    """Pipeline de prétraitement à placer AVANT tout modèle.

    Le StandardScaler doit être ajusté uniquement sur les données
    d'entraînement (ou sur chaque fold d'entraînement en validation croisée).
    Le mettre dans un Pipeline garantit l'absence de fuite de données.
    """
    return Pipeline([("zscore", StandardScaler())])


def zscore(X: pd.DataFrame, ddof: int = 1) -> pd.DataFrame:
    """Standardisation Z-score descriptive (pour l'EDA et l'ACP exploratoire).

    ddof=1 : écart-type corrigé, convention du cours (diagonale de R = 1).
    """
    return (X - X.mean()) / X.std(ddof=ddof)


# ---------------------------------------------------------------------------
# Valeurs aberrantes
# ---------------------------------------------------------------------------
def iqr_flags(X: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
    """Règle de Tukey : True si la valeur sort de [Q1 - k·IQR, Q3 + k·IQR]."""
    q1, q3 = X.quantile(0.25), X.quantile(0.75)
    iqr = q3 - q1
    return (X < q1 - k * iqr) | (X > q3 + k * iqr)


def mad_flags(X: pd.DataFrame, thr: float = 3.5) -> pd.DataFrame:
    """Z-score modifié (médiane / MAD) : True si |M| > thr."""
    med = X.median()
    mad = (X - med).abs().median()
    m = 0.6745 * (X - med) / mad.replace(0, np.nan)
    return m.abs() > thr


def robust_mahalanobis(Z: pd.DataFrame, random_state: int = 0) -> pd.Series:
    """Distance de Mahalanobis² robuste (MCD) sur des données standardisées."""
    mcd = MinCovDet(random_state=random_state).fit(Z.values)
    return pd.Series(mcd.mahalanobis(Z.values), index=Z.index, name="d2_robuste")
