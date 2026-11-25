"""
ACP centrée-réduite « à la main » (même démarche que le cours) + utilitaires.

Toutes les quantités suivent les notations du cours :
    Z  : matrice centrée-réduite (n x p), écart-type corrigé (ddof=1)
    R  : matrice de corrélation  R = Zᵀ Z / (n - 1)
    R u_k = λ_k u_k,   u_kᵀ u_k = 1,   λ_1 ≥ λ_2 ≥ ... ≥ λ_p
    F  = Z U           (scores des individus)
    Corr(Z_j, CP_k) = u_jk √λ_k   (coordonnées du cercle des corrélations)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PCAResult:
    mean: pd.Series            # moyennes des variables (pour projeter un nouveau cas)
    std: pd.Series             # écarts-types corrigés
    R: pd.DataFrame            # matrice de corrélation
    eigenvalues: pd.Series     # λ_k
    U: pd.DataFrame            # vecteurs propres en colonnes (p x p)
    scores: pd.DataFrame       # F = Z U  (n x p)

    # ---- quantités dérivées ------------------------------------------------
    @property
    def explained_ratio(self) -> pd.Series:
        return self.eigenvalues / self.eigenvalues.sum()

    @property
    def cumulative_ratio(self) -> pd.Series:
        return self.explained_ratio.cumsum()

    @property
    def loadings(self) -> pd.DataFrame:
        """Corrélations variable–composante : u_jk √λ_k."""
        return self.U * np.sqrt(self.eigenvalues.clip(lower=0))

    @property
    def contributions(self) -> pd.DataFrame:
        """Contribution (%) de chaque variable à chaque axe : 100 u_jk²."""
        return 100 * self.U ** 2

    def cos2_variables(self, k: int = 2) -> pd.Series:
        """Qualité de représentation des variables sur les k premiers axes."""
        return (self.loadings.iloc[:, :k] ** 2).sum(axis=1)

    def n_components_for(self, threshold: float) -> int:
        """Plus petit k tel que la variance cumulée ≥ threshold."""
        return int(np.argmax(self.cumulative_ratio.values >= threshold) + 1)

    def transform(self, X_new: pd.DataFrame) -> pd.DataFrame:
        """Projette de nouveaux cas avec les paramètres appris (pas de refit)."""
        Z = (X_new[self.mean.index] - self.mean) / self.std
        return pd.DataFrame(Z.values @ self.U.values, index=X_new.index, columns=self.U.columns)


def fit_pca(X: pd.DataFrame) -> PCAResult:
    """ACP normée : centrage-réduction, R, diagonalisation, projection."""
    mean, std = X.mean(), X.std(ddof=1)
    Z = (X - mean) / std
    n = len(Z)
    R = Z.T @ Z / (n - 1)

    vals, vecs = np.linalg.eigh(R.values)          # R symétrique -> eigh
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]

    # Convention de signe (arbitraire en ACP) : coefficient de plus grande
    # valeur absolue positif -> résultats reproductibles d'une machine à l'autre.
    signs = np.sign(vecs[np.abs(vecs).argmax(axis=0), range(vecs.shape[1])])
    vecs = vecs * signs

    cols = [f"CP{k + 1}" for k in range(len(vals))]
    U = pd.DataFrame(vecs, index=X.columns, columns=cols)
    eig = pd.Series(vals, index=cols, name="lambda")
    scores = pd.DataFrame(Z.values @ vecs, index=X.index, columns=cols)
    return PCAResult(mean, std, R, eig, U, scores)


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """Angle (degrés) entre deux directions, au signe près."""
    c = abs(float(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))
