# Prétraitement, exploration et ACP — Breast Cancer Wisconsin (Diagnostic)

Partie « prétraitement + ACP » du projet M1 *Aide au diagnostic médical*.

## Installation

```bash
pip install -r requirements.txt
```

## Exécution

Place le fichier Kaggle `data.csv` dans `data/raw/`, puis exécute les notebooks dans l'ordre :

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_preprocessing_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_pca.ipynb
```

Les figures sont écrites dans `reports/figures/` et les exports dans `data/processed/`.

## Contenu

| Fichier | Rôle |
|---|---|
| `notebooks/01_preprocessing_eda.ipynb` | Audit de qualité, prétraitement justifié, EDA, outliers, standardisation |
| `notebooks/02_pca.ipynb` | ACP normée, choix du nombre d'axes, cercle, contributions, cos², projection, robustesse |
| `notebooks/_style.py` | Couleurs communes (bénin bleu, malin rouge) |
| `src/preprocessing.py` | **Source unique du prétraitement pour tout le groupe** |
| `src/pca_analysis.py` | ACP « à la main » (notations du cours) + projection d'un nouveau cas |
| `data/processed/clean.csv` | Données nettoyées **non standardisées** |
| `data/processed/pca_*.csv` | Scores, corrélations variables–axes, vecteurs/valeurs propres, paramètres (dashboard) |

## Utilisation par le reste du groupe

```python
import sys; sys.path.insert(0, "src")
import preprocessing as pp
import pca_analysis as pa

df, X, y = pp.load_clean_data()          # 569 × 30, y : B=0 / M=1

# Évaluation d'un modèle : standardisation DANS le pipeline (pas de fuite)
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model = make_pipeline(pp.build_preprocessor(), LinearDiscriminantAnalysis())

# Dashboard : projeter un nouveau cas sur les axes de l'ACP
pca = pa.fit_pca(X)
coords = pca.transform(nouveau_cas_dataframe)[["CP1", "CP2"]]
```

## Résultats clés

- Colonne `Unnamed: 32` : 100 % vide (virgule finale dans l'en-tête) → supprimée. Aucune autre valeur manquante.
- 13 zéros de concavité = zéros structurels (contours convexes, tous bénins) → conservés.
- 2 lignes `worst = mean` (855133, 8810703) → conservées + drapeau `flag_worst_eq_mean`.
- Outliers majoritairement malins (59–90 % selon la méthode) → conservés.
- ACP normée : CP1 = 44,3 %, CP1+CP2 = 63,2 % ; **7 axes pour 90 %**, **10 axes pour 95 %**.
- CP1 dominé par concavité / points concaves / compacité puis taille ; AUC(CP1) = 0,97.
- Robuste : CP1 pivote de < 3° sans les extrêmes ou après log.
- ⚠️ Conditionnement de la matrice de corrélation ≈ 10⁵ : régulariser ou passer par les CP pour Mahalanobis / ADL.
