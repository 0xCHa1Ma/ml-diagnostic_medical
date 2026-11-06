import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

# permet d'importer le dossier src/ depuis dashboard/pages/
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.preprocessing import load_data, get_feature_columns

st.set_page_config(page_title="Exploration des données", layout="wide")


@st.cache_data
def charger():
    return load_data()


df = charger()
features = get_feature_columns(df)

st.title("Exploration des données")
st.caption("Démonstrateur académique : ce n'est pas un outil de diagnostic clinique.")

col1, col2, col3 = st.columns(3)
col1.metric("Patientes", len(df))
col2.metric("Variables", len(features))
col3.metric("Cas malins", f"{(df['target'] == 1).mean() * 100:.1f} %")

onglet1, onglet2, onglet3 = st.tabs(["Aperçu", "Statistiques", "Corrélations"])

with onglet1:
    st.subheader("Répartition du diagnostic")
    comptes = df["diagnosis"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(["Bénin (B)", "Malin (M)"], [comptes["B"], comptes["M"]],
           color=["#4c72b0", "#c44e52"])
    ax.set_ylabel("Nombre de patientes")
    st.pyplot(fig)

    st.subheader("Premières lignes")
    st.dataframe(df.head(20))

with onglet2:
    st.subheader("Distribution d'une variable selon le diagnostic")
    variable = st.selectbox("Variable", features)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x=variable, hue="diagnosis", hue_order=["B", "M"],
                 stat="density", common_norm=False, element="step", kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Statistiques descriptives")
    st.dataframe(df[features].describe().T.round(3))

with onglet3:
    corr = df[features].corr()

    st.subheader("Matrice de corrélation")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                xticklabels=True, yticklabels=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Paires de variables fortement corrélées")
    seuil = st.slider("Seuil de corrélation |r|", 0.5, 0.99, 0.9, 0.01)
    paires = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                  .stack().rename("r").reset_index())
    paires.columns = ["variable 1", "variable 2", "r"]
    paires = paires.dropna()
    paires["|r|"] = paires["r"].abs()
    resultat = paires[paires["|r|"] >= seuil].sort_values("|r|", ascending=False)
    st.write(f"{len(resultat)} paires sur {len(paires)}")
    st.dataframe(resultat[["variable 1", "variable 2", "r"]].round(4), hide_index=True)