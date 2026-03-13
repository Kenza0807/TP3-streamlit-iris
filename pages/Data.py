import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from utils.preprocessing import validate_uploaded_df

st.title("Page Data — Exploration Iris")

uploaded = st.file_uploader("Upload ton fichier CSV", type=["csv"])

if uploaded is None:
    st.info("➡️ Upload un CSV pour continuer.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
    validate_uploaded_df(df, min_rows=10)

    # Optionnel : sécuriser les noms de colonnes (strip)
    df.columns = [c.strip() for c in df.columns]

except Exception as e:
    st.error(f"Fichier invalide : {e}")
    st.stop()

# i tout est OK, on continue
st.session_state["df"] = df
st.success(f"Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
st.dataframe(df.head(15), use_container_width=True)


# --- Détection automatique target (Iris: species)
target = "species" if "species" in df.columns else None
if target is None:
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    target = obj_cols[0] if len(obj_cols) else None

st.subheader("Infos rapides")
c1, c2, c3 = st.columns(3)
c1.metric("Lignes", df.shape[0])
c2.metric("Colonnes", df.shape[1])
c3.metric("Missing values", int(df.isna().sum().sum()))

# --- Distribution target
if target:
    st.subheader("Distribution des classes")
    counts = df[target].astype(str).value_counts()
    st.bar_chart(counts)

# --- Stats numériques
st.subheader("Statistiques descriptives (num)")
num_df = df.select_dtypes(include=np.number)
if num_df.shape[1] > 0:
    st.dataframe(num_df.describe().T, use_container_width=True)

# --- Corr heatmap
if num_df.shape[1] >= 2:
    st.subheader("Corrélations (heatmap)")
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Correlation heatmap")
    st.pyplot(fig)

# --- Scatter simple (2 features) coloré par espèce si possible
if num_df.shape[1] >= 2:
    st.subheader("Scatter (2 features)")
    x_col = st.selectbox("Axe X", num_df.columns, index=0)
    y_col = st.selectbox("Axe Y", num_df.columns, index=1)

    if target:
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=target, ax=ax2)
        ax2.set_title(f"{y_col} vs {x_col} (color = {target})")
        st.pyplot(fig2)
    else:
        st.scatter_chart(df[[x_col, y_col]])

# --- Outliers via boxplots (très utile pour ton prof)
if num_df.shape[1] > 0:
    st.subheader("Boxplots (détection valeurs extrêmes)")
    chosen = st.multiselect("Choisir les colonnes", num_df.columns.tolist(), default=num_df.columns.tolist())
    for col in chosen:
        fig3, ax3 = plt.subplots(figsize=(6, 1.8))
        sns.boxplot(x=df[col], ax=ax3)
        ax3.set_title(f"Boxplot — {col}")
        st.pyplot(fig3)