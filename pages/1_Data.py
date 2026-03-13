import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import load_clean_data, to_csv_bytes
from pathlib import Path

st.set_page_config(page_title="Data", page_icon="", layout="wide")

import logging
logger = logging.getLogger(__name__)

# ── Auth guard ────────────────────────────────────────────────────────────────
if not st.session_state.get("authenticated", False):
    st.warning(" Veuillez vous connecter depuis la page d'accueil.")
    st.stop()

logger.info(f"Page chargée — user : {st.session_state.get('username', '?')}")


with open(Path(__file__).parent.parent / "assets" / "style.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title(" 1 — Données & Exploration")

# ── Upload ────────────────────────────────────────────────────────────────────
upload = st.file_uploader("Déposez un fichier CSV (optionnel — sinon House Prices est utilisé)", type=["csv"])

if upload is not None:
    # Validation des entrées : taille et format
    MAX_SIZE_MB = 10
    file_size_mb = upload.size / (1024 * 1024)
    if file_size_mb > MAX_SIZE_MB:
        st.error(f" Fichier trop volumineux ({file_size_mb:.1f} MB). Maximum : {MAX_SIZE_MB} MB.")
        logger.warning(f"Upload refusé — taille : {file_size_mb:.1f} MB")
        st.stop()
    try:
        df = pd.read_csv(upload)
        if df.empty or df.shape[1] < 2:
            st.error(" Le fichier CSV semble vide ou ne contient qu'une seule colonne.")
            st.stop()
        st.session_state["df_uploaded"] = df
        logger.info(f"CSV chargé : {upload.name} — {df.shape}")
        st.success(f"**{upload.name}** chargé — {df.shape[0]} lignes × {df.shape[1]} colonnes")
    except Exception as e:
        st.error(f" Erreur de lecture du CSV : {e}")
        logger.error(f"Erreur lecture CSV : {e}")
        st.stop()
else:
    if "df_uploaded" in st.session_state:
        df = st.session_state["df_uploaded"]
        st.info(" Dataset uploadé (en session).")
    else:
        df = load_clean_data()
        st.info(" Dataset House Prices par défaut.")

num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()

# ── Métriques globales ────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Lignes",              f"{df.shape[0]:,}")
c2.metric("Colonnes",            f"{df.shape[1]}")
c3.metric("Colonnes numériques", str(len(num_cols)))
c4.metric("Valeurs manquantes",  f"{df.isnull().sum().sum():,}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Aperçu", "ℹ️ Info colonnes", " Statistiques", " Visualisations",  "Valeurs manquantes"
])

# ── Tab 1 : Aperçu ────────────────────────────────────────────────────────────
with tab1:
    n = st.slider("Nombre de lignes", 5, 100, 10)
    st.dataframe(df.head(n), use_container_width=True)
    st.download_button("⬇Télécharger CSV", data=to_csv_bytes(df),
                       file_name="dataset.csv", mime="text/csv")

# ── Tab 2 : Info colonnes ─────────────────────────────────────────────────────
with tab2:
    info = pd.DataFrame({
        "Type":     df.dtypes.astype(str),
        "Non-nuls": df.count(),
        "Nuls":     df.isnull().sum(),
        "% Nuls":   (df.isnull().mean() * 100).round(2),
        "Unique":   df.nunique()
    })
    st.dataframe(info, use_container_width=True)

# ── Tab 3 : Statistiques ──────────────────────────────────────────────────────
with tab3:
    st.dataframe(df.describe().T.round(3), use_container_width=True)

    if len(num_cols) >= 2:
        st.markdown("### 🔗 Matrice de corrélation")
        corr = df[num_cols].corr()
        fig_corr = px.imshow(corr, color_continuous_scale="RdBu", zmin=-1, zmax=1,
                             text_auto=".2f", title="Corrélation entre variables numériques",
                             aspect="auto")
        fig_corr.update_layout(height=max(400, len(num_cols) * 30))
        st.plotly_chart(fig_corr, use_container_width=True)

# ── Tab 4 : Visualisations ────────────────────────────────────────────────────
with tab4:
    st.markdown("###  Visualisations interactives")

    viz_type = st.selectbox("Type de graphique", [
        " Histogramme", " Boxplot", " Scatter plot", " Distribution (violin)", " Comptage (barplot)"
    ])

    if viz_type == " Histogramme":
        col_hist = st.selectbox("Variable", num_cols, key="hist_col")
        color_by = st.selectbox("Couleur par (optionnel)", ["Aucun"] + cat_cols, key="hist_color")
        nbins    = st.slider("Nombre de bins", 10, 100, 30)
        color_arg = None if color_by == "Aucun" else color_by
        fig = px.histogram(df, x=col_hist, nbins=nbins, color=color_arg,
                           title=f"Distribution de {col_hist}",
                           marginal="box", opacity=0.8)
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == " Boxplot":
        col_box = st.selectbox("Variable numérique (y)", num_cols, key="box_y")
        grp_box = st.selectbox("Grouper par (optionnel)", ["Aucun"] + cat_cols, key="box_grp")
        grp_arg = None if grp_box == "Aucun" else grp_box
        if grp_arg and df[grp_arg].nunique() > 30:
            top30 = df[grp_arg].value_counts().head(30).index
            df_plot = df[df[grp_arg].isin(top30)]
            st.caption(" Affichage limité aux 30 catégories les plus fréquentes.")
        else:
            df_plot = df
        fig = px.box(df_plot, x=grp_arg, y=col_box, color=grp_arg,
                     title=f"Boxplot de {col_box}" + (f" par {grp_arg}" if grp_arg else ""),
                     points="outliers")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == " Scatter plot":
        col_x    = st.selectbox("Axe X", num_cols, key="sc_x")
        col_y    = st.selectbox("Axe Y", num_cols, index=min(1, len(num_cols)-1), key="sc_y")
        color_sc = st.selectbox("Couleur par (optionnel)", ["Aucun"] + cat_cols + num_cols, key="sc_c")
        size_sc  = st.selectbox("Taille par (optionnel)", ["Aucun"] + num_cols, key="sc_s")
        color_arg = None if color_sc == "Aucun" else color_sc
        size_arg  = None if size_sc  == "Aucun" else size_sc
        fig = px.scatter(df, x=col_x, y=col_y, color=color_arg, size=size_arg,
                         title=f"Scatter : {col_x} vs {col_y}",
                         opacity=0.6, trendline="ols" if color_arg is None else None)
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == " Distribution (violin)":
        col_vio = st.selectbox("Variable numérique", num_cols, key="vio_col")
        grp_vio = st.selectbox("Grouper par", ["Aucun"] + cat_cols, key="vio_grp")
        grp_arg = None if grp_vio == "Aucun" else grp_vio
        if grp_arg and df[grp_arg].nunique() > 15:
            top15 = df[grp_arg].value_counts().head(15).index
            df_plot = df[df[grp_arg].isin(top15)]
            st.caption(" Affichage limité aux 15 catégories les plus fréquentes.")
        else:
            df_plot = df
        fig = px.violin(df_plot, x=grp_arg, y=col_vio, color=grp_arg, box=True,
                        title=f"Violin plot de {col_vio}" + (f" par {grp_arg}" if grp_arg else ""))
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == " Comptage (barplot)":
        if not cat_cols:
            st.warning("Aucune colonne catégorielle dans ce dataset.")
        else:
            col_bar = st.selectbox("Variable catégorielle", cat_cols, key="bar_col")
            top_n   = st.slider("Top N catégories", 5, 30, 15)
            counts  = df[col_bar].value_counts().head(top_n).reset_index()
            counts.columns = [col_bar, "count"]
            fig = px.bar(counts, x=col_bar, y="count", color="count",
                         title=f"Top {top_n} valeurs de {col_bar}",
                         color_continuous_scale="Viridis",
                         labels={"count": "Nombre d'occurrences"})
            fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

# ── Tab 5 : Valeurs manquantes ────────────────────────────────────────────────
with tab5:
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        st.success(" Aucune valeur manquante !")
    else:
        st.markdown(f"**{len(missing)} colonnes** contiennent des valeurs manquantes.")
        fig = px.bar(x=missing.index, y=missing.values,
                     labels={"x": "Colonne", "y": "Valeurs manquantes"},
                     title="Valeurs manquantes par colonne",
                     color=missing.values, color_continuous_scale="Reds")
        fig.update_layout(xaxis_tickangle=-45, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            pd.DataFrame({"Colonne": missing.index, "Manquants": missing.values,
                          "% Manquants": (missing / len(df) * 100).round(2).values}),
            use_container_width=True, hide_index=True
        )

# ── Sidebar filtres ───────────────────────────────────────────────────────────
st.sidebar.header("Filtres")
df_f = df.copy()
for col in num_cols[:3]:
    vmin, vmax = float(df[col].min()), float(df[col].max())
    rng = st.sidebar.slider(col, vmin, vmax, (vmin, vmax))
    df_f = df_f[(df_f[col] >= rng[0]) & (df_f[col] <= rng[1])]

st.divider()
st.subheader(f"Données filtrées — {len(df_f):,} lignes")
st.dataframe(df_f.head(20), use_container_width=True, height=300)
