import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle, os

# PAR
from pathlib import Path



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              accuracy_score, f1_score, confusion_matrix)

from utils.data_loader import load_clean_data
from utils.preprocessing import prepare_features, suggest_target
from utils.visualization import scatter_pred_vs_real, feature_importance_chart

st.set_page_config(page_title="Training", page_icon="", layout="wide")

import logging
logger = logging.getLogger(__name__)

# ── Auth guard ────────────────────────────────────────────────────────────────
if not st.session_state.get("authenticated", False):
    st.warning(" Veuillez vous connecter depuis la page d'accueil.")
    st.stop()

logger.info(f"Page chargée — user : {st.session_state.get('username', '?')}")

with open(Path(__file__).parent.parent / "assets" / "style.css", "r", encoding="utf-8") as f:

    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title(" 2 — Entraînement du Modèle")

# ── Dataset ──────────────────────────────────────────────────────────────────
if "df_uploaded" in st.session_state:
    df = st.session_state["df_uploaded"]
    st.info(" Dataset uploadé en session.")
else:
    df = load_clean_data()
    st.info(" Dataset House Prices par défaut.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header(" Configuration")

default_target = suggest_target(df)
target_col = st.sidebar.selectbox(
    " Colonne cible (y)",
    options=list(df.columns),
    index=list(df.columns).index(default_target),
)

is_numeric = pd.api.types.is_numeric_dtype(df[target_col])
n_unique   = df[target_col].nunique()
task_type  = "régression" if (is_numeric and n_unique > 10) else "classification"
st.sidebar.caption(f"Tâche détectée : **{task_type}**")

all_feats = [c for c in df.columns if c != target_col]
selected  = st.sidebar.multiselect(" Features à utiliser", options=all_feats, default=all_feats)
if not selected:
    st.sidebar.error("Sélectionnez au moins une feature.")
    st.stop()

df_work = df[selected + [target_col]]

if task_type == "régression":
    model_options = ["Random Forest", "Gradient Boosting", "Régression Linéaire", "Ridge"]
else:
    model_options = ["Random Forest", "Gradient Boosting", "Régression Logistique"]

model_name = st.sidebar.selectbox("Algorithme", model_options)
test_size  = st.sidebar.slider("% jeu de test", 10, 40, 20) / 100

if model_name == "Random Forest":
    n_est   = st.sidebar.slider("n_estimators", 50, 500, 100, 50)
    m_depth = st.sidebar.slider("max_depth", 2, 20, 10)
elif model_name == "Gradient Boosting":
    n_est = st.sidebar.slider("n_estimators", 50, 300, 100, 50)
    lr    = st.sidebar.slider("learning_rate", 0.01, 0.3, 0.1, 0.01)
elif model_name == "Ridge":
    alpha = st.sidebar.slider("alpha", 0.01, 100.0, 1.0, 0.01)

train_btn = st.sidebar.button(" Entraîner", type="primary", use_container_width=True)

with st.expander("Aperçu du dataset de travail", expanded=False):
    st.write(f"**{df_work.shape[0]} lignes × {df_work.shape[1]} colonnes** — cible : **{target_col}**")
    st.dataframe(df_work.head(8), use_container_width=True)

# ── Entraînement ──────────────────────────────────────────────────────────────
if train_btn:
    try:
        X, y = prepare_features(df_work, target=target_col)
    except ValueError as e:
        st.error(f" {e}")
        st.stop()

    with st.spinner("Entraînement en cours..."):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        if task_type == "régression":
            if model_name == "Random Forest":
                mdl = RandomForestRegressor(n_estimators=n_est, max_depth=m_depth, random_state=42, n_jobs=-1)
            elif model_name == "Gradient Boosting":
                mdl = GradientBoostingRegressor(n_estimators=n_est, learning_rate=lr, random_state=42)
            elif model_name == "Régression Linéaire":
                mdl = LinearRegression()
            else:
                mdl = Ridge(alpha=alpha)
        else:
            if model_name == "Random Forest":
                mdl = RandomForestClassifier(n_estimators=n_est, max_depth=m_depth, random_state=42, n_jobs=-1)
            elif model_name == "Gradient Boosting":
                mdl = GradientBoostingClassifier(n_estimators=n_est, learning_rate=lr, random_state=42)
            else:
                mdl = LogisticRegression(max_iter=500, random_state=42)

        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)

        label_map = {}
        if task_type == "classification":
            label_map = {int(i): str(v) for i, v in enumerate(sorted(y.unique()))}
            metrics = {
                "Accuracy":      accuracy_score(y_test, y_pred),
                "F1 (weighted)": f1_score(y_test, y_pred, average="weighted")
            }
        else:
            metrics = {
                "MAE":  mean_absolute_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "R²":   r2_score(y_test, y_pred)
            }

        st.session_state.update({
            "model": mdl, "feature_names": X.columns.tolist(),
            "model_name": model_name, "target_col": target_col,
            "task_type": task_type, "metrics": metrics,
            "y_test": y_test.values, "y_pred": y_pred,
            "label_map": label_map,
            "classes": sorted(y.unique()) if task_type == "classification" else []
        })

        os.makedirs("models", exist_ok=True)
        with open("models/trained_model.pkl", "wb") as f:
            pickle.dump({"model": mdl, "features": X.columns.tolist(),
                         "target": target_col, "task_type": task_type,
                         "label_map": label_map}, f)

    st.success(f" **{model_name}** entraîné — cible : **{target_col}** — tâche : **{task_type}**")

# ── Résultats ─────────────────────────────────────────────────────────────────
if "metrics" in st.session_state:
    m         = st.session_state["metrics"]
    task      = st.session_state["task_type"]
    trained   = st.session_state["model"]
    feat_list = st.session_state["feature_names"]
    y_test    = st.session_state["y_test"]
    y_pred    = st.session_state["y_pred"]
    lmap      = st.session_state.get("label_map", {})

    # Métriques
    st.subheader(" Performances")
    cols = st.columns(len(m))
    for c, (k, v) in zip(cols, m.items()):
        fmt = f"{v:.4f}" if k in ("R²","Accuracy","F1 (weighted)") else f"{v:,.2f}"
        c.metric(k, fmt)

    st.divider()

    # ── Graphiques selon la tâche ─────────────────────────────────────────────
    if task == "régression":
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("###  Prédites vs Réelles")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
                                     marker=dict(color="#3498db", opacity=0.5, size=6),
                                     name="Points"))
            lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
            fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                                     line=dict(color="red", dash="dash"), name="Idéal"))
            fig.update_layout(xaxis_title="Réel", yaxis_title="Prédit",
                              title="Valeurs Prédites vs Réelles")
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown("###  Distribution des résidus")
            residuals = y_test - y_pred
            fig_r = px.histogram(x=residuals, nbins=40,
                                 title="Résidus (Réel − Prédit)",
                                 labels={"x": "Résidu"},
                                 color_discrete_sequence=["#e74c3c"])
            fig_r.add_vline(x=0, line_dash="dash", line_color="black")
            fig_r.add_vline(x=np.mean(residuals), line_dash="dot", line_color="blue",
                            annotation_text=f"Moyenne: {np.mean(residuals):.1f}")
            st.plotly_chart(fig_r, use_container_width=True)

        # Importance features (ligne complète si dispo)
        if hasattr(trained, "feature_importances_"):
            st.markdown("###  Importance des features")
            imp = pd.Series(trained.feature_importances_, index=feat_list).sort_values(ascending=False)
            top_n = min(len(imp), 20)  # Afficher jusqu'à 20, pas hardcodé à 15
            imp_top = imp.head(top_n).sort_values()
            fig_imp = px.bar(x=imp_top.values, y=imp_top.index, orientation="h",
                             title=f"Top {top_n} features les plus importantes (sur {len(imp)} au total)",
                             labels={"x": "Importance", "y": "Feature"},
                             color=imp_top.values, color_continuous_scale="Blues")
            fig_imp.update_layout(yaxis_title="", coloraxis_showscale=False,
                                  height=max(400, top_n * 25))
            st.plotly_chart(fig_imp, use_container_width=True)

    else:  # Classification
        col_l, col_r = st.columns(2)

        # Noms réels des classes
        classes_raw = st.session_state.get("classes", sorted(set(y_test)))
        class_names = [lmap.get(int(c), str(c)) if lmap else str(c) for c in classes_raw]

        with col_l:
            st.markdown("###  Matrice de confusion")
            cm = confusion_matrix(y_test, y_pred, labels=classes_raw)
            fig_cm = px.imshow(cm, x=class_names, y=class_names,
                               color_continuous_scale="Blues",
                               title="Matrice de confusion",
                               labels=dict(x="Prédit", y="Réel", color="Nombre"),
                               text_auto=True)
            fig_cm.update_layout(xaxis_title="Classe prédite", yaxis_title="Classe réelle")
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_r:
            st.markdown("###  Distribution des prédictions")
            pred_names  = [lmap.get(int(p), str(p)) if lmap else str(p) for p in y_pred]
            real_names  = [lmap.get(int(r), str(r)) if lmap else str(r) for r in y_test]
            dist_df = pd.DataFrame({"Classe": pred_names + real_names,
                                    "Type": ["Prédite"]*len(pred_names) + ["Réelle"]*len(real_names)})
            fig_dist = px.histogram(dist_df, x="Classe", color="Type", barmode="group",
                                    title="Distribution Prédites vs Réelles par classe",
                                    color_discrete_map={"Prédite": "#3498db", "Réelle": "#e74c3c"})
            st.plotly_chart(fig_dist, use_container_width=True)

        # Importance des features (classification)
        if hasattr(trained, "feature_importances_"):
            st.markdown("###  Importance des features")
            imp = pd.Series(trained.feature_importances_, index=feat_list).sort_values(ascending=False)
            top_n = min(len(imp), 20)
            imp_top = imp.head(top_n).sort_values()
            fig_imp = px.bar(x=imp_top.values, y=imp_top.index, orientation="h",
                             title=f"Top {top_n} features les plus importantes (sur {len(imp)} au total)",
                             labels={"x": "Importance", "y": "Feature"},
                             color=imp_top.values, color_continuous_scale="Greens")
            fig_imp.update_layout(yaxis_title="", coloraxis_showscale=False,
                                  height=max(300, top_n * 40))
            st.plotly_chart(fig_imp, use_container_width=True)

else:
    st.info("Configurez et entraînez un modèle via la barre latérale.")
