import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from pathlib import Path

from utils.data_loader import load_clean_data
from utils.preprocessing import prepare_features

st.set_page_config(page_title="Prédiction", page_icon="🔮", layout="wide")

import logging
logger = logging.getLogger(__name__)

# ── Auth guard ────────────────────────────────────────────────────────────────
if not st.session_state.get("authenticated", False):
    st.warning(" Veuillez vous connecter depuis la page d'accueil.")
    st.stop()

logger.info(f"Page chargée — user : {st.session_state.get('username', '?')}")


with open(Path(__file__).parent.parent / "assets" / "style.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("3 — Interface de Prédiction")

# ── Chargement modèle ─────────────────────────────────────────────────────────
model = feat_names = target_col = task_type = label_map = None

if "model" in st.session_state:
    model      = st.session_state["model"]
    feat_names = st.session_state["feature_names"]
    target_col = st.session_state.get("target_col", "cible")
    task_type  = st.session_state.get("task_type", "régression")
    label_map  = st.session_state.get("label_map", {})
    st.success(f" Modèle **{st.session_state.get('model_name','')}** — cible : **{target_col}** — tâche : **{task_type}**")
else:
    try:
        with open("models/trained_model.pkl", "rb") as f:
            s = pickle.load(f)
        model      = s["model"]
        feat_names = s["features"]
        target_col = s.get("target", "cible")
        task_type  = s.get("task_type", "régression")
        label_map  = s.get("label_map", {})
        st.success(f" Modèle chargé depuis fichier — cible : **{target_col}**")
    except FileNotFoundError:
        st.warning(" Aucun modèle disponible. Allez d'abord sur **2 Training**.")

# ── Référence dataset — FIX : pas de "or" sur DataFrame ──────────────────────
if "df_uploaded" in st.session_state:
    df_ref = st.session_state["df_uploaded"]
else:
    df_ref = load_clean_data()

if model is not None:
    # Encoder le dataset de référence
    try:
        cols_needed = [c for c in feat_names if c in df_ref.columns]
        if target_col in df_ref.columns:
            X_ref, _ = prepare_features(df_ref[cols_needed + [target_col]], target=target_col)
        else:
            from sklearn.preprocessing import LabelEncoder
            tmp = df_ref[cols_needed].copy()
            for col in tmp.select_dtypes(include=["object","category"]).columns:
                tmp[col] = LabelEncoder().fit_transform(tmp[col].astype(str))
            tmp.fillna(tmp.median(numeric_only=True), inplace=True)
            X_ref = tmp
    except Exception:
        X_ref = df_ref[[c for c in feat_names if c in df_ref.columns]].copy()

    # ── Sliders ───────────────────────────────────────────────────────────────
    st.subheader(" Valeurs des features")

    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feat_names)
        top = imp.nlargest(10).index.tolist()
        show_all = st.checkbox(f"Afficher toutes les features ({len(feat_names)})", value=False)
        display  = feat_names if show_all else top
        if not show_all:
            st.caption(f"Affichage du **top 10** features les plus importantes sur {len(feat_names)} au total.")
    else:
        display = feat_names[:15]

    input_vals = {col: float(X_ref[col].median()) if col in X_ref.columns else 0.0 for col in feat_names}

    cols_ui = st.columns(2)
    for i, col in enumerate(display):
        if col not in X_ref.columns:
            continue
        vmin = float(X_ref[col].min())
        vmax = float(X_ref[col].max())
        vmed = float(X_ref[col].median())
        step = max(round((vmax - vmin) / 100, 4), 0.0001)
        with cols_ui[i % 2]:
            input_vals[col] = st.slider(col, vmin, vmax, vmed, step, key=f"s_{col}")

    st.divider()
    predict_btn = st.button("🔮 Prédire", type="primary")

    if predict_btn:
        input_df   = pd.DataFrame([{f: input_vals.get(f, 0.0) for f in feat_names}])
        prediction = model.predict(input_df)[0]

        st.markdown("---")

        # ── Classification ─────────────────────────────────────────────────
        if task_type == "classification":
            class_label = label_map.get(int(prediction), str(prediction)) if label_map else str(prediction)
            st.success(f"###  Classe prédite : **{class_label}**")

            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(input_df)[0]
                class_names = [label_map.get(int(c), str(c)) if label_map else str(c) for c in model.classes_]

                prob_df = pd.DataFrame({"Classe": class_names, "Probabilité": probas})
                prob_df = prob_df.sort_values("Probabilité", ascending=False)

                fig = px.bar(prob_df, x="Classe", y="Probabilité", color="Classe",
                             title="Probabilité d'appartenance à chaque classe",
                             text="Probabilité", range_y=[0, 1])
                fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
                fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#####  Détail des probabilités par classe")
                for _, row in prob_df.iterrows():
                    pct  = row["Probabilité"]
                    bar  = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
                    mark = " ← **Prédite**" if row["Classe"] == class_label else ""
                    st.markdown(f"**{row['Classe']}** — `{pct:.1%}`  `{bar}`{mark}")

        # ── Régression ─────────────────────────────────────────────────────
        else:
            st.success(f"###  **{target_col}** prédit : **{prediction:,.2f}**")
            if target_col in df_ref.columns:
                serie = pd.to_numeric(df_ref[target_col], errors="coerce").dropna()
                pct   = (serie < prediction).mean() * 100
                c1, c2, c3 = st.columns(3)
                c1.metric("Valeur prédite",  f"{prediction:,.2f}")
                c2.metric("Médiane dataset", f"{serie.median():,.2f}")
                c3.metric("Percentile",      f"{pct:.0f}ème")
                fig = px.histogram(serie, nbins=50, opacity=0.7,
                                   title=f"Valeur prédite vs distribution de {target_col}",
                                   labels={"value": target_col})
                fig.add_vline(x=prediction, line_dash="dash", line_color="red",
                              annotation_text=f"Prédit : {prediction:,.2f}",
                              annotation_position="top right")
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Valeurs utilisées")
        recap = pd.DataFrame({"Feature": display, "Valeur": [round(input_vals[f], 4) for f in display]})
        st.dataframe(recap, use_container_width=True, hide_index=True)

else:
    st.info(" Entraînez d'abord un modèle sur la page **2 Training**.")
