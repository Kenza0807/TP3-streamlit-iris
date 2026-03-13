import streamlit as st
import numpy as np
import pandas as pd
import logging

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, log_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns

st.title("Page Training")

# --- Chargement dataset depuis session
if "df" not in st.session_state:
    st.warning("➡️ Aller d’abord sur la page Data pour charger le dataset.")
    st.stop()

df = st.session_state["df"].copy()

# --- Fixer la colonne target automatiquement
#   1) Si 'species' existe -> target = species
#   2) Sinon, on prend la première colonne object (texte) comme target
if "species" in df.columns:
    target = "species"
else:
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if len(obj_cols) == 0:
        st.error("Aucune colonne cible texte trouvée (ex: species).")
        st.stop()
    target = obj_cols[0]

st.info(f"Colonne à prédire (fixée) : **{target}**")

# --- Choix modèle
st.subheader("Modèle")
model_name = st.selectbox("Choisir un modèle", ["Logistic Regression", "Random Forest"])

test_size = st.slider("Taille test", 0.1, 0.5, 0.2, 0.05)
random_state = st.number_input("random_state", value=42, step=1)

# --- X / y
X = df.drop(columns=[target])
y_raw = df[target].astype(str)

# Encodage labels texte -> 0,1,2
le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = list(le.classes_)

# --- Modèles
if model_name == "Logistic Regression":
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=int(random_state)))
    ])
else:
    n_estimators = st.slider("n_estimators", 50, 500, 200, 50)
    max_depth = st.slider("max_depth", 1, 50, 10, 1)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=int(random_state),
        n_jobs=-1
    )

# --- Split stratifié (important pour classification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=float(test_size),
    random_state=int(random_state),
    stratify=y
)

if st.button("Entraîner", type="primary"):
    model.fit(X_train, y_train)

    # --- Prédictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # --- Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    ll = log_loss(y_test, y_proba)

    st.success("Modèle entraîné !")
    st.session_state["model"] = model
    st.session_state["label_encoder"] = le
    st.session_state["target"] = target
    st.session_state["feature_cols"] = X.columns.tolist()

    st.subheader("Performances (test)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{acc:.3f}")
    c2.metric("F1 macro", f"{f1:.3f}")
    c3.metric("Log loss", f"{ll:.3f}")

    logging.info("Modèle entraîné")
    logging.info(f"Modèle: {model_name} | test_size={test_size} | random_state={random_state}")

    # (optionnel) log des métriques
    logging.info(f"Accuracy={acc:.3f} | F1_macro={f1:.3f} | LogLoss={ll:.3f}")

    st.success("Modèle entraîné !")
    st.session_state["model"] = model

    # --- Classification report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    st.dataframe(pd.DataFrame(report).T, use_container_width=True)

    # --- Confusion matrix (heatmap)
    st.subheader("Matrice de confusion")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    # --- Visualisation probas moyennes par classe
    st.subheader("Probabilités moyennes (test)")
    mean_proba = y_proba.mean(axis=0)
    proba_df = pd.DataFrame({"Classe": class_names, "Proba_moyenne": mean_proba}).sort_values("Proba_moyenne", ascending=False)
    st.bar_chart(proba_df.set_index("Classe"))

    # --- Interprétation modèle
    st.subheader("Interprétabilité")
    if model_name == "Random Forest":
        importances = model.feature_importances_
        imp_df = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values("Importance", ascending=False)

        st.write("**Importance des features** (Random Forest)")
        st.dataframe(imp_df, use_container_width=True)

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        sns.barplot(data=imp_df, x="Importance", y="Feature", ax=ax2)
        ax2.set_title("Feature importance")
        st.pyplot(fig2)

    else:
        # Logistic regression coefficients (multi-classe)
        coefs = model.named_steps["clf"].coef_
        coef_df = pd.DataFrame(coefs, columns=X.columns, index=class_names)
        st.write(" **Coefficients par classe** (Logistic Regression)")
        st.dataframe(coef_df, use_container_width=True)

        st.caption("Astuce : valeur absolue élevée ⇒ feature plus influente pour une classe donnée.")