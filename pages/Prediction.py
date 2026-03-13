import streamlit as st
import numpy as np
import pandas as pd
import logging

st.title("Page Prediction")

# Vérifs
if "model" not in st.session_state:
    st.warning("➡️ Aller sur la page Training pour entraîner le modèle d'abord.")
    st.stop()

model = st.session_state["model"]
le = st.session_state.get("label_encoder", None)
feature_cols = st.session_state.get("feature_cols", None)

if le is None or feature_cols is None:
    st.error("label_encoder ou feature_cols manquant. Réentraîne le modèle dans Training.")
    st.stop()

st.subheader("Entrer les caractéristiques de la fleur")

# --- Sliders Iris (adapter si tes colonnes ont d'autres noms)
# On suppose que feature_cols correspond bien aux 4 features (dans le bon ordre)
# Exemple classique : sepal_length, sepal_width, petal_length, petal_width
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8, 0.1)
    sepal_width  = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0, 0.1)

with col2:
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.0, 0.1)
    petal_width  = st.slider("Petal width (cm)", 0.1, 2.5, 1.3, 0.1)

# Construire X_new avec les bons noms de colonnes (TRÈS important)
inputs = [sepal_length, sepal_width, petal_length, petal_width]
X_new = pd.DataFrame([inputs], columns=feature_cols)

st.write("Données envoyées au modèle :")
st.dataframe(X_new, use_container_width=True)

if st.button("Prédire l'espèce", type="primary"):
    # Classe prédite (encodée 0/1/2)
    pred_class = int(model.predict(X_new)[0])

    # Décodage -> "setosa" / "versicolor" / "virginica"
    pred_label = le.inverse_transform([pred_class])[0]

    st.success(f"Espèce prédite : **{pred_label}**")
    
    logging.info("Prédiction effectuée")
    logging.info(f"Espèce prédite={pred_label} | inputs={X_new.to_dict(orient='records')[0]}")



    # Probabilités
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_new)[0]
        class_names = list(le.classes_)

        proba_df = pd.DataFrame({
            "Espèce": class_names,
            "Probabilité": proba
        }).sort_values("Probabilité", ascending=False)

        st.subheader("Probabilités par espèce")
        st.dataframe(proba_df, use_container_width=True)

        st.bar_chart(proba_df.set_index("Espèce"))
    else:
        st.info("Ce modèle ne supporte pas predict_proba(). Choisis Logistic Regression ou Random Forest.")