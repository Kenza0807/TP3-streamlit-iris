import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def hist_target(df: pd.DataFrame, col: str):
    return px.histogram(df, x=col, nbins=40, title=f"Distribution de {col}")


def scatter_pred_vs_real(y_test, y_pred):
    df = pd.DataFrame({"Réel": y_test, "Prédit": y_pred})
    fig = px.scatter(
        df, x="Réel", y="Prédit",
        title="Valeurs Prédites vs Réelles",
        opacity=0.6
    )
    max_val = max(df["Réel"].max(), df["Prédit"].max())
    fig.add_shape(
        type="line", x0=0, y0=0, x1=max_val, y1=max_val,
        line=dict(color="red", dash="dash")
    )
    return fig


def feature_importance_chart(importances: pd.Series, top_n: int = 15):
    top = importances.nlargest(top_n).sort_values()
    fig = px.bar(
        x=top.values, y=top.index,
        orientation="h",
        title=f"Top {top_n} features les plus importantes",
        labels={"x": "Importance", "y": "Feature"}
    )
    return fig


def corr_matrix(df: pd.DataFrame):
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return None, "Pas assez de colonnes numériques."
    corr = num_df.corr()
    fig = px.imshow(
        corr,
        title="Matrice de corrélation",
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
        text_auto=True
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig, None
