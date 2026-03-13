import pandas as pd
import streamlit as st


@st.cache_data
def load_clean_data():
    return pd.read_csv("data/house_prices_clean.csv")


@st.cache_data
def load_raw_data():
    return pd.read_csv("data/train.csv")


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
