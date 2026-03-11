import streamlit as st
import pandas as pd


def run():

    st.header("Dataset Explorer")

    df = pd.read_csv(
        "data/processed/multimodal_dataset.csv"
    )

    st.write("Dataset shape:", df.shape)

    st.dataframe(df.head())

    st.subheader("Feature Distribution")

    column = st.selectbox(

        "Select feature",

        df.columns
    )

    st.bar_chart(df[column])