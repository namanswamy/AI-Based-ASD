import streamlit as st
import pandas as pd


def run():

    st.header("Model Performance")

    try:

        df = pd.read_csv(
            "outputs/reports/model_comparison.csv"
        )

        st.dataframe(df)

    except:

        st.warning("Run evaluation pipeline first")

    st.subheader("ROC Curve")

    try:

        st.image("outputs/plots/roc_curve.png")

    except:

        st.warning("ROC plot not found")

    st.subheader("Feature Importance")

    try:

        st.image("outputs/plots/feature_importance.png")

    except:

        st.warning("Feature importance plot not found")