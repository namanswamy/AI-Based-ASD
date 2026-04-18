import sys
import os

# Ensure project root is on the path so page imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

st.set_page_config(
    page_title="ASD AI Research Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------- Custom CSS ---------------
st.markdown("""
<style>
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0e1117;
        min-width: 260px;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 1.05rem;
        padding: 4px 0;
    }
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #1a1c23;
        border: 1px solid #2d2f36;
        border-radius: 10px;
        padding: 16px 20px;
    }
    /* Dataframe header */
    .stDataFrame thead th {
        background-color: #1e1e2e !important;
    }
</style>
""", unsafe_allow_html=True)

# --------------- Sidebar ---------------
st.sidebar.markdown("## ASD Research Platform")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Home", "Prediction", "Model Analysis", "Dataset Explorer"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption("AI-Based ASD Screening Tool  \nFor research purposes only.")

# --------------- Pages ---------------
if page == "Home":
    st.title("ASD AI Research Platform")
    st.markdown(
        "An end-to-end machine learning system for **Autism Spectrum Disorder** "
        "risk screening using multimodal data."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Models Trained", "4")
    col2.metric("Dataset Size", "2,100 samples")
    col3.metric("Features", "78")
    col4.metric("Best Accuracy", "100%")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Data Modalities")
        st.markdown("""
- **Questionnaire (AQ-10)** - 10-item behavioral screening
- **Clinical Indicators** - communication, social, cognitive, motor scores
- **Eye-Tracking Signals** - CARS scores and gaze features
- **Motor Behavior** - motion statistics from sensor data
        """)
    with c2:
        st.subheader("Models")
        st.markdown("""
- **Random Forest** - ensemble of decision trees
- **XGBoost** - gradient-boosted trees
- **LightGBM** - fast gradient boosting
- **SVM** - support vector machine
        """)

    st.markdown("---")
    st.subheader("Quick Start")
    st.markdown(
        "Use the **sidebar** to navigate between pages:\n"
        "- **Prediction** - enter patient data and get an ASD risk estimate\n"
        "- **Model Analysis** - compare model performance, ROC curves, and SHAP explanations\n"
        "- **Dataset Explorer** - explore distributions, correlations, and statistics"
    )

elif page == "Prediction":
    from views.prediction import run
    run()

elif page == "Model Analysis":
    from views.model_analysis import run
    run()

elif page == "Dataset Explorer":
    from views.dataset_explorer import run
    run()
