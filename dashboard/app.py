import streamlit as st

st.set_page_config(
    page_title="ASD AI Research Dashboard",
    layout="wide"
)

st.title("ASD AI Research Platform")

st.markdown("""
This system provides AI-powered autism screening using:

• Questionnaire analysis  
• Motor behavior analysis  
• Eye-tracking signals  
• Multimodal machine learning models
""")

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Go to",

    [

        "Prediction",

        "Model Analysis",

        "Dataset Explorer"
    ]
)

if page == "Prediction":

    from pages.prediction import run
    run()

elif page == "Model Analysis":

    from pages.model_analysis import run
    run()

elif page == "Dataset Explorer":

    from .pages.dataset_explorer import run
    run()