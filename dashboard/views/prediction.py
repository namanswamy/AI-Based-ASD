import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

BASE = os.path.join(os.path.dirname(__file__), "..", "..")


def run():
    st.title("ASD Risk Prediction")
    st.markdown("Fill in the screening questionnaire and clinical indicators below, then click **Predict**.")

    # --- Load model and dataset defaults ---
    model_path = os.path.join(BASE, "models", "saved", "best_model.joblib")
    data_path = os.path.join(BASE, "data", "processed", "multimodal_dataset.csv")

    if not os.path.exists(model_path):
        st.error("No trained model found. Run the training pipeline first.")
        return

    model = joblib.load(model_path)
    feature_names = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else []

    # Load dataset to get medians for default values
    try:
        df = pd.read_csv(data_path)
        # Sanitize column names to match model features
        df.columns = df.columns.str.replace(" ", "_").str.replace("/", "_").str.replace(".", "_")
        if "Class_ASD" in df.columns:
            df = df.drop(columns=["Class_ASD"], errors="ignore")
        medians = df.median(numeric_only=True)
    except Exception:
        medians = pd.Series(dtype=float)

    def default(name):
        return float(medians.get(name, 0))

    # ---- Form ----
    st.markdown("---")

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.subheader("AQ-10 Screening Questions")
        st.caption("Rate each item from 0 (not at all) to 1 (yes).")
        aq = {}
        labels = {
            "A1_Score": "Prefers doing things alone",
            "A2_Score": "Prefers the same routine",
            "A3_Score": "Difficulty imagining what a character thinks/feels",
            "A4_Score": "Easily absorbed in one thing",
            "A5_Score": "Notices small sounds others don't",
            "A6_Score": "Notices patterns in things",
            "A7_Score": "Difficulty reading between the lines",
            "A8_Score": "Difficulty understanding social rules",
            "A9_Score": "Difficulty working out people's intentions",
            "A10_Score": "Difficulty making new friends",
        }
        for key, label in labels.items():
            aq[key] = st.selectbox(label, [0, 1], index=0, key=key)

    with col_right:
        st.subheader("Demographics")
        age = st.number_input("Age", min_value=1, max_value=100, value=10, key="demo_age")
        gender = st.selectbox("Gender", ["Male", "Female"], key="demo_gender")
        ethnicity = st.number_input("Ethnicity (encoded)", min_value=0, max_value=15, value=0, key="demo_eth")
        jaundice = st.selectbox("Born with jaundice?", ["No", "Yes"], key="demo_jaundice")
        family_asd = st.selectbox("Family member with ASD?", ["No", "Yes"], key="demo_fam")

        st.markdown("---")
        st.subheader("Clinical Scores")
        communication = st.slider("Communication Score", 0.0, 30.0, 15.0, 0.5, key="clin_comm")
        social = st.slider("Social Interaction Score", 0.0, 30.0, 15.0, 0.5, key="clin_social")
        behavioral = st.slider("Behavioral Score", 0.0, 30.0, 15.0, 0.5, key="clin_behav")
        cognitive = st.slider("Cognitive Score", 0.0, 30.0, 15.0, 0.5, key="clin_cog")
        motor = st.slider("Motor Skills Score", 0.0, 30.0, 15.0, 0.5, key="clin_motor")
        symptom_int = st.slider("Symptom Intensity", 0.0, 10.0, 5.0, 0.5, key="clin_symp")

    st.markdown("---")

    if st.button("Predict ASD Risk", type="primary", use_container_width=True):
        # Build feature vector in model feature order
        values = {}
        # AQ scores
        for k, v in aq.items():
            values[k] = float(v)
        # Demographics
        values["age"] = float(age)
        values["Age"] = float(age)
        values["gender"] = 1.0 if gender == "Male" else 0.0
        values["Gender"] = 1.0 if gender == "Male" else 0.0
        values["ethnicity"] = float(ethnicity)
        values["jundice"] = 1.0 if jaundice == "Yes" else 0.0
        values["austim"] = 1.0 if family_asd == "Yes" else 0.0
        values["result"] = float(sum(aq.values()))
        # Clinical
        values["Communication_Score"] = communication
        values["Social_Interaction_Score"] = social
        values["Behavioral_Score"] = behavioral
        values["Cognitive_Score"] = cognitive
        values["Motor_Skills_Score"] = motor
        values["Symptom_Intensity"] = symptom_int

        # Fill remaining features with dataset medians
        row = []
        for feat in feature_names:
            if feat in values:
                row.append(values[feat])
            else:
                row.append(default(feat))

        X = np.array(row).reshape(1, -1)

        try:
            prob = model.predict_proba(X)[0][1]
            pred = model.predict(X)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        # Determine risk level
        if prob < 0.3:
            risk, color = "Low Risk", "green"
        elif prob < 0.6:
            risk, color = "Moderate Risk", "orange"
        else:
            risk, color = "High Risk", "red"

        st.markdown("---")
        st.subheader("Results")

        r1, r2, r3 = st.columns(3)
        r1.metric("ASD Probability", f"{prob:.1%}")
        r2.metric("Risk Level", risk)
        r3.metric("Prediction", "ASD" if pred == 1 else "No ASD")

        # Visual progress bar
        st.progress(float(min(prob, 1.0)))

        if risk == "High Risk":
            st.error(
                "The model predicts a **high probability** of ASD traits. "
                "This is a screening tool only — please consult a qualified professional."
            )
        elif risk == "Moderate Risk":
            st.warning(
                "The model predicts a **moderate probability** of ASD traits. "
                "Further assessment may be warranted."
            )
        else:
            st.success(
                "The model predicts a **low probability** of ASD traits."
            )

        st.caption("Disclaimer: This tool is for research purposes only and does not constitute a medical diagnosis.")
