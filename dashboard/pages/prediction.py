import streamlit as st
import requests


def run():

    st.header("ASD Risk Prediction")

    st.write(
        "Answer the following behavioral screening questions."
    )

    q1 = st.slider("Difficulty with social interaction", 0, 5, 2)
    q2 = st.slider("Prefers routines and repetition", 0, 5, 2)
    q3 = st.slider("Avoids eye contact", 0, 5, 2)
    q4 = st.slider("Sensitivity to sound/light", 0, 5, 2)
    q5 = st.slider("Difficulty understanding emotions", 0, 5, 2)

    st.subheader("Behavioral Indicators")

    q6 = st.slider("Repetitive movements", 0, 5, 2)
    q7 = st.slider("Focus on specific interests", 0, 5, 2)
    q8 = st.slider("Difficulty adapting to change", 0, 5, 2)

    st.subheader("Demographic Information")

    age = st.number_input("Age", 1, 100, 10)
    gender = st.selectbox("Gender", ["Male", "Female"])

    gender_val = 1 if gender == "Male" else 0

    features = [
        q1, q2, q3, q4, q5,
        q6, q7, q8,
        age,
        gender_val
    ]

    while len(features) < 47:
        features.append(0)

    if st.button("Predict ASD Risk"):

        try:

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"features": features}
            )

            result = response.json()

            prob = result["asd_probability"]
            risk = result["risk_level"]

            st.success(
                f"ASD Probability: {prob:.3f}"
            )

            if risk == "High Risk":
                st.error(f"Risk Level: {risk}")

            elif risk == "Moderate Risk":
                st.warning(f"Risk Level: {risk}")

            else:
                st.info(f"Risk Level: {risk}")

        except Exception as e:

            st.error(
                "Could not connect to the prediction API. "
                "Make sure the FastAPI server is running."
            )
