import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="centered",
)

st.title("🩺 Medical Decision Support System")
st.subheader("Diabetes Risk Assessment Using Machine Learning")
st.write("Enter the patient's clinical parameters below to evaluate their risk probability.")

MODEL_PATH = "models/Random_Forest.joblib"

@st.cache_resource
def load_trained_pipeline(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

pipeline = load_trained_pipeline(MODEL_PATH)

if pipeline is None:
    st.error(
        f"❌ Model not found at path `{MODEL_PATH}`. "
        "Please execute `python src/train.py` first."
    )
    st.stop()

st.sidebar.header("Clinical Parameters")

def user_input_features():
    pregnancies = st.sidebar.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.sidebar.slider("Glucose Level (mg/dL)", min_value=40, max_value=300, value=120)
    blood_pressure = st.sidebar.slider("Blood Pressure (mm Hg)", min_value=40, max_value=140, value=70)
    skin_thickness = st.sidebar.slider("Skin Thickness (mm)", min_value=5, max_value=100, value=20)
    insulin = st.sidebar.slider("Insulin Level (mu U/ml)", min_value=10, max_value=800, value=80)
    bmi = st.sidebar.slider("Body Mass Index (BMI)", min_value=10.0, max_value=70.0, value=25.0)
    dpf = st.sidebar.slider("Diabetes Pedigree Function (DPF)", min_value=0.07, max_value=2.50, value=0.47)
    age = st.sidebar.slider("Age", min_value=18, max_value=100, value=33)

    data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

st.write("### Patient Profile Data")
st.dataframe(input_df)

if st.button("Evaluate Risk"):
    prediction = pipeline.predict(input_df)[0]
    probabilities = pipeline.predict_proba(input_df)[0]
    diabetes_probability = probabilities[1] * 100

    st.write("---")
    st.write("### Diagnostic Result")

    if prediction == 1:
        st.error(f"⚠️ **High Risk of Diabetes** (Probability: {diabetes_probability:.1f}%)")
        st.write("Recommendation: In-depth clinical follow-up and biological screening are advised.")
    else:
        st.success(f"✅ **Low Risk of Diabetes** (Probability: {diabetes_probability:.1f}%)")
        st.write("Recommendation: Patient constants are currently within standard ranges.")