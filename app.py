from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diabetes Risk Prediction", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Classification Demo")
st.subheader("Machine Learning on the Pima Indians Diabetes Dataset")
st.write(
    "Enter the dataset features below to obtain a model prediction and estimated class probability. "
    "This is an educational demonstration, not a clinical diagnostic tool."
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"


def find_model() -> Path | None:
    """Return a trained pipeline from the models directory."""
    candidates = sorted(MODEL_DIR.glob("*.joblib"))
    return candidates[0] if candidates else None


@st.cache_resource
def load_trained_pipeline(path: str):
    return joblib.load(path)


model_path = find_model()
if model_path is None:
    st.error(
        "No trained model was found. Run `python src/train.py` from the project root, "
        "then restart the Streamlit app."
    )
    st.stop()

pipeline = load_trained_pipeline(str(model_path))
st.caption(f"Loaded model: {model_path.stem}")

st.sidebar.header("Input Features")

pregnancies = st.sidebar.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.sidebar.slider("Glucose Level (mg/dL)", min_value=40, max_value=300, value=120)
blood_pressure = st.sidebar.slider("Blood Pressure (mm Hg)", min_value=40, max_value=140, value=70)
skin_thickness = st.sidebar.slider("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
insulin = st.sidebar.slider("Insulin Level (mu U/ml)", min_value=0, max_value=800, value=80)
bmi = st.sidebar.slider("Body Mass Index (BMI)", min_value=10.0, max_value=70.0, value=25.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function (DPF)", min_value=0.07, max_value=2.50, value=0.47)
age = st.sidebar.slider("Age", min_value=18, max_value=100, value=33)

input_df = pd.DataFrame({
    "Pregnancies": [pregnancies],
    "Glucose": [glucose],
    "BloodPressure": [blood_pressure],
    "SkinThickness": [skin_thickness],
    "Insulin": [insulin],
    "BMI": [bmi],
    "DiabetesPedigreeFunction": [dpf],
    "Age": [age],
})

st.write("### Input Data")
st.dataframe(input_df, use_container_width=True)

if st.button("Evaluate Risk"):
    prediction = pipeline.predict(input_df)[0]
    probabilities = pipeline.predict_proba(input_df)[0]
    diabetes_probability = probabilities[1] * 100

    st.write("---")
    st.write("### Model Prediction")
    if prediction == 1:
        st.warning(f"**Predicted Class: 1** — estimated class probability: {diabetes_probability:.1f}%")
    else:
        st.success(f"**Predicted Class: 0** — estimated class-1 probability: {diabetes_probability:.1f}%")

    st.caption(
        "The result is a dataset-based machine-learning output and must not be interpreted as a medical diagnosis."
    )
