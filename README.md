# 🩺 Diabetes Risk Prediction — Machine Learning

An academic machine-learning project exploring a tabular diabetes-classification workflow. The tracked file `data/raw/medical_decision_dataset.csv` contains 200 rows, and its provenance is not documented in this repository. It should not be described as the Pima Indians Diabetes dataset unless the source is verified.

## 🎯 Objective

The project studies how preprocessing, feature handling, model training and evaluation can be combined for a binary classification task.

It is an **educational machine-learning project**, not a validated medical diagnostic tool.

## ⚙️ Pipeline

The workflow includes:

- missing-value handling / imputation
- numerical feature scaling
- train/test splitting
- stratified cross-validation
- model training and comparison
- evaluation with classification metrics
- Streamlit visualization

The repository uses scikit-learn pipelines to keep preprocessing steps organized with model training.

## 🛠️ Technologies

Python · Pandas · NumPy · scikit-learn · Streamlit · Matplotlib

## ⚠️ Limitations

The dataset is small and its source is currently undocumented in the repository. Its labels and data provenance should be verified before interpreting evaluation results. Model performance is educational only and must not be interpreted as clinical performance or generalized to other populations.

## 🚀 Run

### Streamlit interface

\`\`\`bash
python -m pip install -r requirements.txt
python src/train.py
streamlit run app.py
\`\`\`

The Streamlit app requires a trained `.joblib` pipeline under `models/`. Model artifacts are not committed; run the training script first.

### Optional React workbench

Install the API dependencies and start the local API:

\`\`\`bash
python -m pip install -r requirements-react-api.txt
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
\`\`\`

In a second terminal:

\`\`\`bash
cd frontend
npm install
npm run dev
\`\`\`

Open the local URL printed by Vite. The React workbench also needs a trained `.joblib` file under `models/`. This project is for education, not clinical decisions.
