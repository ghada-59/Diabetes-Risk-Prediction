# 🩺 Diabetes Risk Prediction — Machine Learning

An academic machine-learning project using the Pima Indians Diabetes dataset to explore a complete tabular-data classification workflow.

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

The dataset is relatively small and comes from a specific population. Model performance from this dataset should not be interpreted as clinical performance or generalized to other populations without external validation.

## 🚀 Run

\`\`\`bash
pip install -r requirements.txt
streamlit run app.py
\`\`\`
