# 🩺 Diabetes Risk Prediction: Medical Decision Support System

This project implements a comprehensive Machine Learning pipeline designed to assess diabetes risk in patients. Beyond a standard algorithmic classification exercise, this repository demonstrates a **rigorous engineering approach applied to healthcare data**, covering everything from clinical bias mitigation to the deployment of an interactive application.

---

## 🎯 Approach & Technical Choices

Working with medical data imposes strict constraints. Here is how I designed the architecture of this project to address them:

1. **Clinical Consistency (Data Preprocessing):**  
   Medical datasets often contain artifacts (e.g., a glucose level or BMI set to `0`). These physiologically impossible zeros distort model learning. I rigorously identified them, converted them to missing values (`NaN`), and subsequently imputed them using the median.
2. **Robustness & Data Leakage Prevention:**  
   To ensure the model does not cheat by inspecting test data, I enforced the strict use of Scikit-Learn `Pipeline` and `ColumnTransformer`, coupled with **Stratified 5-Fold Cross-Validation**. This prevents overfitting and guarantees prediction stability.
3. **Health-Oriented Metric Optimization (Recall):**  
   In medical diagnostics, a False Negative (labeling a sick patient as healthy) carries dramatic consequences. Consequently, I evaluated models by prioritizing **Recall (Sensitivity ≥ 0.90)** rather than simple overall accuracy.
4. **Continuous Deployment:**  
   I set up a local production environment featuring an interactive web interface (Streamlit) allowing healthcare professionals to simulate the impact of biological constants in real time.

---

## 📁 Project Modular Architecture

```text
diabetes-risk-prediction/
├── data/
│   └── raw/                    # Reference dataset (medical_decision_dataset.csv)
├── models/                     # Serialized and regularized models (.joblib)
├── notebooks/
│   └── 01_eda_and_preprocessing.ipynb # Exploratory analysis and graphical reasoning
├── src/
│   ├── __init__.py             # Package initialization and versioning
│   ├── data_loader.py          # Raw data ingestion and initial cleaning
│   ├── preprocessing.py        # Imputation and normalization pipelines
│   ├── train.py                # Model training, CV, and serialization
│   └── evaluate.py             # Clinical report generation (Confusion Matrix)
├── app.py                      # Interactive Streamlit interface
├── requirements.txt            # Environment dependencies
├── .gitignore                  # Files excluded from version control
└── README.md                   # Documentation

```

---

## 📊 Clinical and Algorithmic Performance Analysis

Selecting the final model was not arbitrary. It is based on a critical analysis of the results obtained during cross-validation and testing:

### 1. The Decision Tree Trap: 100% 🚨

The Decision Tree achieved **Accuracy = 1.000 (100%)** and **Recall = 1.000 (100%)**. Although this looks flawless on paper, in medical data science, a 100% score is often a "red flag."

* **The Explanation:** Since the test set contained only 40 patients (21 healthy, 19 diabetic), the tree simply memorized the data (overfitting). Faced with 100 new patients from a hospital tomorrow, its score would plummet. It is a model that is far too unstable for production deployment.

### 2. The True Hero: Random Forest 🏆

This is the model I selected for the final application. The metrics prove the solidity of this choice:

* **Recall (Cross-Validation):** 0.923 (92.3%)
* **Recall (Test Set):** 0.895 (89.5%)
* **Clinical Stability:** The gap between cross-validation and testing is minimal (less than 3%). This proves that the model generalizes exceptionally well to unseen data.
* **Confusion Matrix:** It correctly identified 17 out of 19 diabetic patients, generating only **2 False Negatives**. It also perfectly classified all 21 healthy patients (0 False Positives). This is the safe and reliable behavior expected of a clinical tool.

### 3. Naive Bayes and KNN 📉

These algorithms underperformed on the test set (Recall dropping to 68.4%), demonstrating their inability to capture the complexity of non-linear relationships among variables (glucose, insulin, BMI) as effectively as an ensemble method.

### 💡 Summary

The automated pipeline allowed me to test multiple hypotheses and confirm that the **Random Forest** offers the best trade-off between overall performance (95% Accuracy) and clinical safety (minimizing False Negatives). The currently saved `Random_Forest.joblib` model is robust and production-ready.

---

## 🚀 Installation & Deployment

To execute the training pipeline or launch the application locally on your machine, follow these steps:

```bash
# 1. Clone the repository
git clone
cd diabetes-risk-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Execute the training pipeline (Optional, the model is already pre-generated in /models)
python src/data_loader.py
python src/preprocessing.py
python src/train.py
python src/evaluate.py

# 4. Launch the decision support tool
$ streamlit run app.py

  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501