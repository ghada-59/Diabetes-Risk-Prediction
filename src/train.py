from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from data_loader import load_raw_data
from preprocessing import create_preprocessing_pipeline, split_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

def build_models() -> dict[str, Pipeline]:
    """Create candidate preprocessing + classifier pipelines."""
    return {
        "Naive_Bayes": Pipeline([("preprocessor", create_preprocessing_pipeline()), ("classifier", GaussianNB())]),
        "KNN": Pipeline([("preprocessor", create_preprocessing_pipeline()), ("classifier", KNeighborsClassifier(n_neighbors=5))]),
        "Decision_Tree": Pipeline([("preprocessor", create_preprocessing_pipeline()), ("classifier", DecisionTreeClassifier(max_depth=4, min_samples_split=5, random_state=42))]),
        "Random_Forest": Pipeline([("preprocessor", create_preprocessing_pipeline()), ("classifier", RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=4, random_state=42))]),
    }

def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    """Train candidates and compare them with stratified 5-fold recall."""
    models = build_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    print("=== CROSS-VALIDATION EVALUATION (5-Fold) ===")
    for name, pipeline in models.items():
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="recall")
        results.append({"Model": name, "Mean_Recall": scores.mean(), "Std_Recall": scores.std()})
        print(f"Model [{name:<15}] -> Mean Recall: {scores.mean():.3f} (+/- {scores.std():.3f})")
        pipeline.fit(X_train, y_train)
    results_df = pd.DataFrame(results).sort_values(["Mean_Recall", "Std_Recall"], ascending=[False, True]).reset_index(drop=True)
    return models, results_df

def save_best_model(trained_models: dict[str, Pipeline], cv_results: pd.DataFrame) -> str:
    """Save the model selected from measured cross-validation recall."""
    if cv_results.empty:
        raise ValueError("No cross-validation results are available.")
    best_name = str(cv_results.iloc[0]["Model"])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{best_name}.joblib"
    joblib.dump(trained_models[best_name], model_path)
    print(f"[INFO] Selected model: {best_name}")
    print(f"[INFO] Saved pipeline to: {model_path}")
    return best_name

if __name__ == "__main__":
    df = load_raw_data()
    X_train, _, y_train, _ = split_data(df)
    trained_models, cv_results = train_models(X_train, y_train)
    save_best_model(trained_models, cv_results)