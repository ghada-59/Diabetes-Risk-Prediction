from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loader import load_raw_data
from preprocessing import split_data
from train import train_models


def evaluate_all_models(pipelines: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Evaluate fitted pipelines on the held-out test set."""
    summary_results = []

    for name, pipeline in pipelines.items():
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        summary_results.append({
            "Model": name,
            "Accuracy": round(acc, 3),
            "Recall": round(rec, 3),
        })

        print(f"\n--- Model: {name} ---")
        print(f"Accuracy : {acc:.3f}")
        print(f"Recall   : {rec:.3f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(
            y_test,
            y_pred,
            target_names=["Healthy (0)", "Diabetic (1)"],
            zero_division=0,
        ))

    return pd.DataFrame(summary_results).sort_values(
        by="Recall", ascending=False
    ).reset_index(drop=True)


if __name__ == "__main__":
    df = load_raw_data()
    X_train, X_test, y_train, y_test = split_data(df)
    pipelines, cv_results = train_models(X_train, y_train)

    print("\n=== CROSS-VALIDATION MODEL SELECTION ===")
    print(cv_results.to_string(index=False))

    summary = evaluate_all_models(pipelines, X_test, y_test)
    print("\n=== HELD-OUT TEST SET SUMMARY ===")
    print(summary.to_string(index=False))
