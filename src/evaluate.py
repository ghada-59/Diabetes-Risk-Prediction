import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
)

from data_loader import load_raw_data
from preprocessing import split_data
from train import train_models


def evaluate_all_models(pipelines: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Evaluate all trained model pipelines on the test set using clinical 
    and general performance metrics (Accuracy, Recall, Confusion Matrix).

    Parameters:
    -----------
    pipelines : dict
        A dictionary mapping model names to their corresponding trained Scikit-Learn pipelines.
    X_test : pd.DataFrame
        The feature matrix for the test split.
    y_test : pd.Series
        The target vector for the test split.

    Returns:
    --------
    pd.DataFrame
        A summary DataFrame containing model names, accuracies, and recall scores, 
        sorted by descending recall performance.
    """
    summary_results = []

    print("\n" + "=" * 50)
    print(" DETAILED MODEL EVALUATION ")
    print("=" * 50)

    for name, pipeline in pipelines.items():
        # Generate predictions on unseen test data
        y_pred = pipeline.predict(X_test)

        # Compute key metrics
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        summary_results.append({
            "Model": name, 
            "Accuracy": round(acc, 3), 
            "Recall": round(rec, 3)
        })

        # Log detailed report per model
        print(f"\n--- Model: {name} ---")
        print(f"Accuracy : {acc:.3f}")
        print(f"Recall   : {rec:.3f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Healthy (0)", "Diabetic (1)"]))

    # Aggregate and sort summary results prioritizing clinical recall
    df_summary = pd.DataFrame(summary_results).sort_values(by="Recall", ascending=False)
    return df_summary


if __name__ == "__main__":
    df = load_raw_data()
    X_train, X_test, y_train, y_test = split_data(df)
    pipelines = train_models(X_train, y_train)
    summary = evaluate_all_models(pipelines, X_test, y_test)

    print("\n" + "=" * 50)
    print(" COMPARATIVE PERFORMANCE SUMMARY ")
    print("=" * 50)
    print(summary.to_string(index=False))