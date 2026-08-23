import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

from data_loader import load_raw_data
from preprocessing import split_data, create_preprocessing_pipeline


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Train and evaluate multiple machine learning models using 
    Stratified 5-Fold Cross-Validation with recall optimization.

    Parameters:
    -----------
    X_train : pd.DataFrame
        The feature matrix for the training split.
    y_train : pd.Series
        The target vector for the training split.

    Returns:
    --------
    dict
        A dictionary mapping model names to their fully fitted Scikit-Learn pipelines.
    """
    preprocessor = create_preprocessing_pipeline()

    # Encapsulate preprocessing and classification into end-to-end pipelines for production
    models = {
        "Naive_Bayes": Pipeline([("preprocessor", preprocessor), ("classifier", GaussianNB())]),
        "KNN": Pipeline([("preprocessor", preprocessor), ("classifier", KNeighborsClassifier(n_neighbors=5))]),
        "Decision_Tree": Pipeline([("preprocessor", preprocessor), ("classifier", DecisionTreeClassifier(max_depth=4, min_samples_split=5, random_state=42))]),
        "Random_Forest": Pipeline([("preprocessor", preprocessor), ("classifier", RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=4, random_state=42))]),
    }

    print("=== CROSS-VALIDATION EVALUATION (5-Fold) ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, pipeline in models.items():
        # Evaluate model performance via cross-validation prioritizing recall
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="recall")
        print(f"Model [{name:<15}] -> Mean Recall (CV): {scores.mean():.3f} (+/- {scores.std():.3f})")
        
        # Fit the complete pipeline on the entire training set
        pipeline.fit(X_train, y_train)

    return models


def save_best_model(trained_models: dict, best_name: str = "Random_Forest"):
    """
    Serialize and save the best performing model pipeline to disk.

    Parameters:
    -----------
    trained_models : dict
        A dictionary containing all trained pipelines.
    best_name : str, default="Random_Forest"
        The key identifier of the best model to save.
    """
    os.makedirs("models", exist_ok=True)
    joblib.dump(trained_models[best_name], f"models/{best_name}.joblib")
    print(f"\nComplete model pipeline (Pipeline + {best_name}) successfully saved to models/{best_name}.joblib")


if __name__ == "__main__":
    df = load_raw_data()
    X_train, X_test, y_train, y_test = split_data(df)
    trained_pipelines = train_models(X_train, y_train)
    save_best_model(trained_pipelines, "Random_Forest")