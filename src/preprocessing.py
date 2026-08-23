from typing import Tuple
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def split_data(
    df: pd.DataFrame, target_col: str = "Outcome", test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the DataFrame into feature matrices and target vectors, 
    then partition them into stratified training and testing sets.

    Parameters:
    -----------
    df : pd.DataFrame
        The input pandas DataFrame containing features and the target column.
    target_col : str, default="Outcome"
        The name of the target/label column.
    test_size : float, default=0.2
        The proportion of the dataset to include in the test split.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test splits.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Stratify by y to maintain class distribution across splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


def create_preprocessing_pipeline() -> Pipeline:
    """
    Construct a scikit-learn Pipeline for missing value imputation 
    and feature scaling to prevent data leakage.

    Returns:
    --------
    Pipeline
        A scikit-learn Pipeline chaining median imputation and standard scaling.
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return pipeline