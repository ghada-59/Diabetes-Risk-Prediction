import os
import numpy as np
import pandas as pd


def load_raw_data(filepath: str = "data/raw/medical_decision_dataset.csv") -> pd.DataFrame:
    """
    Load the raw medical decision dataset from disk and handle biological 
    anomalies by converting physiological zero values into missing values (NaN).

    Parameters:
    -----------
    filepath : str
        The relative or absolute path to the raw CSV dataset.

    Returns:
    --------
    pd.DataFrame
        The cleaned DataFrame with invalid zeros replaced by NaN.

    Raises:
    -------
    FileNotFoundError
        If the specified dataset path does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at target path: {filepath}")

    # Load dataset into memory
    df = pd.read_csv(filepath)

    # Define clinical features where a value of 0 violates physiological laws
    invalid_zero_cols = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]

    # Convert physiological zeros to NaN to prevent skewing future imputation steps
    df[invalid_zero_cols] = df[invalid_zero_cols].replace(0, np.nan)

    # Log ingestion summary
    print("[INFO] Data ingestion completed successfully.")
    print(f"[INFO] Dataset dimensions: {df.shape}")
    print("\n[INFO] Missing values summary per feature:")
    print(df.isnull().sum())

    return df


if __name__ == "__main__":
    load_raw_data()