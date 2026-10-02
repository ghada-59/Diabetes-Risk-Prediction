from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "medical_decision_dataset.csv"
EXPECTED_FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
TARGET_COLUMN = "Outcome"
INVALID_ZERO_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

def load_raw_data(filepath: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the diabetes dataset and replace physiologically invalid zeros with NaN."""
    path = Path(filepath)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    required_columns = EXPECTED_FEATURES + [TARGET_COLUMN]
    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    df[INVALID_ZERO_COLUMNS] = df[INVALID_ZERO_COLUMNS].replace(0, np.nan)
    print(f"[INFO] Loaded dataset: {path}")
    print(f"[INFO] Dataset dimensions: {df.shape}")
    print("[INFO] Missing values after invalid-zero handling:")
    print(df[required_columns].isna().sum())
    return df

if __name__ == "__main__":
    load_raw_data()