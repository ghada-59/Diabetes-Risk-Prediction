"""Local API adapter for the educational tabular diabetes ML project."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
app = FastAPI(title="Public Health Analytics API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class Features(BaseModel):
    pregnancies: int = Field(ge=0, le=20)
    glucose: float = Field(ge=0, le=400)
    blood_pressure: float = Field(ge=0, le=250)
    skin_thickness: float = Field(ge=0, le=150)
    insulin: float = Field(ge=0, le=1500)
    bmi: float = Field(ge=0, le=100)
    diabetes_pedigree_function: float = Field(ge=0, le=5)
    age: int = Field(ge=18, le=120)


@lru_cache(maxsize=1)
def _load_model():
    candidates = sorted(MODEL_DIR.glob("*.joblib"))
    if not candidates:
        return None, None
    path = candidates[0]
    try:
        return joblib.load(path), path.name
    except Exception as exc:
        raise RuntimeError("Could not load model " + path.name + ": " + str(exc)) from exc


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "Public Health Analytics API"}


@app.get("/api/status")
def status():
    candidates = sorted(MODEL_DIR.glob("*.joblib")) if MODEL_DIR.is_dir() else []
    return {
        "model_available": bool(candidates),
        "model_name": candidates[0].stem if candidates else None,
        "training_command": "python src/train.py",
        "scope": "Educational dataset-based classification; not a clinical risk calculator.",
    }


@app.post("/api/predict")
def predict(features: Features):
    try:
        model, model_name = _load_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model was found in models/*.joblib. From the project root, run python src/train.py, then restart the API.",
        )
    row = pd.DataFrame([{
        "Pregnancies": features.pregnancies,
        "Glucose": features.glucose,
        "BloodPressure": features.blood_pressure,
        "SkinThickness": features.skin_thickness,
        "Insulin": features.insulin,
        "BMI": features.bmi,
        "DiabetesPedigreeFunction": features.diabetes_pedigree_function,
        "Age": features.age,
    }], columns=FEATURES)
    if not np.isfinite(row.to_numpy(dtype=float)).all():
        raise HTTPException(status_code=422, detail="All input values must be finite.")
    try:
        predicted = int(model.predict(row)[0])
        probabilities = model.predict_proba(row)[0] if hasattr(model, "predict_proba") else None
        positive_probability = None
        if probabilities is not None:
            classes = list(getattr(model, "classes_", [0, 1]))
            if 1 in classes:
                positive_probability = float(probabilities[classes.index(1)])
        return {
            "prediction": predicted,
            "class_label": "Class 1" if predicted == 1 else "Class 0",
            "class_1_probability": positive_probability,
            "model_name": model_name,
            "input": row.iloc[0].to_dict(),
            "note": "Educational prediction on a small, population-specific dataset. This output is not a medical diagnosis or a clinically validated individual risk estimate.",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed: " + str(exc)) from exc
