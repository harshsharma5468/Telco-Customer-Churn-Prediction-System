"""
Inference helpers.
Single source of truth for feature preparation.
"""

from pathlib import Path
import json
import joblib
import pandas as pd

from src.features.engineering import engineer_features


def load_best_model():
    root = Path(__file__).resolve().parents[2]
    model_dir = root / "models"

    with open(model_dir / "metrics.json") as f:
        best = json.load(f)["selected_model"]

    return joblib.load(model_dir / f"{best}_pipeline.joblib")


def _prepare_features(model, raw_df: pd.DataFrame):
    """
    Applies feature engineering + aligns columns exactly as training expects.
    Returns transformed matrix and feature names.
    """
    # 1️⃣ Feature engineering
    df_fe = engineer_features(raw_df)

    # 2️⃣ Align to preprocessor expectations
    preprocess = model.named_steps["preprocess"]
    expected = preprocess.feature_names_in_

    for col in expected:
        if col not in df_fe:
            df_fe[col] = 0

    df_fe = df_fe[expected]

    # 3️⃣ Transform
    X = preprocess.transform(df_fe)
    feature_names = preprocess.get_feature_names_out()

    return X, feature_names


def predict_churn(raw_df: pd.DataFrame) -> pd.Series:
    model = load_best_model()
    X, _ = _prepare_features(model, raw_df)
    probs = model.named_steps["model"].predict_proba(X)[:, 1]
    return pd.Series(probs, name="churn_probability")
