"""
SHAP explanation helpers.
Uses the SAME feature-preparation logic as prediction.
"""

import shap
import pandas as pd
import numpy as np

from src.models.predict import _prepare_features


def explain_prediction(model, raw_df: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """
    Generate SHAP explanations for a single prediction.
    """

    # ✅ SAME preparation as predict_churn
    X, feature_names = _prepare_features(model, raw_df)

    estimator = model.named_steps["model"]

    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    impacts = shap_values[0]

    return (
        pd.DataFrame({
            "feature": feature_names,
            "impact": impacts,
        })
        .assign(abs=lambda d: d["impact"].abs())
        .sort_values("abs", ascending=False)
        .head(top_n)
        .drop(columns="abs")
    )
