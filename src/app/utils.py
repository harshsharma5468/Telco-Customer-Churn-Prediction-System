# src/app/utils.py
"""
Helper functions that keep the Streamlit UI tidy.
All functions are pure (no side‑effects) and can be unit‑tested.
"""

import json
import joblib
import pandas as pd
import numpy as np
import shap
from pathlib import Path
from typing import Tuple, Dict, Any

from src.utils.logger import get_logger
logger = get_logger(__name__)

# ----------------------------------------------------------------------
# 1️⃣ Load the JSON metrics file → pick the best model name
# ----------------------------------------------------------------------
def get_best_model_name(model_dir: Path) -> str:
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Metrics file not found at {metrics_path}")
    with metrics_path.open() as f:
        data = json.load(f)
    return data["selected_model"]


# ----------------------------------------------------------------------
# 2️⃣ Load the requested pipeline (joblib)
# ----------------------------------------------------------------------
def load_pipeline(model_dir: Path, model_name: str):
    """Load a persisted sklearn/XGBoost pipeline."""
    pipeline_path = model_dir / f"{model_name}_pipeline.joblib"
    if not pipeline_path.is_file():
        raise FileNotFoundError(f"Pipeline not found at {pipeline_path}")
    pipeline = joblib.load(pipeline_path)
    logger.info(f"Loaded pipeline `{model_name}` from {pipeline_path}")
    return pipeline


# ----------------------------------------------------------------------
# 3️⃣ Convert a dict of UI inputs into a one‑row DataFrame that matches
#     the training schema (including all engineered columns).  The UI
#     only asks for the *raw* columns – the pipeline will do the
#     engineering & preprocessing.
# ----------------------------------------------------------------------
def build_input_dataframe(raw_input: Dict[str, Any]) -> pd.DataFrame:
    """
    Parameters
    ----------
    raw_input : dict
        Keys are the raw column names (same as in the original CSV, e.g.
        `tenure`, `MonthlyCharges`, `Contract`, …).  Values are whatever the
        user typed / selected.

    Returns
    -------
    pd.DataFrame
        A single‑row DataFrame (shape = (1, n_features_raw)).
    """
    # Ensure proper dtypes (categorical → pandas.Categorical)
    df = pd.DataFrame([raw_input])

    # Cast manually known categorical columns to `category` dtype.
    # This mirrors the logic in `src/data/cleaning.py`.
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = df[col].astype("category")

    # The pipeline will handle the rest (feature engineering, encoding,
    # scaling).  No further preprocessing needed here.
    return df


# ----------------------------------------------------------------------
# 4️⃣ SHAP explainer – we cache the explainer per model so it is built
#     only once per session (Streamlit `@st.cache_resource` does that).
# ----------------------------------------------------------------------
def get_shap_explainer(pipeline):
    """
    XGBoost pipelines expose the underlying booster through
    `pipeline.named_steps["model"].get_booster()`.  For scikit‑learn
    models we fall back to KernelExplainer (slower, but works for any
    estimator).  This function is deliberately small because the
    `shap` library does most of the heavy lifting.
    """
    model = pipeline.named_steps["model"]
    # Try to obtain a native XGBoost Booster – fast TreeExplainer.
    if hasattr(model, "get_booster"):
        # We need a background dataset; we’ll use a small random sample
        # from the training data (the pipeline itself can provide it).
        # In production you could store a pre‑computed background array.
        raise NotImplementedError("XGBoost SHAP fast path not implemented yet.")
    else:
        # Generic model – use KernelExplainer (model‑agnostic).
        # We will later feed a single row, so the explainer can be cheap.
        return shap.KernelExplainer(pipeline.predict_proba, shap.sample(pipeline.transform(np.zeros((1, pipeline.named_steps["preprocess"].n_features_in_))), 100))

# ----------------------------------------------------------------------
# 5️⃣ Helper to render SHAP values for a single prediction
# ----------------------------------------------------------------------
def plot_shap_single(explainer, input_row: pd.DataFrame, model, top_n: int = 5):
    """
    Returns a Matplotlib Figure with the top‑N SHAP values for the given
    row.  The caller can embed it directly via `st.pyplot(fig)`.
    """
    shap_values = explainer.shap_values(input_row)
    # For binary classification XGBoost returns a list of two arrays
    # (one per class).  We are interested in the **positive class**.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]   # index 1 = churn probability

    # Reduce to top‑N absolute contributions
    shap_abs = np.abs(shap_values[0])
    top_idx = np.argsort(shap_abs)[-top_n:][::-1]

    # Build a small figure
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 2 + top_n * 0.4))
    feature_names = input_row.columns
    ax.barh(
        np.arange(top_n),
        shap_values[0][top_idx],
        align="center",
        color=["#ff7f0e" if v > 0 else "#1f77b4" for v in shap_values[0][top_idx]],
    )
    ax.set_yticks(np.arange(top_n))
    ax.set_yticklabels(feature_names[top_idx])
    ax.set_xlabel("SHAP value (impact on churn probability)")
    ax.invert_yaxis()
    ax.axvline(0, color="k", linewidth=0.7)
    fig.tight_layout()
    return fig
