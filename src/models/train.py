"""
Training entry‑point.

* Loads the pre‑processed train / test splits (parquet files) produced by
  `src.data.preprocess_and_save`.
* Builds a scikit‑learn Pipeline for every model:
      preprocess (ColumnTransformer)  →  model
* Fits each pipeline, predicts on the test set, evaluates a set of
  business‑critical metrics, and persists the artefacts.
* Writes a JSON report (`models/metrics.json`) that lists every model’s
  performance and marks the best one (by ROC‑AUC).

Run with:
    python -m src.models.train
"""

import json
from pathlib import Path
import yaml
import joblib

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.utils.logger import get_logger
from src.features.preprocessing import identify_feature_types
from src.pipelines.churn_pipeline import get_churn_pipeline
from src.models.evaluate import evaluate_model

logger = get_logger(__name__)

def load_config() -> dict:
    """Load and resolve YAML configuration."""
    cfg_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    with cfg_path.open() as f:
        raw_cfg = yaml.safe_load(f)

    rnd_state = raw_cfg["data"]["random_state"]
    for model_cfg in raw_cfg["model"].values():
        for key, val in list(model_cfg.items()):
            if isinstance(val, str) and "${data.random_state}" in val:
                model_cfg[key] = rnd_state
    return raw_cfg

def process_labels(y_series: pd.Series, name: str) -> pd.Series:
    """
    Robustly converts Churn labels to 1/0.
    Handles 'Yes'/'No', numeric 1/0, and potential whitespace issues.
    """
    # 1. Standardize: Convert to string, strip spaces, and map
    y_mapped = y_series.astype(str).str.strip().map({"Yes": 1, "No": 0, "1": 1, "0": 0})
    
    # 2. Check for mapping failures (all NaNs)
    if y_mapped.isna().all():
        logger.error(f"Critical error mapping {name} labels. Unique values found: {y_series.unique()}")
        raise ValueError(f"Mapping failed for {name}. Ensure labels are 'Yes'/'No' or '1'/'0'.")

    # 3. Handle specific NaNs and cast to int
    y_final = y_mapped.fillna(0).astype(int)
    
    # 4. Final safety check: Do we have both classes?
    classes = np.unique(y_final)
    if len(classes) < 2:
        logger.error(f"{name} set only contains one class: {classes}. Target distribution: {y_final.value_counts().to_dict()}")
        raise ValueError(f"The {name} set needs at least two classes to train/evaluate a model.")
    
    return y_final

def main():
    logger.info("=== Model training started ===")
    cfg = load_config()

    # ------------------------------------------------------------------
    # 1️⃣ Load processed data with robust label processing
    # ------------------------------------------------------------------
    data_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    X_train = pd.read_parquet(data_dir / "X_train.parquet")
    X_test  = pd.read_parquet(data_dir / "X_test.parquet")
    
    y_train_raw = pd.read_parquet(data_dir / "y_train.parquet")["Churn"]
    y_test_raw  = pd.read_parquet(data_dir / "y_test.parquet")["Churn"]

    y_train = process_labels(y_train_raw, "train")
    y_test  = process_labels(y_test_raw, "test")
    
    logger.info(f"Loaded train={X_train.shape}, test={X_test.shape}")
    logger.info(f"Target distribution (train): {y_train.value_counts().to_dict()}")

    # ------------------------------------------------------------------
    # 2️⃣ Identify feature types
    # ------------------------------------------------------------------
    numeric_features, categorical_features = identify_feature_types(
        X_train, target_col="Churn"
    )

    # ------------------------------------------------------------------
    # 3️⃣ Build model definitions
    # ------------------------------------------------------------------
    model_defs = {
        "logistic_regression": LogisticRegression(**cfg["model"]["logistic_regression"]),
        "random_forest":       RandomForestClassifier(**cfg["model"]["random_forest"]),
        "xgboost":             XGBClassifier(**cfg["model"]["xgboost"]),
    }

    # ------------------------------------------------------------------
    # 4️⃣ Train each model, evaluate, and persist.
    # ------------------------------------------------------------------
    results = {}
    model_dir = Path(__file__).resolve().parents[2] / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    for name, estimator in model_defs.items():
        logger.info(f"Training {name} …")
        
        pipeline = get_churn_pipeline(
            model=estimator,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
        
        pipeline.fit(X_train, y_train)

        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred  = (y_proba >= 0.5).astype(int)

        metrics = evaluate_model(y_true=y_test, y_pred=y_pred, y_proba=y_proba)
        results[name] = metrics
        logger.info(f"{name} metrics: {metrics}")

        model_path = model_dir / f"{name}_pipeline.joblib"
        joblib.dump(pipeline, model_path)
        logger.info(f"Saved {name} pipeline to {model_path}")

    # ----------------------------------------------------------------------
    # 5️⃣ Best model selection and reporting
    # ----------------------------------------------------------------------
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    summary = {
        "selected_model": best_name,
        "selected_metrics": results[best_name],
        "all_models": results,
    }

    report_path = model_dir / "metrics.json"
    with report_path.open("w") as f:
        json.dump(summary, f, indent=2)
        
    logger.info(f"✅ Training complete – best model = {best_name}")
    logger.info(f"Metrics report written to {report_path}")

if __name__ == "__main__":
    main()