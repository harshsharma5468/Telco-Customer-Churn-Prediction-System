"""
A scikit‑learn Pipeline that combines:

* The ColumnTransformer from ``src.features.preprocessing`` (numeric scaling + OHE)
* A model (LogisticRegression, RandomForest, XGBoost – injected later)

The pipeline will be instantiated in Step 6 after the model objects are
created.
"""

from sklearn.pipeline import Pipeline
from src.features.preprocessing import build_preprocessor, identify_feature_types
from src.utils.logger import get_logger

logger = get_logger(__name__)

def get_churn_pipeline(model, *, numeric_features, categorical_features):
    """
    Assemble a full pipeline given a trained model and pre‑identified feature
    names.

    Parameters
    ----------
    model : estimator
        Any scikit‑learn compatible estimator (e.g., LogisticRegression).
    numeric_features, categorical_features : list[str]
        Column names that the preprocessor should handle.

    Returns
    -------
    Pipeline
        ``preprocess -> model`` pipeline ready for ``fit`` or ``predict``.
    """
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    logger.info(f"Created churn pipeline with model {model.__class__.__name__}")
    return pipeline
