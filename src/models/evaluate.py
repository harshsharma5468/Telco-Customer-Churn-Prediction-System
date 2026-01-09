"""
Utility functions that turn raw predictions into business‑relevant metrics.
All metrics are **binary‑classification** scores, expressed as floats
(0‑1 range). The function is deliberately tiny so that it can be called
from notebooks, CI pipelines, or the training script.
"""

from typing import Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
)

def evaluate_model(*, y_true, y_pred, y_proba) -> Dict[str, float]:
    """
    Compute a fixed set of classification metrics.

    Parameters
    ----------
    y_true : array‑like
        Ground‑truth binary labels (0 = no churn, 1 = churn).
    y_pred : array‑like
        Binary predictions obtained by thresholding `y_proba` (default 0.5).
    y_proba : array‑like
        Predicted probability for the positive class (churn).

    Returns
    -------
    dict
        {
            "accuracy": ...,
            "precision": ...,
            "recall": ...,
            "f1_score": ...,
            "roc_auc": ...,
            "brier_score": ...
        }
    """
    # Guard against empty inputs
    if len(y_true) == 0:
        raise ValueError("Empty y_true – cannot compute metrics.")

    # Calculate metrics and round to 4 decimal places for cleaner reporting
    metrics = {
        "accuracy":      round(float(accuracy_score(y_true, y_pred)), 4),
        "precision":     round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":        round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score":      round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc":       round(float(roc_auc_score(y_true, y_proba)), 4),
        "brier_score":   round(float(brier_score_loss(y_true, y_proba)), 4),
    }
    
    return metrics
