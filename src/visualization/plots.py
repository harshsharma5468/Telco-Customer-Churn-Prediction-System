"""
Hybrid Visualization Module.
Used for both automated production reports and interactive notebook exploration.
"""

import json
from pathlib import Path
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve

from src.utils.logger import get_logger

# Use 'Agg' backend for headless environments (Docker/Terminal)
if __name__ == "__main__":
    matplotlib.use('Agg')

logger = get_logger(__name__)

# --- 1️⃣ Missing Helpers (RESTORED FOR NOTEBOOK 02) ---

def plot_missing_matrix(df, title="Missing values heatmap"):
    """Compact heatmap where True = missing."""
    plt.figure(figsize=(12, 4))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
    plt.title(title)
    plt.tight_layout()
    # If running in a notebook, this shows the plot; in Docker, it prevents an error
    if matplotlib.get_backend().lower() != 'agg':
        plt.show()
    else:
        plt.close()

def plot_distribution_comparison(original, engineered, col_name):
    """Overlapped histograms for original vs processed features."""
    plt.figure(figsize=(10, 4))
    if original is not None:
        sns.kdeplot(original, label="original", fill=True, bw_adjust=0.5)
    sns.kdeplot(engineered, label="engineered", fill=True, bw_adjust=0.5)
    plt.title(f"Distribution of `{col_name}`")
    plt.legend()
    plt.tight_layout()
    if matplotlib.get_backend().lower() != 'agg':
        plt.show()
    else:
        plt.close()

# --- 2️⃣ Your Existing Performance Functions ---

def plot_roc_curve(y_true, y_proba, model_name="model", ax=None):
    """Adds a single ROC curve to the supplied axes."""
    if len(np.unique(y_true)) < 2:
        logger.warning(f"Skipping ROC for {model_name}: Data contains only one class.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    if ax is None:
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        
    ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

def plot_confusion_matrix(y_true, y_pred, model_name="model", normalize=False):
    """Displays a styled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, normalize="true" if normalize else None)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt=".2f" if normalize else "d", 
        cmap="Blues", 
        xticklabels=["No", "Yes"], 
        yticklabels=["No", "Yes"]
    )
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

def plot_feature_importance(model, feature_names, model_name="model", top_n=15, save_path=None):
    """Plots top-n feature importances."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        logger.warning(f"Model {model_name} does not support importance plotting.")
        return

    if len(feature_names) != len(importances):
        feature_names = [f"feat_{i}" for i in range(len(importances))]

    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    top = fi.head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top.values, y=top.index, hue=top.index, palette="viridis", legend=False)
    plt.title(f"{model_name} - Top {top_n} Features")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# --- 3️⃣ Automated Reporting Logic ---

def generate_evaluation_plots():
    """Automates champion model reporting by calling individual functions."""
    root_dir = Path(__file__).resolve().parents[2]
    model_dir = root_dir / "models"
    data_dir = root_dir / "data" / "processed"
    output_dir = root_dir / "reports" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "metrics.json", "r") as f:
        metrics = json.load(f)
    best_name = metrics["selected_model"]
    
    model_pipeline = joblib.load(model_dir / f"{best_name}_pipeline.joblib")
    X_test = pd.read_parquet(data_dir / "X_test.parquet")
    y_test = pd.read_parquet(data_dir / "y_test.parquet")["Churn"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    y_proba = model_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    # ROC
    plt.figure(figsize=(8, 6))
    plot_roc_curve(y_test, y_proba, model_name=best_name, ax=plt.gca())
    plt.savefig(output_dir / "roc_curve.png")
    plt.close()

    # Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, model_name=best_name)
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()

    # Feature Importance
    model_step = model_pipeline.steps[-1][1]
    preprocessor = next(s for n, s in model_pipeline.named_steps.items() if hasattr(s, 'get_feature_names_out'))
    plot_feature_importance(
        model_step, 
        preprocessor.get_feature_names_out(), 
        best_name, 
        save_path=output_dir / "feature_importance.png"
    )

    logger.info(f"✅ Success! Reports saved to {output_dir}")

if __name__ == "__main__":
    generate_evaluation_plots()