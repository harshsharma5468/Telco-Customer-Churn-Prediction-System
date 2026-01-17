"""
Prediction module for churn prediction.

Loads trained models and makes predictions on new data.
"""

import json
from pathlib import Path
import pandas as pd
import joblib
from src.features.engineering import engineer_features
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_best_model():
    """
    Load the best performing model from disk.
    
    Returns:
        Tuple of (model_pipeline, threshold, model_name)
    """
    model_dir = Path(__file__).resolve().parents[2] / "models"
    reports_dir = Path(__file__).resolve().parents[2] / "reports"
    
    # Try multiple possible locations for metrics
    metrics_files = [
        reports_dir / "model_comparison.json",
        model_dir / "metrics.json",
        model_dir / "model_comparison.json",
    ]
    
    metrics = None
    for metrics_file in metrics_files:
        if metrics_file.exists():
            logger.info(f"Loading metrics from {metrics_file}")
            with open(metrics_file) as f:
                metrics = json.load(f)
            break
    
    if metrics is None:
        # Fallback: load any available model
        logger.warning("No metrics file found, loading first available model")
        model_files = list(model_dir.glob("*.pkl"))
        
        if not model_files:
            raise FileNotFoundError(
                f"No trained models found in {model_dir}. "
                "Please run training first: python -m src.models.train"
            )
        
        model_path = model_files[0]
        model_name = model_path.stem
        logger.info(f"Loading model: {model_name}")
        
        model = joblib.load(model_path)
        
        # Try to load threshold
        threshold_path = model_dir / f"{model_name}_threshold.txt"
        if threshold_path.exists():
            with open(threshold_path) as f:
                threshold = float(f.read().strip())
        else:
            threshold = 0.5
            logger.warning(f"No threshold file found, using default: {threshold}")
        
        return model, threshold, model_name
    
    # Load best model from metrics
    selected_model = metrics.get("selected_model", "logistic_regression")
    model_path = model_dir / f"{selected_model}.pkl"
    
    if not model_path.exists():
        # Try to find any model
        available_models = list(model_dir.glob("*.pkl"))
        if available_models:
            model_path = available_models[0]
            selected_model = model_path.stem
            logger.warning(
                f"Selected model {selected_model} not found. "
                f"Using {model_path.name} instead."
            )
        else:
            raise FileNotFoundError(
                f"No trained models found in {model_dir}. "
                "Please run training first: python -m src.models.train"
            )
    
    logger.info(f"Loading best model: {selected_model}")
    model = joblib.load(model_path)
    
    # Load threshold
    threshold_path = model_dir / f"{selected_model}_threshold.txt"
    if threshold_path.exists():
        with open(threshold_path) as f:
            threshold = float(f.read().strip())
        logger.info(f"Loaded threshold: {threshold}")
    else:
        # Try to get from metrics
        threshold = metrics.get("selected_metrics", {}).get("threshold", 0.5)
        logger.warning(f"No threshold file found, using from metrics: {threshold}")
    
    return model, threshold, selected_model


def predict_churn(input_df: pd.DataFrame) -> pd.Series:
    """
    Predict churn probability for input data.
    
    Args:
        input_df: DataFrame with customer features (raw features, not engineered)
        
    Returns:
        Series of churn probabilities (0-1)
    """
    try:
        # Load model
        model, threshold, model_name = load_best_model()
        logger.info(f"Using model: {model_name} with threshold: {threshold}")
        
        # Apply feature engineering
        logger.info("Applying feature engineering...")
        input_engineered = engineer_features(input_df)
        logger.info(f"Engineered features: {input_engineered.shape}")
        
        # Make predictions (probability of churn)
        logger.info("Making predictions...")
        predictions = model.predict_proba(input_engineered)[:, 1]
        
        logger.info(f"Generated {len(predictions)} predictions")
        return pd.Series(predictions, index=input_df.index)
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


def predict_churn_binary(input_df: pd.DataFrame) -> pd.Series:
    """
    Predict binary churn labels (0/1) using optimal threshold.
    
    Args:
        input_df: DataFrame with customer features
        
    Returns:
        Series of binary predictions (0 or 1)
    """
    model, threshold, model_name = load_best_model()
    
    # Get probabilities
    probabilities = predict_churn(input_df)
    
    # Apply threshold
    predictions = (probabilities >= threshold).astype(int)
    
    logger.info(
        f"Binary predictions: {predictions.sum()} churners, "
        f"{(predictions == 0).sum()} non-churners"
    )
    
    return predictions


def batch_predict_from_csv(csv_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Batch prediction from CSV file.
    
    Args:
        csv_path: Path to input CSV file
        output_path: Path to save results (optional)
        
    Returns:
        DataFrame with predictions added
    """
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Make predictions
    probabilities = predict_churn(df)
    predictions = predict_churn_binary(df)
    
    # Add to dataframe
    df['churn_probability'] = probabilities
    df['churn_prediction'] = predictions
    df['risk_level'] = pd.cut(
        probabilities,
        bins=[0, 0.4, 0.7, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    
    # Save if output path provided
    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"Saved predictions to {output_path}")
    
    return df


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "predictions.csv"
        
        results = batch_predict_from_csv(input_file, output_file)
        print(f"\nPredictions saved to {output_file}")
        print(f"\nSummary:")
        print(results['risk_level'].value_counts())
    else:
        print("Usage: python -m src.models.predict <input.csv> [output.csv]")
