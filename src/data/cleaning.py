import pandas as pd
from pathlib import Path
import logging

# Initialize logger
logger = logging.getLogger(__name__)

def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw telco data for both training and inference.
    Handles the case where the 'Churn' target column may be missing during prediction.
    """
    # Define keep_cols locally or import if defined elsewhere
    # Based on standard Telco Churn datasets, this usually includes features like:
    # 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', etc.
    # For now, we ensure we only select columns that actually exist in the input df.
    
    # Identify which columns from the dataframe to keep (excluding customerID)
    # We use a list comprehension to avoid KeyError during inference (when 'Churn' is absent)
    cols_to_filter = [col for col in df.columns if col != 'customerID']
    df = df[cols_to_filter].copy()

    # 5. Final Categorization
    # Convert all object types to category for memory efficiency and model compatibility
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].astype("category")

    # 6. Drop ID (Safety check if customerID wasn't filtered out above)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # --- FIX FOR LINE 83: Conditional Logging ---
    # This prevents the KeyError: 'Churn' when running predict_churn()
    if 'Churn' in df.columns:
        distribution = df['Churn'].value_counts().to_dict()
        logger.info(f"Cleaning finished. Final Churn distribution: {distribution}")
    else:
        logger.info("Cleaning finished. (Inference mode: No 'Churn' column present in input data)")

    return df

def save_interim(df: pd.DataFrame, 
                 interim_dir: Path = Path(__file__).resolve().parents[2] / "data" / "interim",
                 filename: str = "cleaned.csv") -> Path:
    """
    Saves the cleaned dataframe to the interim data folder.
    """
    interim_dir.mkdir(parents=True, exist_ok=True)
    out_path = interim_dir / filename
    df.to_csv(out_path, index=False)
    return out_path