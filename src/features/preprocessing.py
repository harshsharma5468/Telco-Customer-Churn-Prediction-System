"""Pre‑processing utilities that build a scikit‑learn ColumnTransformer.

The transformer will:
* One‑hot encode every categorical column (including engineered categories).
* Standard‑scale all numeric columns.
"""

from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils.logger import get_logger

logger = get_logger(__name__)

def identify_feature_types(df, target_col="Churn"):
    """
    Separates numeric and categorical columns. 
    Skips the target column if it is present in the DataFrame.
    """
    # Create a local copy to avoid modifying the original X_train
    feature_df = df.copy()
    
    # 1. Only drop if the target column exists
    if target_col in feature_df.columns:
        feature_df = feature_df.drop(columns=[target_col])
    
    # 2. Identify numeric columns
    numeric_features = feature_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # 3. Identify categorical columns (including object and 'category' dtypes)
    categorical_features = feature_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    return numeric_features, categorical_features


def build_preprocessor(numeric_features: List[str],
                       categorical_features: List[str]) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    * Scales numeric columns (zero‑mean, unit‑variance).
    * One‑hot encodes categorical columns (ignore unknown categories at inference).
    
    """
    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    num_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numeric_features),
            ("cat", cat_transformer, categorical_features),
        ],
        remainder="drop",
    )
    logger.info(
        f"Preprocessor created with {len(numeric_features)} numeric "
        f"and {len(categorical_features)} categorical features."
    )
    return preprocessor
