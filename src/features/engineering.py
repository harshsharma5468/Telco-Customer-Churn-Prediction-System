"""Feature engineering module with robust dtype handling."""

import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _safe_divide(num, den, fill=0):
    """
    Safely divide two series, handling division by zero and dtype issues.
    
    Args:
        num: Numerator (Series)
        den: Denominator (Series)
        fill: Fill value for division by zero or NaN
        
    Returns:
        Series with division results
    """
    # Ensure numeric types
    num = pd.to_numeric(num, errors='coerce')
    den = pd.to_numeric(den, errors='coerce')
    
    # Handle fill value
    if isinstance(fill, pd.Series):
        fill = pd.to_numeric(fill, errors='coerce')
    
    # Perform safe division
    result = np.where((den != 0) & (den.notna()) & (num.notna()), 
                     num / den, 
                     fill)
    
    return pd.Series(result, index=num.index)


def ensure_numeric_columns(df):
    """
    Ensure expected numeric columns are actually numeric.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with corrected dtypes
    """
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    
    for col in numeric_cols:
        if col in df.columns:
            # Convert to numeric, coercing errors
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Log if any NaN were created
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                logger.warning(f"Created {nan_count} NaN values in {col} during conversion")
    
    return df


def add_tenure_group(df):
    """Add tenure grouping feature."""
    df = df.copy()
    
    # Ensure tenure is numeric
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
    
    # Create tenure groups
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72, np.inf],
        labels=["0-12", "12-24", "24-48", "48-72", "72+"],
        include_lowest=True
    ).astype(str)  # Convert to string explicitly
    
    return df


def add_average_monthly_charge(df):
    """Add average monthly charge feature."""
    df = df.copy()
    
    # Ensure columns are numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
    
    # Calculate average monthly charge
    df["avg_monthly_charge"] = _safe_divide(
        df["TotalCharges"],
        df["tenure"],
        fill=df["MonthlyCharges"]
    )
    
    return df


def add_contract_features(df):
    """Add contract-related binary features."""
    df = df.copy()
    
    if 'Contract' in df.columns:
        df["is_monthly_contract"] = (df["Contract"] == "Month-to-month").astype(int)
        df["is_one_year_contract"] = (df["Contract"] == "One year").astype(int)
        df["is_two_year_contract"] = (df["Contract"] == "Two year").astype(int)
    
    return df


def add_internet_service_flag(df):
    """Add internet service flag."""
    df = df.copy()
    
    if 'InternetService' in df.columns:
        df["has_internet_service"] = (df["InternetService"] != "No").astype(int)
    
    return df


def add_senior_flag(df):
    """Add senior citizen flag."""
    df = df.copy()
    
    if 'SeniorCitizen' in df.columns:
        # Ensure it's numeric first
        df['SeniorCitizen'] = pd.to_numeric(df['SeniorCitizen'], errors='coerce')
        df["is_senior"] = (df["SeniorCitizen"] == 1).astype(int)
    
    return df


def add_service_count(df):
    """Count number of active services."""
    df = df.copy()
    
    service_columns = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    
    # Count services that are "Yes"
    service_count = 0
    for col in service_columns:
        if col in df.columns:
            service_count = service_count + (df[col] == "Yes").astype(int)
    
    df["num_active_services"] = service_count
    
    return df


def add_payment_method_flags(df):
    """Add payment method binary flags."""
    df = df.copy()
    
    if 'PaymentMethod' in df.columns:
        df["pay_electronic_check"] = (
            df["PaymentMethod"] == "Electronic check"
        ).astype(int)
        
        df["pay_mailed_check"] = (
            df["PaymentMethod"] == "Mailed check"
        ).astype(int)
        
        df["pay_bank_transfer_automatic"] = (
            df["PaymentMethod"] == "Bank transfer (automatic)"
        ).astype(int)
        
        df["pay_credit_card_automatic"] = (
            df["PaymentMethod"] == "Credit card (automatic)"
        ).astype(int)
    
    return df


def add_billing_lag(df):
    """Add billing lag feature (difference between total and expected charges)."""
    df = df.copy()
    
    # Ensure numeric types
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
    
    expected_charges = df["tenure"] * df["MonthlyCharges"]
    df["billing_lag"] = df["TotalCharges"] - expected_charges
    
    # Fill NaN with 0
    df["billing_lag"] = df["billing_lag"].fillna(0)
    
    return df


def lowercase_columns(df):
    """Convert column names to lowercase for consistency."""
    df = df.copy()
    df.columns = df.columns.str.lower()
    return df


def engineer_features(df):
    """
    Main feature engineering pipeline.
    
    Args:
        df: Input DataFrame with raw features
        
    Returns:
        DataFrame with engineered features
    """
    logger.info("Starting feature-engineering pipeline.")
    
    df = df.copy()
    
    # Step 0: Ensure numeric columns are actually numeric
    df = ensure_numeric_columns(df)
    
    # Apply all transformations
    df = (
        df
        .pipe(add_tenure_group)
        .pipe(add_average_monthly_charge)
        .pipe(add_contract_features)
        .pipe(add_internet_service_flag)
        .pipe(add_senior_flag)
        .pipe(add_service_count)
        .pipe(add_payment_method_flags)
        .pipe(add_billing_lag)
        .pipe(lowercase_columns)
    )
    
    logger.info("Feature-engineering completed.")
    
    return df


if __name__ == "__main__":
    # Test with sample data
    sample = pd.DataFrame({
        'tenure': [1, 12, 24, 48],
        'MonthlyCharges': [50, 60, 70, 80],
        'TotalCharges': [50, 720, 1680, 3840],
        'Contract': ['Month-to-month', 'One year', 'Two year', 'Two year'],
        'InternetService': ['DSL', 'Fiber optic', 'Fiber optic', 'No'],
        'SeniorCitizen': [0, 1, 0, 1],
        'PhoneService': ['Yes', 'Yes', 'No', 'Yes'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 
                         'Bank transfer (automatic)', 'Credit card (automatic)']
    })
    
    result = engineer_features(sample)
    print("Engineered features:")
    print(result.columns.tolist())
    print("\nSample output:")
    print(result.head())
