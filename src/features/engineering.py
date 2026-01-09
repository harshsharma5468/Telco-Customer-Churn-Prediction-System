"""
Feature-engineering helpers for the Telco churn dataset.
Safe for training, inference, notebooks, Streamlit, and Docker.
"""

import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Services used to compute active service count
_SERVICE_COLS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


# ------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------

def _safe_divide(num: pd.Series, den: pd.Series, fill: float = 0.0) -> pd.Series:
    """Element-wise division with safe fallback when denominator is zero."""
    return np.where(den != 0, num / den, fill)


def _binary_from_category(series: pd.Series, positive: str) -> pd.Series:
    """
    Convert categorical/text labels to 0/1 safely.
    CRITICAL: always cast to string to avoid pandas Categorical crashes.
    """
    return (series.astype(str) == positive).astype(int)


# ------------------------------------------------------------------
# Feature builders
# ------------------------------------------------------------------

def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket tenure months into categories."""
    df = df.copy()
    bins = [0, 12, 24, 48, np.inf]
    labels = ["0-12", "13-24", "25-48", "49+"]
    df["tenure_group"] = (
        pd.cut(df["tenure"], bins=bins, labels=labels, right=False)
        .astype("category")
    )
    logger.debug("Added `tenure_group` feature.")
    return df


def add_average_monthly_charge(df: pd.DataFrame) -> pd.DataFrame:
    """Average monthly charge derived from total charges and tenure."""
    df = df.copy()
    df["avg_monthly_charge"] = _safe_divide(
        df["TotalCharges"], df["tenure"], fill=df["MonthlyCharges"]
    )
    logger.debug("Added `avg_monthly_charge` feature.")
    return df


def add_contract_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flags for contract duration."""
    df = df.copy()
    df["is_monthly_contract"] = (df["Contract"] == "Month-to-month").astype(int)
    df["is_one_year_contract"] = (df["Contract"] == "One year").astype(int)
    df["is_two_year_contract"] = (df["Contract"] == "Two year").astype(int)
    logger.debug("Added contract flags.")
    return df


def add_internet_service_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flag indicating whether customer has internet service."""
    df = df.copy()
    df["has_internet_service"] = (df["InternetService"] != "No").astype(int)
    logger.debug("Added internet service flag.")
    return df


def add_senior_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes SeniorCitizen into a clean integer flag.
    Handles numeric, string, and categorical safely.
    """
    df = df.copy()

    if "SeniorCitizen" in df.columns:
        df["is_senior"] = (
            df["SeniorCitizen"]
            .astype(str)  # ✅ CRITICAL FIX
            .map({"Yes": 1, "No": 0, "1": 1, "0": 0})
            .fillna(0)
            .astype(int)
        )

    logger.debug("Added `is_senior` flag.")
    return df


def add_active_service_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Counts active add-on services.
    SAFE for inference even if columns are missing.
    """
    df = df.copy()
    temp_cols = []

    for col in _SERVICE_COLS:
        col_name = f"svc_{col}"
        if col in df.columns:
            df[col_name] = _binary_from_category(df[col], "Yes")
        else:
            df[col_name] = 0  # ✅ SAFE DEFAULT FOR STREAMLIT / INFERENCE
        temp_cols.append(col_name)

    df["num_active_services"] = df[temp_cols].sum(axis=1)
    df = df.drop(columns=temp_cols)
    logger.debug("Added `num_active_services` feature.")
    return df


def add_payment_method_flags(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encodes payment methods."""
    df = df.copy()
    methods = [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    for m in methods:
        col_name = f"pay_{m.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
        df[col_name] = (df["PaymentMethod"] == m).astype(int)

    logger.debug("Added payment method flags.")
    return df


def add_billing_lag(df: pd.DataFrame) -> pd.DataFrame:
    """Difference between actual and expected billing."""
    df = df.copy()
    df["billing_lag"] = df["TotalCharges"] - (df["MonthlyCharges"] * df["tenure"])
    logger.debug("Added `billing_lag` feature.")
    return df


# ------------------------------------------------------------------
# Master pipeline
# ------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature-engineering pipeline.
    Order matters and matches training exactly.
    """
    logger.info("Starting feature-engineering pipeline.")

    df = (
        df.pipe(add_tenure_group)
          .pipe(add_average_monthly_charge)
          .pipe(add_contract_flags)
          .pipe(add_internet_service_flag)
          .pipe(add_senior_flag)
          .pipe(add_active_service_count)
          .pipe(add_payment_method_flags)
          .pipe(add_billing_lag)
    )

    logger.info("Feature-engineering completed.")
    return df
