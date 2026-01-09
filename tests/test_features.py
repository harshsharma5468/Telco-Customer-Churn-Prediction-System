# tests/test_features.py
import pandas as pd
import numpy as np
from src.features.engineering import (
    add_tenure_group,
    add_average_monthly_charge,
    add_contract_flags,
    add_internet_service_flag,
    add_senior_flag,
    add_active_service_count,
    add_payment_method_flags,
    add_billing_lag,
    engineer_features,
)

def _min_df():
    """A tiny synthetic dataframe that touches every column we expect."""
    return pd.DataFrame({
        "customerID": ["C1"],
        "gender": pd.Categorical(["Female"]),
        "SeniorCitizen": [0],
        "Partner": pd.Categorical(["Yes"]),
        "Dependents": pd.Categorical(["No"]),
        "tenure": [1],
        "PhoneService": pd.Categorical(["Yes"]),
        "MultipleLines": pd.Categorical(["No"]),
        "InternetService": pd.Categorical(["Fiber optic"]),
        "OnlineSecurity": pd.Categorical(["No"]),
        "OnlineBackup": pd.Categorical(["Yes"]),
        "DeviceProtection": pd.Categorical(["No"]),
        "TechSupport": pd.Categorical(["No"]),
        "StreamingTV": pd.Categorical(["No"]),
        "StreamingMovies": pd.Categorical(["No"]),
        "Contract": pd.Categorical(["Month-to-month"]),
        "PaperlessBilling": pd.Categorical(["Yes"]),
        "PaymentMethod": pd.Categorical(["Electronic check"]),
        "MonthlyCharges": [70.0],
        "TotalCharges": [70.0],
        "Churn": pd.Categorical(["No"])
    })

def test_engineer_features_all_columns_present():
    df = _min_df()
    df_fe = engineer_features(df)

    # Tenure group buckets
    assert "tenure_group" in df_fe.columns
    assert df_fe["tenure_group"].iloc[0] == "0-12"

    # Average monthly charge (should equal MonthlyCharges for tenure=1)
    assert np.isclose(df_fe["avg_monthly_charge"].iloc[0], 70.0)

    # Contract flags
    assert df_fe["is_monthly_contract"].iloc[0] == 1
    assert df_fe["is_one_year_contract"].iloc[0] == 0
    assert df_fe["is_two_year_contract"].iloc[0] == 0

    # Internet flag
    assert df_fe["has_internet_service"].iloc[0] == 1

    # Senior flag
    assert df_fe["is_senior"].iloc[0] == 0

    # Active service count – we have PhoneService (Yes) + OnlineBackup (Yes) = 2
    assert df_fe["num_active_services"].iloc[0] == 2

    # Payment‑method flags – only electronic check should be 1
    assert df_fe["pay_electronic_check"].iloc[0] == 1
    assert df_fe["pay_mailed_check"].iloc[0] == 0

    # Billing lag – TotalCharges equals MonthlyCharges*tenure, so lag == 0
    assert np.isclose(df_fe["billing_lag"].iloc[0], 0.0)
