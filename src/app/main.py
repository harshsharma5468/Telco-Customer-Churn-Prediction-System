"""
Streamlit application for Telco Customer Churn prediction.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Fix PYTHONPATH so `src.*` imports work when running locally
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.models.predict import predict_churn


# ------------------------------------------------------------------
# Normalize Streamlit input to match training data types
# ------------------------------------------------------------------
def normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure Streamlit inputs are compatible with sklearn pipelines:
    - No None values
    - Correct numeric types
    - Categoricals are strings
    """
    df = df.copy()

    for col in df.columns:
        # Replace None with NaN
        df[col] = df[col].apply(lambda x: np.nan if x is None else x)

        # Convert booleans to strings (training data used strings)
        if df[col].dtype == bool:
            df[col] = df[col].astype(str)

        # Try converting to numeric where possible
        df[col] = pd.to_numeric(df[col], errors="ignore")

        # Ensure categoricals are strings
        if df[col].dtype == object:
            df[col] = df[col].astype(str)

    return df


# ------------------------------------------------------------------
# Streamlit App
# ------------------------------------------------------------------
def run_app():
    st.set_page_config(page_title="Telco Churn Predictor", layout="centered")

    st.title("📞 Telco Customer Churn Predictor")
    st.write("Enter customer details to predict churn probability.")

    # -----------------------------
    # User Inputs (RAW FEATURES)
    # -----------------------------
    user_input = {
        "gender": st.selectbox("Gender", ["Male", "Female"]),
        "SeniorCitizen": st.selectbox("Senior Citizen", ["Yes", "No"]),
        "Partner": st.selectbox("Partner", ["Yes", "No"]),
        "Dependents": st.selectbox("Dependents", ["Yes", "No"]),
        "tenure": st.number_input("Tenure (months)", min_value=0, max_value=100, value=12),
        "PhoneService": st.selectbox("Phone Service", ["Yes", "No"]),
        "MultipleLines": st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"]),
        "InternetService": st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"]),
        "OnlineSecurity": st.selectbox("Online Security", ["Yes", "No", "No internet service"]),
        "OnlineBackup": st.selectbox("Online Backup", ["Yes", "No", "No internet service"]),
        "DeviceProtection": st.selectbox("Device Protection", ["Yes", "No", "No internet service"]),
        "TechSupport": st.selectbox("Tech Support", ["Yes", "No", "No internet service"]),
        "StreamingTV": st.selectbox("Streaming TV", ["Yes", "No", "No internet service"]),
        "StreamingMovies": st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"]),
        "Contract": st.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
        "PaperlessBilling": st.selectbox("Paperless Billing", ["Yes", "No"]),
        "PaymentMethod": st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        ),
        "MonthlyCharges": st.number_input("Monthly Charges", min_value=0.0, value=70.0),
        "TotalCharges": st.number_input("Total Charges", min_value=0.0, value=1000.0),
    }

    if st.button("🔮 Predict Churn"):
        input_df = pd.DataFrame([user_input])
        input_df = normalize_input(input_df)

        try:
            prob = predict_churn(input_df).iloc[0]

            st.success(f"📊 **Churn Probability:** {prob:.2%}")

            if prob >= 0.5:
                st.warning("⚠️ Customer is likely to churn.")
            else:
                st.info("✅ Customer is unlikely to churn.")

        except Exception as e:
            st.error("❌ Prediction failed.")
            st.exception(e)


if __name__ == "__main__":
    run_app()
