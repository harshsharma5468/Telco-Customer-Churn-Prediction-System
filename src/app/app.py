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

import streamlit as st
import pandas as pd
import plotly.express as px
from src.models.predict import predict_churn

def run_app():
    st.set_page_config(page_title="Telco Churn Analytics", layout="wide")
    st.title("📊 Telco Customer Churn Predictor")

    # --- SIDEBAR: Bulk Upload ---
    st.sidebar.header("Bulk Prediction")
    uploaded_file = st.sidebar.file_uploader("Upload Customer CSV", type=["csv"])
    
    if uploaded_file:
        bulk_df = pd.read_csv(uploaded_file)
        # Ensure cleaning logic is applied inside predict_churn
        results = predict_churn(bulk_df) 
        st.sidebar.success("Predictions Complete!")
        st.sidebar.download_button("Download Results", results.to_csv(), "predictions.csv")

    # --- MAIN UI: What-If Analysis ---
    st.subheader("Interactive 'What-If' Analysis")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.info("Adjust parameters to see real-time impact")
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 70)
        total_charges = st.number_input("Total Charges ($)", value=tenure * monthly_charges)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

        # Create input dataframe for prediction
        input_data = pd.DataFrame({
            'tenure': [tenure],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges],
            'Contract': [contract],
            'TechSupport': [tech_support],
            # Add other necessary columns with default values if needed
        })

    with col2:
        # Get prediction and probability
        # Assuming predict_churn returns a df with a 'Probability' column
        pred_df = predict_churn(input_data)
        prob = pred_df['Probability'].iloc[0]
        
        # Display Gauge/Metric
        st.metric(label="Churn Probability", value=f"{prob:.1%}")
        
        if prob > 0.5:
            st.error("⚠️ High Risk of Churn")
        else:
            st.success("✅ Low Risk / Loyal Customer")

        # --- Feature Importance Chart ---
        st.write("---")
        st.subheader("Why this prediction?")
        
        # Mock Feature Importance (Replace with model.feature_importances_ or SHAP values)
        # For a live model, you'd extract these from your trained pipeline
        importance_data = pd.DataFrame({
            'Factor': ['Tenure', 'Monthly Charges', 'Contract', 'Tech Support'],
            'Influence': [tenure * -0.5, monthly_charges * 0.8, -20 if contract != "Month-to-month" else 20, -10 if tech_support == "Yes" else 10]
        }).sort_values(by='Influence', ascending=True)

        fig = px.bar(importance_data, x='Influence', y='Factor', orientation='h',
                     title="Key Drivers for this Customer",
                     color='Influence', color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    run_app()