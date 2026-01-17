"""

Features:
- Modern UI with tabs and sections
- Batch prediction support
- Feature importance visualization
- Risk assessment with recommendations
- Model performance metrics display
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ------------------------------------------------------------------
# Fix PYTHONPATH so `src.*` imports work when running locally
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.models.predict import predict_churn


# ------------------------------------------------------------------
# Configuration and Styling
# ------------------------------------------------------------------
def set_page_config():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Telco Churn Predictor",
        page_icon="📞",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            color: #1f77b4;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
        }
        .risk-high {
            color: #d62728;
            font-weight: bold;
        }
        .risk-medium {
            color: #ff7f0e;
            font-weight: bold;
        }
        .risk-low {
            color: #2ca02c;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------
def normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure Streamlit inputs are compatible with sklearn pipelines.
    """
    df = df.copy()

    for col in df.columns:
        # Replace None with NaN
        df[col] = df[col].apply(lambda x: np.nan if x is None else x)

        # Convert booleans to strings
        if df[col].dtype == bool:
            df[col] = df[col].astype(str)

        # Try converting to numeric (fixed deprecation warning)
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass  # Keep as is if conversion fails

        # Ensure categoricals are strings
        if df[col].dtype == object:
            df[col] = df[col].astype(str)

    return df


def get_risk_level(prob: float) -> tuple:
    """Get risk level and color based on probability."""
    if prob >= 0.7:
        return "🔴 HIGH RISK", "risk-high", "#d62728"
    elif prob >= 0.4:
        return "🟡 MEDIUM RISK", "risk-medium", "#ff7f0e"
    else:
        return "🟢 LOW RISK", "risk-low", "#2ca02c"


def get_retention_recommendations(input_data: dict, prob: float) -> list:
    """Generate personalized retention recommendations."""
    recommendations = []
    
    if prob < 0.3:
        return ["✅ Customer is stable. Continue providing excellent service."]
    
    # Contract-based recommendations
    if input_data.get("Contract") == "Month-to-month":
        recommendations.append("📋 Offer long-term contract with discount (12-24 months)")
    
    # Tenure-based recommendations
    if input_data.get("tenure", 0) < 12:
        recommendations.append("🎁 Provide new customer onboarding perks and loyalty bonus")
    
    # Service-based recommendations
    if input_data.get("InternetService") == "Fiber optic":
        recommendations.append("🌐 Offer fiber optic package upgrade or bundle discount")
    
    if input_data.get("TechSupport") == "No":
        recommendations.append("🛠️ Include free tech support for 3 months")
    
    if input_data.get("OnlineSecurity") == "No":
        recommendations.append("🔒 Offer free online security package trial")
    
    # Payment method recommendations
    if input_data.get("PaymentMethod") == "Electronic check":
        recommendations.append("💳 Incentivize automatic payment methods with discount")
    
    # Charges-based recommendations
    if input_data.get("MonthlyCharges", 0) > 80:
        recommendations.append("💰 Review pricing and offer competitive retention discount")
    
    # General high-risk recommendations
    if prob >= 0.7:
        recommendations.append("📞 Priority: Schedule immediate retention call")
        recommendations.append("🎯 Assign dedicated account manager")
    
    return recommendations if recommendations else ["📊 Monitor customer engagement closely"]


def create_probability_gauge(prob: float):
    """Create a gauge chart for churn probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Probability (%)", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#d4edda'},
                {'range': [40, 70], 'color': '#fff3cd'},
                {'range': [70, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def load_model_metrics():
    """Load model performance metrics."""
    try:
        import json
        metrics_path = ROOT / "reports" / "model_comparison.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                return json.load(f)
    except:
        pass
    return None


# ------------------------------------------------------------------
# Main App Components
# ------------------------------------------------------------------
def render_sidebar():
    """Render sidebar with info and model selection."""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/phone.png", width=80)
        st.title("About")
        st.info("""
        This app predicts customer churn probability using machine learning.
        
        **Features:**
        - Real-time predictions
        - Batch processing
        - Retention recommendations
        - Risk assessment
        """)
        
        st.markdown("---")
        
        # Model info
        metrics = load_model_metrics()
        if metrics:
            st.subheader("📊 Model Performance")
            selected_model = metrics.get('selected_model', 'Unknown')
            model_metrics = metrics.get('selected_metrics', {})
            
            st.metric("Model", selected_model.replace('_', ' ').title())
            st.metric("Accuracy", f"{model_metrics.get('accuracy', 0):.1%}")
            st.metric("F1 Score", f"{model_metrics.get('f1_score', 0):.1%}")
            st.metric("ROC AUC", f"{model_metrics.get('roc_auc', 0):.1%}")
        
        st.markdown("---")
        st.caption("Built with ❤️ using Streamlit")


def render_single_prediction_tab():
    """Render single customer prediction interface."""
    st.header("🔍 Single Customer Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
        senior = st.selectbox("Senior Citizen", ["No", "Yes"], key="senior")
        partner = st.selectbox("Has Partner", ["No", "Yes"], key="partner")
        dependents = st.selectbox("Has Dependents", ["No", "Yes"], key="dependents")
    
    with col2:
        st.subheader("📱 Services")
        phone = st.selectbox("Phone Service", ["Yes", "No"], key="phone")
        multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], key="multiple")
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], key="internet")
        
        # Conditional services based on internet
        if internet != "No":
            online_sec = st.selectbox("Online Security", ["No", "Yes"], key="online_sec")
            online_backup = st.selectbox("Online Backup", ["No", "Yes"], key="online_backup")
            device_prot = st.selectbox("Device Protection", ["No", "Yes"], key="device_prot")
            tech_support = st.selectbox("Tech Support", ["No", "Yes"], key="tech_support")
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"], key="streaming_tv")
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"], key="streaming_movies")
        else:
            online_sec = online_backup = device_prot = "No internet service"
            tech_support = streaming_tv = streaming_movies = "No internet service"
    
    with col3:
        st.subheader("💳 Account Info")
        tenure = st.slider("Tenure (months)", 0, 72, 12, key="tenure")
        contract = st.selectbox("Contract Type", 
                                ["Month-to-month", "One year", "Two year"], 
                                key="contract")
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"], key="paperless")
        payment = st.selectbox("Payment Method", 
                               ["Electronic check", "Mailed check", 
                                "Bank transfer (automatic)", "Credit card (automatic)"],
                               key="payment")
        monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, key="monthly")
        total = st.number_input("Total Charges ($)", 0.0, 10000.0, 
                               monthly * tenure, key="total")
    
    st.markdown("---")
    
    if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
        user_input = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_prot,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }
        
        input_df = pd.DataFrame([user_input])
        input_df = normalize_input(input_df)
        
        try:
            with st.spinner("Analyzing customer data..."):
                prob = predict_churn(input_df).iloc[0]
            
            # Display results
            st.markdown("### 📊 Prediction Results")
            
            # Three columns for metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                risk_label, risk_class, risk_color = get_risk_level(prob)
                st.markdown(f"<div class='metric-card'>"
                          f"<h3>{risk_label}</h3>"
                          f"<h1 style='color: {risk_color};'>{prob:.1%}</h1>"
                          f"</div>", unsafe_allow_html=True)
            
            with col2:
                st.plotly_chart(create_probability_gauge(prob), width='stretch')
            
            with col3:
                retention_cost = monthly * 3  # 3 months of service
                churn_cost = monthly * 12  # Lost annual revenue
                st.metric("Retention Cost", f"${retention_cost:.2f}")
                st.metric("Potential Loss", f"${churn_cost:.2f}")
                st.metric("ROI", f"{((churn_cost - retention_cost) / retention_cost * 100):.0f}%")
            
            # Recommendations
            st.markdown("### 💡 Retention Recommendations")
            recommendations = get_retention_recommendations(user_input, prob)
            for rec in recommendations:
                st.markdown(f"- {rec}")
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            with st.expander("Show error details"):
                st.exception(e)


def render_batch_prediction_tab():
    """Render batch prediction interface."""
    st.header("📊 Batch Prediction")
    
    st.info("Upload a CSV file with customer data to predict churn for multiple customers.")
    
    # Sample template
    with st.expander("📥 Download CSV Template"):
        template_data = {
            "gender": ["Male", "Female"],
            "SeniorCitizen": ["No", "Yes"],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "Yes"],
            "tenure": [12, 24],
            "PhoneService": ["Yes", "Yes"],
            "MultipleLines": ["No", "Yes"],
            "InternetService": ["Fiber optic", "DSL"],
            "OnlineSecurity": ["No", "Yes"],
            "OnlineBackup": ["Yes", "No"],
            "DeviceProtection": ["No", "Yes"],
            "TechSupport": ["No", "Yes"],
            "StreamingTV": ["No", "Yes"],
            "StreamingMovies": ["Yes", "No"],
            "Contract": ["Month-to-month", "One year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "MonthlyCharges": [70.0, 80.0],
            "TotalCharges": [840.0, 1920.0],
        }
        template_df = pd.DataFrame(template_data)
        csv = template_df.to_csv(index=False)
        st.download_button(
            label="Download Template CSV",
            data=csv,
            file_name="churn_prediction_template.csv",
            mime="text/csv"
        )
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} customers")
            
            with st.expander("Preview data"):
                st.dataframe(df.head())
            
            if st.button("🔮 Predict All", type="primary"):
                with st.spinner("Processing predictions..."):
                    df_normalized = normalize_input(df)
                    predictions = predict_churn(df_normalized)
                    
                    # Add predictions to dataframe
                    results_df = df.copy()
                    results_df['Churn_Probability'] = predictions
                    results_df['Risk_Level'] = predictions.apply(
                        lambda x: get_risk_level(x)[0]
                    )
                    
                    # Summary statistics
                    st.markdown("### 📈 Summary Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    high_risk = (predictions >= 0.7).sum()
                    medium_risk = ((predictions >= 0.4) & (predictions < 0.7)).sum()
                    low_risk = (predictions < 0.4).sum()
                    avg_prob = predictions.mean()
                    
                    col1.metric("Total Customers", len(df))
                    col2.metric("High Risk 🔴", high_risk)
                    col3.metric("Medium Risk 🟡", medium_risk)
                    col4.metric("Low Risk 🟢", low_risk)
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Risk distribution
                        risk_counts = pd.Series({
                            'High Risk': high_risk,
                            'Medium Risk': medium_risk,
                            'Low Risk': low_risk
                        })
                        fig = px.pie(values=risk_counts.values, 
                                   names=risk_counts.index,
                                   title="Risk Distribution",
                                   color_discrete_sequence=['#d62728', '#ff7f0e', '#2ca02c'])
                        st.plotly_chart(fig, width='stretch')
                    
                    with col2:
                        # Probability distribution
                        fig = px.histogram(predictions, 
                                         nbins=30,
                                         title="Churn Probability Distribution",
                                         labels={'value': 'Churn Probability', 'count': 'Number of Customers'})
                        st.plotly_chart(fig, width='stretch')
                    
                    # Results table
                    st.markdown("### 📋 Detailed Results")
                    st.dataframe(
                        results_df.sort_values('Churn_Probability', ascending=False),
                        width='stretch'
                    )
                    
                    # Download results
                    csv_results = results_df.to_csv(index=False)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        label="📥 Download Results",
                        data=csv_results,
                        file_name=f"churn_predictions_{timestamp}.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            with st.expander("Show error details"):
                st.exception(e)


# ------------------------------------------------------------------
# Main Application
# ------------------------------------------------------------------
def run_app():
    """Main application entry point."""
    set_page_config()
    render_sidebar()
    
    # Main header
    st.markdown('<p class="main-header">📞 Telco Customer Churn Predictor</p>', 
                unsafe_allow_html=True)
    st.markdown("Predict customer churn risk and get actionable retention strategies")
    
    # Tabs
    tab1, tab2 = st.tabs(["🔍 Single Prediction", "📊 Batch Prediction"])
    
    with tab1:
        render_single_prediction_tab()
    
    with tab2:
        render_batch_prediction_tab()


if __name__ == "__main__":
    run_app()
