import streamlit as st

def validate_inputs(data):
    warnings = []

    if data["tenure"].iloc[0] < 3:
        warnings.append("⚠️ Very new customer (low tenure)")

    if data["MonthlyCharges"].iloc[0] > 100:
        warnings.append("⚠️ Unusually high monthly charges")

    if data["TotalCharges"].iloc[0] < data["MonthlyCharges"].iloc[0]:
        warnings.append("⚠️ Total charges seem inconsistent")

    for w in warnings:
        st.warning(w)
