import pandas as pd
import numpy as np
from src.data.cleaning import clean_raw

def test_clean_raw_imputes_total_charges():
    # Minimal raw example with the known issue (blank TotalCharges)
    raw = pd.DataFrame({
        "customerID": ["AAA-123"],
        "tenure": [5],
        "MonthlyCharges": [70.0],
        "TotalCharges": [" "],  # blank string → NaN after conversion
        "Churn": ["No"]
    })
    cleaned = clean_raw(raw)
    # Expected: TotalCharges = tenure * MonthlyCharges = 350
    assert np.isclose(cleaned["TotalCharges"].iloc[0], 350.0)
    # customerID should be removed
    assert "customerID" not in cleaned.columns
