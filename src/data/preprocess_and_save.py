# src/data/preprocess_and_save.py
"""
Top‑level script that runs the full data‑preparation pipeline.

Steps:
1️⃣ Load raw CSV (Kaggle fallback inside the function)  
2️⃣ Clean the data (missing‑value handling, type fixing)  
3️⃣ **Feature engineering** (new columns)  
4️⃣ Encode target (0/1)  
5️⃣ Train / test split (stratified)  
6️⃣ Persist splits as parquet files.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger
from src.data.ingestion import load_raw_data
from src.data.cleaning import clean_raw, save_interim
from src.features.engineering import engineer_features   # <-- NEW IMPORT

logger = get_logger(__name__)

def main():
    # ------------------------------------------------------------------
    # Resolve project‑relative paths (so the script works from any cwd)
    # ------------------------------------------------------------------
    root = Path(__file__).resolve().parents[2]
    raw_dir = root / "data" / "raw"
    interim_dir = root / "data" / "interim"
    processed_dir = root / "data" / "processed"

    logger.info("=== Data preparation pipeline started ===")

    # 1️⃣ Load raw CSV (Kaggle fallback inside the function)
    df_raw = load_raw_data(raw_dir=raw_dir)

    # 2️⃣ Clean
    df_clean = clean_raw(df_raw)

    # 3️⃣ Save the clean version for debugging / reproducibility
    save_interim(df_clean, interim_dir=interim_dir)

    # 4️⃣ **Feature engineering**
    df_fe = engineer_features(df_clean)

    # 5️⃣ Encode target (original is "Yes"/"No")
    df_fe["Churn"] = df_fe["Churn"].apply(lambda v: 1 if v == "Yes" else 0)

    # 6️⃣ Train‑val split – stratify to preserve churn prevalence
    X = df_fe.drop(columns=["Churn"])
    y = df_fe["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # 7️⃣ Persist as parquet (fast, columnar, compression‑friendly)
    processed_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(processed_dir / "X_train.parquet", index=False)
    X_test.to_parquet(processed_dir / "X_test.parquet", index=False)
    y_train.to_frame(name="Churn").to_parquet(processed_dir / "y_train.parquet", index=False)
    y_test.to_frame(name="Churn").to_parquet(processed_dir / "y_test.parquet", index=False)

    logger.info(f"Saved train / test splits to {processed_dir}")
    logger.info("=== Data preparation pipeline finished ===")

if __name__ == "__main__":
    main()

