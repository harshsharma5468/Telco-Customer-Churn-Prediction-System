import pandas as pd
import os

# Check if the file exists before reading
file_path = 'data/processed/X_train.parquet'

if os.path.exists(file_path):
    X_train = pd.read_parquet(file_path)
    print("Successfully loaded X_train!")
    print(X_train.head())
else:
    print(f"File not found at: {file_path}. Did you run the preprocessing script?")