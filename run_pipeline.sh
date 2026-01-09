#!/bin/bash
set -e

echo "🚀 Starting Full Automation Pipeline..."

# Run Training
./run_training.sh

# Run Verification
echo "📓 Running Verification Notebook..."
jupyter nbconvert notebooks/02_preprocess.ipynb \
  --to notebook \
  --execute \
  --output 02_executed.ipynb \
  --output-dir notebooks

# Run Inference (App)
./run_inference.sh