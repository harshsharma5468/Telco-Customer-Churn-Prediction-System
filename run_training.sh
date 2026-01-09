#!/bin/bash
set -e

echo "🧹 Cleaning and Preprocessing..."
python -m src.data.preprocess_and_save

echo "🧠 Training Models..."
python -m src.models.train

echo "📊 Generating Plots..."
python -m src.visualization.plots

echo "✅ Training Complete!"