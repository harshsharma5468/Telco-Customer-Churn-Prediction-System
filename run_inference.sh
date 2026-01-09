#!/bin/bash
set -e

echo "🧪 Running Pytest..."
pytest -q tests/test_models.py

echo "🌐 Starting Streamlit Dashboard..."
streamlit run src/app/main.py --server.address=0.0.0.0