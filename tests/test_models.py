import os
import sys
import subprocess
from pathlib import Path
import pytest

def test_training_creates_artifacts(tmp_path, monkeypatch):
    """
    End‑to‑end sanity check:
    * Runs `src.models.train` in a subprocess to avoid polluting state.
    * Verifies that the model pipelines and metrics JSON exist.
    """
    # Fix: Path(__file__).resolve() is /.../churn_prediction/tests/test_models.py
    # parents[0] is tests/
    # parents[1] is churn_prediction/ (The correct repo root)
    repo_root = Path(__file__).resolve().parents[1]
    
    # Ensure the repo root is in the PYTHONPATH so 'src' is findable
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    

    # Run the training script as a module
    result = subprocess.run(
        [sys.executable, "-m", "src.models.train"],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )

    # 1. Assert the process finished successfully
    assert result.returncode == 0, f"Training failed with error: {result.stderr}"

    # 2. Check if the artifacts directory was created
    model_dir = repo_root / "models"
    assert model_dir.exists(), "The 'models' directory was not created."

    # 3. Check for the expected model files
    expected_files = [
        "logistic_regression_pipeline.joblib",
        "random_forest_pipeline.joblib",
        "xgboost_pipeline.joblib",
        "metrics.json"
    ]
    
    for filename in expected_files:
        file_path = model_dir / filename
        assert file_path.exists(), f"Artifact {filename} is missing from {model_dir}"

    print("\n✅ All training artifacts verified successfully.")