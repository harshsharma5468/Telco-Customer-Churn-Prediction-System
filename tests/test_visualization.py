import pytest
from pathlib import Path
from src.visualization.plots import generate_evaluation_plots

def test_visualization_creates_plots():
    """
    Verifies that running the visualization script generates 
    the Confusion Matrix and ROC Curve image files.
    """
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "reports" / "figures"
    
    # 1. Run the plot generator
    generate_evaluation_plots()
    
    # 2. Check for file existence
    assert (output_dir / "confusion_matrix.png").exists()
    assert (output_dir / "roc_curve.png").exists()
    
    print("\n✅ Visualization artifacts verified.")