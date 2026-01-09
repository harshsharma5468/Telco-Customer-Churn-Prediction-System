"""
Data ingestion utilities.

This module handles loading the raw Telco Customer Churn dataset
from local storage. It supports both CSV and Excel (.xlsx) formats
and works reliably in local development, Docker, and CI environments.
"""

from pathlib import Path
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_raw_data(
    raw_dir: Path | None = None,
    filename: str = "Telco_customer_churn.xlsx",
) -> pd.DataFrame:
    """
    Load the raw churn dataset from data/raw.

    Parameters
    ----------
    raw_dir : Path | None
        Directory containing raw data. Defaults to <repo_root>/data/raw.
    filename : str
        Preferred filename to load if present.

    Returns
    -------
    pd.DataFrame
        Loaded raw dataset.

    Raises
    ------
    FileNotFoundError
        If no CSV or XLSX files are found.
    ValueError
        If file type is unsupported.
    """

    # Resolve default raw directory safely
    if raw_dir is None:
        raw_dir = Path(__file__).resolve().parents[2] / "data" / "raw"

    raw_dir = raw_dir.resolve()
    raw_path = raw_dir / filename

    # 1️⃣ Check for preferred filename
    if not raw_path.is_file():
        logger.warning(
            f"Default file '{filename}' not found. "
            f"Searching {raw_dir} for any data file..."
        )

        # 2️⃣ Auto-discover any CSV or XLSX
        candidates = list(raw_dir.glob("*.csv")) + list(raw_dir.glob("*.xlsx"))

        if not candidates:
            error_msg = (
                f"No data files (.csv or .xlsx) found in {raw_dir}. "
                "Please place your dataset file inside data/raw/."
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        raw_path = candidates[0]
        logger.info(f"Using detected file: {raw_path.name}")

    # 3️⃣ Load file
    logger.info(f"Reading raw data from {raw_path}")

    try:
        if raw_path.suffix.lower() == ".csv":
            df = pd.read_csv(raw_path)
        elif raw_path.suffix.lower() == ".xlsx":
            df = pd.read_excel(raw_path)
        else:
            raise ValueError(f"Unsupported file type: {raw_path.suffix}")

        logger.info(f"Successfully loaded data. Shape: {df.shape}")
        return df

    except Exception as e:
        logger.exception(f"Failed to read raw data file: {e}")
        raise
