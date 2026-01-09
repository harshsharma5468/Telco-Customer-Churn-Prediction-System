# src/utils/logger.py
"""
Centralised logger using the standard ``logging`` package.

* Loads configuration from ``config/logging.yaml``.
* Automatically converts relative log paths to absolute paths based on PROJECT_ROOT.
* Guarantees that the ``logs/`` directory exists.
"""

import logging
import logging.config
from pathlib import Path
import yaml

# 1️⃣ Resolve project‑root (three parents up from src/utils/logger.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CFG_PATH = PROJECT_ROOT / "config" / "logging.yaml"

def setup_logging():
    """Load config and ensure log paths are absolute to avoid FileNotFoundError in notebooks."""
    if _CFG_PATH.is_file():
        with _CFG_PATH.open() as f:
            config = yaml.safe_load(f)

        # 2️⃣ Fix relative paths in handlers
        # This prevents notebooks from looking for logs/ inside the notebooks/ folder
        handlers = config.get("handlers", {})
        for handler_name, handler_cfg in handlers.items():
            if "filename" in handler_cfg:
                # Convert relative 'logs/app.log' to absolute '/path/to/project/logs/app.log'
                original_path = Path(handler_cfg["filename"])
                if not original_path.is_absolute():
                    absolute_path = PROJECT_ROOT / original_path
                    handler_cfg["filename"] = str(absolute_path)
                
                # 3️⃣ Create the directory for this log file if it doesn't exist
                Path(handler_cfg["filename"]).parent.mkdir(parents=True, exist_ok=True)

        logging.config.dictConfig(config)
    else:
        # Fallback if config/logging.yaml is missing
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

# Initialize logging immediately when the module is imported
setup_logging()

def get_logger(name: str = "churn") -> logging.Logger:
    """Return a logger instance configured with project standards."""
    return logging.getLogger(name)