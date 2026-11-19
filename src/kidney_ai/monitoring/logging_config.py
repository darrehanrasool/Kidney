"""Logging setup, system telemetry, and experiment tracking."""

import logging
import sys
from pathlib import Path


def setup_logging(log_dir: Path = None, level: int = logging.INFO) -> logging.Logger:
    """Configures structured stream and file logging."""
    if log_dir is None:
        log_dir = Path("logs/application")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("kidney_ai")
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(log_dir / "kidney_ai.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
