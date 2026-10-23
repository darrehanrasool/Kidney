"""Structured logging configuration with patient privacy safeguards and automatic persistent file logging."""

import logging
import re
import sys
from pathlib import Path
from typing import Optional

from kidneycare.utils.file_utils import ensure_dir

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"

# Patterns to sanitize/redact
SENSITIVE_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|password|secret|token|bearer)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{8,})['\"]?"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), # SSN pattern
]


class PrivacySafeFilter(logging.Filter):
    """Filter to redact sensitive keys, tokens, and potential patient identifiers."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = SENSITIVE_PATTERNS[0].sub(r"\1: [REDACTED]", record.msg)
            record.msg = SENSITIVE_PATTERNS[1].sub("[REDACTED-ID]", record.msg)
        return True


def _attach_file_handler(logger: logging.Logger, path: Path, formatter: logging.Formatter, filter_obj: logging.Filter, level: int = logging.INFO):
    """Attach a UTF-8 file handler safely."""
    ensure_dir(path.parent)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(formatter)
    handler.addFilter(filter_obj)
    logger.addHandler(handler)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_name: str = "kidneycare",
) -> logging.Logger:
    """Configure structured logging with console and persistent file output.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional explicit path to write log files.
        log_name: Root logger name.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(log_name)
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Clear existing handlers to prevent duplicate lines
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )
    privacy_filter = PrivacySafeFilter()

    # 1. Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(privacy_filter)
    logger.addHandler(console_handler)

    # 2. Main Application File Log (logs/application/kidneycare.log)
    app_log = LOGS_DIR / "application" / "kidneycare.log"
    _attach_file_handler(logger, app_log, formatter, privacy_filter, log_level)

    # 3. Error Log (logs/errors/errors.log)
    err_log = LOGS_DIR / "errors" / "errors.log"
    _attach_file_handler(logger, err_log, formatter, privacy_filter, logging.WARNING)

    # 4. Optional explicit log file
    if log_file:
        _attach_file_handler(logger, Path(log_file), formatter, privacy_filter, log_level)

    return logger


def get_logger(name: str = "kidneycare") -> logging.Logger:
    """Retrieve logger with appropriate hierarchy and dedicated category file handler."""
    logger = logging.getLogger(name)

    # If logger or root has no handlers, run setup_logging
    root_kidney = logging.getLogger("kidneycare")
    if not root_kidney.hasHandlers():
        setup_logging()

    # Attach category-specific file handlers if not already present
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )
    privacy_filter = PrivacySafeFilter()

    if "train" in name and not any(isinstance(h, logging.FileHandler) and "training" in str(getattr(h, "baseFilename", "")) for h in logger.handlers):
        train_log = LOGS_DIR / "training" / "training.log"
        _attach_file_handler(logger, train_log, formatter, privacy_filter)

    elif "hardware" in name and not any(isinstance(h, logging.FileHandler) and "hardware" in str(getattr(h, "baseFilename", "")) for h in logger.handlers):
        hw_log = LOGS_DIR / "hardware" / "hardware.log"
        _attach_file_handler(logger, hw_log, formatter, privacy_filter)

    elif ("data" in name or "download" in name or "validator" in name) and not any(isinstance(h, logging.FileHandler) and "datasets" in str(getattr(h, "baseFilename", "")) for h in logger.handlers):
        data_log = LOGS_DIR / "datasets" / "datasets.log"
        _attach_file_handler(logger, data_log, formatter, privacy_filter)

    return logger
