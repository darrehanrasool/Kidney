"""Structured logging configuration with patient privacy safeguards."""

import logging
import re
import sys
from pathlib import Path
from typing import Optional

from kidneycare.utils.file_utils import ensure_dir

# Patterns to sanitize/redact
SENSITIVE_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|password|secret|token|bearer)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{8,})['\"]?"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), # SSN pattern
]


class PrivacySafeFilter(logging.Filter):
    """Filter to redact sensitive keys, tokens, and potential patient identifiers."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            for pattern in SENSITIVE_PATTERNS:
                record.msg = pattern.sub(r"\1: [REDACTED]", record.msg)
        return True


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_name: str = "kidneycare",
) -> logging.Logger:
    """Configure structured logging with console and optional file output.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to write log files.
        log_name: Root logger name.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers to prevent duplicate lines
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )
    privacy_filter = PrivacySafeFilter()

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(privacy_filter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        p = Path(log_file)
        ensure_dir(p.parent)
        file_handler = logging.FileHandler(p, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(privacy_filter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "kidneycare") -> logging.Logger:
    """Retrieve logger with appropriate hierarchy."""
    return logging.getLogger(name)
