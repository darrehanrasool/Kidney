"""Filesystem and atomic file operations."""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Union


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, creating parents if necessary."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_text(path: Union[str, Path], text: str, encoding: str = "utf-8") -> Path:
    """Write text atomically using a temporary file and atomic rename."""
    target = Path(path)
    ensure_dir(target.parent)
    # Write to temp file in same directory to guarantee same filesystem for atomic rename
    with tempfile.NamedTemporaryFile("w", dir=target.parent, delete=False, encoding=encoding) as tf:
        tf.write(text)
        temp_name = tf.name
    shutil.move(temp_name, target)
    return target


def atomic_write_json(path: Union[str, Path], data: Any, indent: int = 2) -> Path:
    """Write JSON data atomically."""
    content = json.dumps(data, indent=indent, default=str)
    return atomic_write_text(path, content)


def safe_load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Safely load JSON data from a file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file does not exist: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
