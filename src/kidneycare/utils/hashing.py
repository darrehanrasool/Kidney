"""Cryptographic hashing utilities for data verification and reproducibility."""

import hashlib
from pathlib import Path
from typing import Union, Any
try:
    import pandas as pd
except ImportError:
    pd = None


def calculate_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Calculate SHA-256 checksum of a file using chunked streaming.

    Args:
        file_path: Path to file.
        chunk_size: Byte chunk size for streaming.

    Returns:
        Hexadecimal SHA-256 string.
    """
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"File not found for checksum calculation: {p}")

    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def hash_dataframe(df: Any) -> str:
    """Calculate a deterministic hash for a pandas DataFrame."""
    if pd is not None and isinstance(df, pd.DataFrame):
        hashed_series = pd.util.hash_pandas_object(df, index=True)
        combined = hashed_series.values.tobytes()
        return hashlib.sha256(combined).hexdigest()
    return hashlib.sha256(str(df).encode("utf-8")).hexdigest()
