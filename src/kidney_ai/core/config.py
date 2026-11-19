"""Hierarchical configuration loader supporting YAML files and dot notation."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class Config:
    """Dictionary wrapper enabling dot-notation key lookups."""

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self._data = data or {}

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split(".")
        val = self._data
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def __getitem__(self, item: str) -> Any:
        return self._data[item]

    def __contains__(self, item: str) -> bool:
        return item in self._data

    def to_dict(self) -> Dict[str, Any]:
        return self._data


def load_config(config_dir: Optional[Path] = None) -> Config:
    """Loads all YAML configurations from the configs directory."""
    if config_dir is None:
        # Resolve relative to project root
        config_dir = Path(__file__).resolve().parent.parent.parent.parent / "configs"

    merged: Dict[str, Any] = {}
    if config_dir.exists():
        for yml_file in sorted(config_dir.glob("*.yaml")):
            try:
                with open(yml_file, "r") as f:
                    content = yaml.safe_load(f) or {}
                    merged.update(content)
            except Exception as e:
                print(f"Warning: Failed to load {yml_file}: {e}")

    return Config(merged)
