"""Configuration management and validation for KidneyCare-AI."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List
try:
    import yaml
except ImportError:
    yaml = None

from kidneycare.core.exceptions import ConfigurationError


@dataclass
class Config:
    """Consolidated KidneyCare-AI configuration object."""

    project: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    models: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    hardware: Dict[str, Any] = field(default_factory=dict)
    visualization: Dict[str, Any] = field(default_factory=dict)
    raw_configs: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate configuration settings for correctness and safety."""
        # Validate data splitting
        splitting = self.data.get("splitting", {})
        train_r = splitting.get("train_ratio", 0.7)
        val_r = splitting.get("val_ratio", 0.15)
        test_r = splitting.get("test_ratio", 0.15)
        total_r = round(train_r + val_r + test_r, 4)
        if total_r != 1.0:
            raise ConfigurationError(
                f"Data split ratios must sum to 1.0, got train={train_r}, val={val_r}, test={test_r} (sum={total_r})"
            )

        # Validate batch size
        batch_size = self.training.get("training", {}).get("batch_size", 32)
        if batch_size <= 0:
            raise ConfigurationError(f"batch_size must be positive, got {batch_size}")

        # Validate target column
        target_col = self.data.get("features", {}).get("target_column")
        if not target_col:
            raise ConfigurationError("Target column must be specified in data config")


def _parse_yaml_fallback(text: str) -> Dict[str, Any]:
    """Lightweight fallback parser for YAML configuration files when PyYAML is unavailable."""
    import re
    root: Dict[str, Any] = {}
    stack: list = [(root, -1)]

    for raw_line in text.splitlines():
        # Strip trailing comments
        line = re.sub(r"#.*$", "", raw_line).rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip())
        content = line.strip()

        # Pop stack to matching parent indent
        while len(stack) > 1 and stack[-1][1] >= indent:
            stack.pop()

        parent, _ = stack[-1]

        if content.startswith("- "):
            val_str = content[2:].strip().strip("\"'")
            if isinstance(parent, list):
                parent.append(_parse_val(val_str))
            elif isinstance(parent, dict) and stack[-1][0]:
                # find the last key added to parent dict
                last_key = list(parent.keys())[-1]
                if not isinstance(parent[last_key], list):
                    parent[last_key] = []
                parent[last_key].append(_parse_val(val_str))
        elif ":" in content:
            key, val_str = content.split(":", 1)
            key = key.strip().strip("\"'")
            val_str = val_str.strip()

            if not val_str:
                new_dict: Dict[str, Any] = {}
                if isinstance(parent, dict):
                    parent[key] = new_dict
                    stack.append((new_dict, indent))
            else:
                val = _parse_val(val_str)
                if isinstance(parent, dict):
                    parent[key] = val


def _parse_val(val_str: str) -> Any:
    val_str = val_str.strip()
    if val_str.lower() in ("true", "yes", "on"):
        return True
    if val_str.lower() in ("false", "no", "off"):
        return False
    if val_str.lower() in ("null", "none", "~", ""):
        return None
    try:
        return int(val_str)
    except ValueError:
        pass
    try:
        return float(val_str)
    except ValueError:
        pass
    return val_str.strip("\"'")


def load_yaml(file_path: Path) -> Dict[str, Any]:
    """Safely load a YAML file with PyYAML or built-in fallback parser."""
    if not file_path.exists():
        raise ConfigurationError(f"Configuration file not found: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            if yaml is not None:
                content = yaml.safe_load(text)
                return content if content is not None else {}
            else:
                parsed = _parse_yaml_fallback(text)
                return parsed if parsed is not None else {}
    except Exception as e:
        raise ConfigurationError(f"Error parsing YAML file {file_path}: {e}") from e


def load_config(config_dir: str = "configs") -> Config:
    """Load and merge all configuration files from the config directory.

    Args:
        config_dir: Directory containing YAML configuration files.

    Returns:
        Validated Config instance.
    """
    base = Path(config_dir)
    if not base.exists():
        raise ConfigurationError(f"Config directory does not exist: {config_dir}")

    project_cfg = load_yaml(base / "project.yaml") if (base / "project.yaml").exists() else {}
    data_cfg = load_yaml(base / "data.yaml") if (base / "data.yaml").exists() else {}
    models_cfg = load_yaml(base / "models.yaml") if (base / "models.yaml").exists() else {}
    training_cfg = load_yaml(base / "training.yaml") if (base / "training.yaml").exists() else {}
    hardware_cfg = load_yaml(base / "hardware.yaml") if (base / "hardware.yaml").exists() else {}
    viz_cfg = load_yaml(base / "visualization.yaml") if (base / "visualization.yaml").exists() else {}

    cfg = Config(
        project=project_cfg,
        data=data_cfg,
        models=models_cfg,
        training=training_cfg,
        hardware=hardware_cfg,
        visualization=viz_cfg,
        raw_configs={
            "project": project_cfg,
            "data": data_cfg,
            "models": models_cfg,
            "training": training_cfg,
            "hardware": hardware_cfg,
            "visualization": viz_cfg,
        },
    )
    cfg.validate()
    return cfg
