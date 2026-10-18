"""Robust checkpoint management with atomic persistence."""

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
import joblib

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from kidneycare.core.constants import (
    CHECKPOINT_LATEST_DIR,
    CHECKPOINT_BEST_DIR,
    CHECKPOINT_PERIODIC_DIR,
    CHECKPOINT_METADATA_DIR,
)
from kidneycare.core.exceptions import CheckpointCorruptError
from kidneycare.utils.file_utils import ensure_dir, atomic_write_json
from kidneycare.monitoring.logging_config import get_logger

logger = get_logger("kidneycare.checkpoints")


class CheckpointManager:
    """Manages atomic saving, indexing, and verification of model checkpoints."""

    def __init__(self, base_dir: str = "checkpoints"):
        self.base_dir = Path(base_dir)
        self.latest_dir = self.base_dir / "latest"
        self.best_dir = self.base_dir / "best"
        self.periodic_dir = self.base_dir / "periodic"
        self.metadata_dir = self.base_dir / "metadata"

        ensure_dir(self.latest_dir)
        ensure_dir(self.best_dir)
        ensure_dir(self.periodic_dir)
        ensure_dir(self.metadata_dir)

    def save_neural_checkpoint(
        self,
        model: Any,
        optimizer: Any,
        scheduler: Any,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        config: Optional[Dict[str, Any]] = None,
        early_stopping_state: Optional[Dict[str, Any]] = None,
        model_name: str = "tabular_mlp",
    ) -> Path:
        """Atomically persist PyTorch model training state.

        Args:
            model: PyTorch module.
            optimizer: PyTorch optimizer.
            scheduler: PyTorch LR scheduler.
            epoch: Current training epoch.
            metrics: Validation and training metrics dictionary.
            is_best: True if this checkpoint achieves highest validation score.
            config: Optional experiment configuration dictionary.
            early_stopping_state: Serialized state of EarlyStopping tracker.
            model_name: Base name for file identification.

        Returns:
            Path to latest saved checkpoint file.
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required to save neural network checkpoints.")

        checkpoint_data = {
            "format_version": "1.0.0",
            "model_name": model_name,
            "epoch": epoch,
            "timestamp": time.time(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "metrics": metrics,
            "early_stopping_state": early_stopping_state,
            "rng_state": torch.get_rng_state(),
            "config": config or {},
        }

        # Write to temporary file in latest_dir first
        temp_file = tempfile.NamedTemporaryFile("wb", dir=self.latest_dir, delete=False, suffix=".pt")
        temp_path = Path(temp_file.name)
        temp_file.close()

        try:
            torch.save(checkpoint_data, temp_path)
            # Verify file is readable before atomic rename
            _ = torch.load(temp_path, map_location="cpu")
        except Exception as e:
            if temp_path.exists():
                os.remove(temp_path)
            raise CheckpointCorruptError(f"Failed to write/verify checkpoint: {e}") from e

        latest_path = self.latest_dir / f"{model_name}_latest.pt"
        shutil.move(str(temp_path), str(latest_path))

        # Save metadata JSON
        meta = {
            "model_name": model_name,
            "epoch": epoch,
            "is_best": is_best,
            "metrics": metrics,
            "latest_file": str(latest_path),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        atomic_write_json(self.metadata_dir / f"{model_name}_metadata.json", meta)

        if is_best:
            best_path = self.best_dir / f"{model_name}_best.pt"
            shutil.copy2(str(latest_path), str(best_path))
            logger.info(f"Saved new BEST checkpoint for {model_name} at epoch {epoch} -> {best_path}")

        return latest_path

    def save_classical_model(
        self,
        model: Any,
        model_name: str,
        metrics: Dict[str, float],
        preprocessor: Optional[Any] = None,
        feature_names: Optional[list] = None,
    ) -> Path:
        """Atomically persist scikit-learn estimator and fitted preprocessor."""
        out_dir = Path("outputs/trained_models")
        ensure_dir(out_dir)

        payload = {
            "model_name": model_name,
            "model": model,
            "metrics": metrics,
            "feature_names": feature_names or [],
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        temp_file = tempfile.NamedTemporaryFile("wb", dir=out_dir, delete=False, suffix=".pkl")
        temp_path = Path(temp_file.name)
        temp_file.close()

        joblib.dump(payload, temp_path)
        final_path = out_dir / f"{model_name}.pkl"
        shutil.move(str(temp_path), str(final_path))

        if preprocessor is not None:
            prep_path = out_dir / "preprocessor.pkl"
            joblib.dump(preprocessor, prep_path)

        logger.info(f"Saved classical model {model_name} to {final_path}")
        return final_path
