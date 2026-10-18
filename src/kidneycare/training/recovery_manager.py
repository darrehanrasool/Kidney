"""Fault recovery and training resumption manager."""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import joblib

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from kidneycare.core.exceptions import CheckpointCorruptError
from kidneycare.monitoring.logging_config import get_logger

logger = get_logger("kidneycare.recovery")


class RecoveryManager:
    """Safely recovers models, optimizers, and training state from checkpoints."""

    def __init__(self, checkpoints_dir: str = "checkpoints"):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.latest_dir = self.checkpoints_dir / "latest"
        self.best_dir = self.checkpoints_dir / "best"

    def has_checkpoint(self, model_name: str = "tabular_mlp", use_best: bool = False) -> bool:
        """Check if a valid checkpoint exists for the model."""
        target_dir = self.best_dir if use_best else self.latest_dir
        path = target_dir / f"{model_name}_{'best' if use_best else 'latest'}.pt"
        return path.exists() and path.stat().st_size > 0

    def load_neural_checkpoint(
        self,
        model: Any,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        model_name: str = "tabular_mlp",
        use_best: bool = False,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Safely restore PyTorch training state from checkpoint file.

        Args:
            model: PyTorch module into which weights will be loaded.
            optimizer: Optional optimizer to restore state into.
            scheduler: Optional scheduler to restore state into.
            model_name: Model name.
            use_best: If True, load from best/ directory, otherwise latest/.
            device: Target execution device string ('cpu', 'cuda', 'mps').

        Returns:
            Dictionary containing metadata, epoch, and metric history from checkpoint.
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for loading neural network checkpoints.")

        target_dir = self.best_dir if use_best else self.latest_dir
        suffix = "best" if use_best else "latest"
        path = target_dir / f"{model_name}_{suffix}.pt"

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file does not exist: {path}")

        try:
            checkpoint = torch.load(path, map_location=device)
        except Exception as e:
            raise CheckpointCorruptError(f"Corrupted or unreadable checkpoint at {path}: {e}") from e

        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise CheckpointCorruptError(f"Invalid checkpoint format at {path}: missing required state keys")

        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded model state from {path} (Epoch {checkpoint.get('epoch', 0)})")

        if optimizer is not None and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Restored optimizer state from checkpoint")

        if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.info("Restored learning rate scheduler state")

        # Restore RNG state if compatible
        if "rng_state" in checkpoint and checkpoint["rng_state"] is not None:
            try:
                torch.set_rng_state(checkpoint["rng_state"])
            except Exception:
                pass

        return {
            "epoch": checkpoint.get("epoch", 0),
            "metrics": checkpoint.get("metrics", {}),
            "early_stopping_state": checkpoint.get("early_stopping_state"),
            "config": checkpoint.get("config", {}),
        }

    def load_classical_model(self, model_name: str) -> Dict[str, Any]:
        """Safely load a classical model bundle from outputs/trained_models."""
        path = Path("outputs/trained_models") / f"{model_name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Classical model file does not exist: {path}")
        try:
            return joblib.load(path)
        except Exception as e:
            raise CheckpointCorruptError(f"Failed to load classical model from {path}: {e}") from e
