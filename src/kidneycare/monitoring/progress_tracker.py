"""Epoch and step-level progress tracking during model training."""

from typing import Dict, List, Any


class ProgressTracker:
    """Records per-epoch training and validation loss and metrics history."""

    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def record_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float,
    ) -> None:
        """Store metrics record for a completed training epoch."""
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "lr": lr,
            "train_metrics": {k: round(v, 4) for k, v in train_metrics.items()},
            "val_metrics": {k: round(v, 4) for k, v in val_metrics.items()},
        }
        self.history.append(record)

    def get_metric_series(self, metric_name: str, split: str = "val") -> List[float]:
        """Extract a single metric series across epochs."""
        key = f"{split}_metrics"
        return [h.get(key, {}).get(metric_name, 0.0) for h in self.history]

    def get_loss_series(self, split: str = "val") -> List[float]:
        """Extract loss series across epochs."""
        key = f"{split}_loss"
        return [h.get(key, 0.0) for h in self.history]

    def to_dict(self) -> Dict[str, Any]:
        """Export history to dictionary."""
        return {"epochs": len(self.history), "history": self.history}
