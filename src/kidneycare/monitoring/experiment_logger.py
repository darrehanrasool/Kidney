"""Experiment logging and tracking for model comparison."""

import csv
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

from kidneycare.utils.file_utils import ensure_dir, atomic_write_json


class ExperimentLogger:
    """Logs experiment parameters, evaluation metrics, and artifact references."""

    def __init__(self, log_dir: str = "experiments", registry_csv: str = "experiments/experiment_registry.csv"):
        self.log_dir = Path(log_dir)
        self.registry_csv = Path(registry_csv)
        ensure_dir(self.log_dir)
        ensure_dir(self.registry_csv.parent)
        self._init_registry()

    def _init_registry(self) -> None:
        """Initialize the experiment registry CSV if not present."""
        if not self.registry_csv.exists() or self.registry_csv.stat().st_size == 0:
            headers = [
                "experiment_id",
                "timestamp",
                "model_name",
                "dataset",
                "val_accuracy",
                "val_f1",
                "val_roc_auc",
                "val_sensitivity",
                "val_specificity",
                "status",
                "checkpoint_path",
            ]
            with open(self.registry_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def log_experiment(
        self,
        experiment_id: str,
        model_name: str,
        dataset: str,
        metrics: Dict[str, Any],
        params: Dict[str, Any],
        status: str = "SUCCESS",
        checkpoint_path: str = "",
    ) -> None:
        """Log a completed experiment run."""
        now = time.strftime("%Y-%m-%d %H:%M:%S")

        # Save individual experiment JSON
        exp_record = {
            "experiment_id": experiment_id,
            "timestamp": now,
            "model_name": model_name,
            "dataset": dataset,
            "status": status,
            "metrics": metrics,
            "params": params,
            "checkpoint_path": checkpoint_path,
        }
        exp_file = self.log_dir / f"{experiment_id}.json"
        atomic_write_json(exp_file, exp_record)

        # Append to registry CSV
        row = [
            experiment_id,
            now,
            model_name,
            dataset,
            metrics.get("accuracy", ""),
            metrics.get("f1", ""),
            metrics.get("roc_auc", ""),
            metrics.get("recall", metrics.get("sensitivity", "")),
            metrics.get("specificity", ""),
            status,
            checkpoint_path,
        ]
        with open(self.registry_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
