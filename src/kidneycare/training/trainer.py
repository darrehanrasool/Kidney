"""ClinicalTrainer orchestrates model training, per-epoch telemetry, and checkpointing."""

import time
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from kidneycare.hardware.device_manager import DeviceManager
from kidneycare.monitoring.logging_config import get_logger
from kidneycare.monitoring.progress_tracker import ProgressTracker
from kidneycare.training.checkpoint_manager import CheckpointManager
from kidneycare.training.early_stopping import EarlyStopping
from kidneycare.training.optimizer import get_optimizer
from kidneycare.training.scheduler import get_scheduler

logger = get_logger("kidneycare.trainer")


class ClinicalTrainer:
    """Orchestrates PyTorch neural network training loops with early stopping and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        model_name: str = "tabular_mlp",
    ):
        self.model_name = model_name
        self.config = config or {}
        self.device_manager = DeviceManager(preferred=device or "auto")
        self.device = self.device_manager.get_device()

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        training_cfg = self.config.get("training", {})
        opt_cfg = self.config.get("optimizer", {})
        sched_cfg = self.config.get("scheduler", {})
        es_cfg = self.config.get("early_stopping", {})

        self.epochs = training_cfg.get("epochs", 50)
        self.lr = opt_cfg.get("lr", 0.001)
        self.weight_decay = opt_cfg.get("weight_decay", 0.001)

        self.optimizer = get_optimizer(
            self.model,
            name=opt_cfg.get("name", "AdamW"),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = get_scheduler(
            self.optimizer,
            name=sched_cfg.get("name", "CosineAnnealingLR"),
            epochs=self.epochs,
        )

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 10),
            mode=es_cfg.get("mode", "max"),
            min_delta=es_cfg.get("min_delta", 0.001),
        )
        self.checkpoint_manager = CheckpointManager()
        self.progress_tracker = ProgressTracker()
        self.best_metric: float = -1.0

    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Run single training epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for X_batch, y_batch in self.train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * len(X_batch)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / len(self.train_loader.dataset)
        f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
        acc = float(accuracy_score(all_targets, all_preds))

        return avg_loss, {"loss": avg_loss, "f1": f1, "accuracy": acc}

    def evaluate(self, loader: Optional[DataLoader] = None) -> Tuple[float, Dict[str, float]]:
        """Evaluate model on a DataLoader (validation or test)."""
        data_loader = loader or self.val_loader
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)

                total_loss += loss.item() * len(X_batch)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=-1)

                all_probs.extend(probs[:, 1] if probs.shape[1] > 1 else probs[:, 0])
                all_preds.extend(preds)
                all_targets.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / len(data_loader.dataset)
        f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
        acc = float(accuracy_score(all_targets, all_preds))

        try:
            auc = float(roc_auc_score(all_targets, all_probs))
        except Exception:
            auc = 0.5

        metrics = {
            "loss": avg_loss,
            "f1": f1,
            "accuracy": acc,
            "roc_auc": auc,
        }
        return avg_loss, metrics

    def train(self) -> Dict[str, Any]:
        """Execute complete multi-epoch training loop with early stopping and checkpointing."""
        logger.info(f"Starting training for {self.model_name} on {self.device.type.upper()} ({self.epochs} epochs)...")
        start_time = time.perf_counter()

        for epoch in range(1, self.epochs + 1):
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.evaluate(self.val_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                self.scheduler.step()

            # Record per-epoch metrics
            self.progress_tracker.record_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                lr=current_lr,
            )

            monitored_val = val_metrics.get("f1", 0.0)
            is_best = monitored_val > self.best_metric
            if is_best:
                self.best_metric = monitored_val

            # Save checkpoint
            self.checkpoint_manager.save_neural_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                metrics=val_metrics,
                is_best=is_best,
                config=self.config,
                early_stopping_state=self.early_stopping.state_dict(),
                model_name=self.model_name,
            )

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch:02d}/{self.epochs:02d} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val F1: {val_metrics['f1']:.4f} - Val AUC: {val_metrics['roc_auc']:.4f}"
                )

            # Early stopping check
            if self.early_stopping.step(monitored_val):
                logger.info(f"Early stopping triggered at epoch {epoch} (patience reached).")
                break

        total_time = time.perf_counter() - start_time
        logger.info(f"Training completed for {self.model_name} in {total_time:.2f}s. Best Val F1: {self.best_metric:.4f}")

        return {
            "best_metric": self.best_metric,
            "total_epochs": epoch,
            "training_time_seconds": round(total_time, 2),
            "history": self.progress_tracker.to_dict(),
        }
