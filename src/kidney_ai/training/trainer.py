"""PyTorch training execution loop with validation and checkpointing."""

from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score
from kidney_ai.training.optimizer import FocalLoss, build_optimizer
from kidney_ai.training.scheduler import build_scheduler
from kidney_ai.training.early_stopping import EarlyStopping
from kidney_ai.training.checkpoint_manager import CheckpointManager
from kidney_ai.hardware.device_manager import get_torch_device


class PyTorchTrainer:
    """Trains deep learning architectures on clinical tabular matrices."""

    def __init__(self, model: nn.Module, device: str = "auto", lr: float = 0.001, epochs: int = 50, batch_size: int = 64):
        self.device = get_torch_device(device)
        self.model = model.to(self.device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = FocalLoss()
        self.optimizer = build_optimizer(self.model, lr=lr)
        self.scheduler = build_scheduler(self.optimizer)
        self.early_stopping = EarlyStopping(patience=8)
        self.checkpoint_manager = CheckpointManager()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, model_name: str = "tabular_dl") -> Dict[str, list]:
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_accuracy": []}
        best_val_f1 = 0.0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                running_loss += loss.item() * len(batch_y)

            epoch_train_loss = running_loss / len(train_dataset)

            # Validation
            val_loss, val_acc, val_f1 = self._evaluate(val_loader)
            self.scheduler.step()

            history["train_loss"].append(epoch_train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)
            history["val_f1"].append(val_f1)

            is_best = val_f1 > best_val_f1
            if is_best:
                best_val_f1 = val_f1

            self.checkpoint_manager.save_checkpoint(
                self.model, epoch, {"val_f1": val_f1, "val_loss": val_loss}, model_name, is_best=is_best
            )

            if self.early_stopping(val_f1):
                break

        return history

    def _evaluate(self, loader: DataLoader) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        preds_all = []
        targets_all = []

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                total_loss += loss.item() * len(batch_y)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                preds_all.extend(preds)
                targets_all.extend(batch_y.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        acc = accuracy_score(targets_all, preds_all)
        f1 = f1_score(targets_all, preds_all, average="macro", zero_division=0)
        return avg_loss, acc, f1

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.FloatTensor(X).to(self.device)
            logits = self.model(tensor_x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs
