"""Unified training entrypoint for classical and deep learning models."""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score

from kidneycare.models.model_factory import ModelFactory
from kidneycare.models.deep_learning.training_utils import create_data_loaders
from kidneycare.training.trainer import ClinicalTrainer
from kidneycare.training.checkpoint_manager import CheckpointManager
from kidneycare.monitoring.logging_config import get_logger

logger = get_logger("kidneycare.train")


def train_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
    preprocessor: Optional[Any] = None,
    feature_names: Optional[list] = None,
    output_dir: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Train either a classical estimator or a deep learning architecture.

    Args:
        model_name: Name of model.
        X_train: Training feature matrix.
        y_train: Training labels.
        X_val: Validation feature matrix.
        y_val: Validation labels.
        config: Full configuration dictionary.
        preprocessor: Fitted ClinicalPreprocessor.
        feature_names: List of encoded feature names.

    Returns:
        Tuple of (trained_model, results_dict).
    """
    cfg = config or {}
    model_params = cfg.get("models", {}).get("classical", {}).get(model_name, {})
    checkpoint_mgr = CheckpointManager()

    if ModelFactory.is_deep_learning(model_name):
        dl_cfg = cfg.get("models", {}).get("deep_learning", {}).get(model_name, {})
        input_dim = X_train.shape[1]
        model = ModelFactory.create_model(model_name, config=dl_cfg, input_dim=input_dim)

        batch_size = cfg.get("training", {}).get("training", {}).get("batch_size", 32)
        train_loader, val_loader, _ = create_data_loaders(
            X_train, y_train, X_val, y_val, batch_size=batch_size
        )

        trainer = ClinicalTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=cfg,
            model_name=model_name,
        )
        train_res = trainer.train()
        _, val_metrics = trainer.evaluate(val_loader)
        train_res["val_metrics"] = val_metrics
        return model, train_res

    else:
        # Classical model training
        model = ModelFactory.create_model(model_name, config=model_params)
        logger.info(f"Fitting classical estimator {model_name} on {len(X_train)} samples...")
        model.fit(X_train, y_train)

        # Validation evaluation
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else y_pred

        val_acc = float(accuracy_score(y_val, y_pred))
        val_f1 = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
        val_recall = float(recall_score(y_val, y_pred, average="macro", zero_division=0))
        try:
            val_auc = float(roc_auc_score(y_val, y_prob))
        except Exception:
            val_auc = 0.5

        val_metrics = {
            "accuracy": val_acc,
            "f1": val_f1,
            "recall": val_recall,
            "roc_auc": val_auc,
        }

        # Save classical model bundle
        saved_path = checkpoint_mgr.save_classical_model(
            model=model,
            model_name=model_name,
            metrics=val_metrics,
            preprocessor=preprocessor,
            feature_names=feature_names,
            out_dir=output_dir,
        )

        logger.info(
            f"Fitted {model_name}: Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}, Val AUC={val_auc:.4f} -> {saved_path}"
        )

        return model, {"metrics": val_metrics, "model_path": str(saved_path)}
