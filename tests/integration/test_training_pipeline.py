"""Integration test for KidneyCare-AI training, validation, and evaluation loops."""

import tempfile
from pathlib import Path
import pytest
import numpy as np
import torch

from kidneycare.models.model_factory import ModelFactory
from kidneycare.training.train import train_model
from kidneycare.evaluation.evaluation import ModelEvaluator


def test_classical_training_and_eval_integration():
    np.random.seed(42)
    X_train = np.random.randn(80, 20).astype(np.float32)
    y_train = np.random.choice([0, 1], size=80)

    X_val = np.random.randn(20, 20).astype(np.float32)
    y_val = np.random.choice([0, 1], size=20)

    X_test = np.random.randn(20, 20).astype(np.float32)
    y_test = np.random.choice([0, 1], size=20)

    model, train_res = train_model(
        model_name="random_forest",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )
    assert "metrics" in train_res

    evaluator = ModelEvaluator(optimal_threshold=0.5)
    test_res = evaluator.evaluate_test_set(model, X_test, y_test, model_name="RandomForest")
    assert "metrics" in test_res
    assert test_res["metrics"]["sample_size"] == 20
    assert 0.0 <= test_res["metrics"]["accuracy"] <= 1.0


def test_deep_learning_training_and_eval_integration():
    torch.manual_seed(42)
    np.random.seed(42)
    X_train = np.random.randn(64, 16).astype(np.float32)
    y_train = np.random.choice([0, 1], size=64)

    X_val = np.random.randn(16, 16).astype(np.float32)
    y_val = np.random.choice([0, 1], size=16)

    model, train_res = train_model(
        model_name="tabular_mlp",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        config={"training": {"training": {"epochs": 3, "batch_size": 16}}},
    )
    assert "best_metric" in train_res
    assert train_res["total_epochs"] == 3
