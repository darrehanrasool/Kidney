"""Unit tests for KidneyCare-AI classical, ensemble, and deep learning models."""

import numpy as np
import torch
import pytest

from kidneycare.models.model_factory import ModelFactory


def test_classical_models():
    np.random.seed(42)
    X = np.random.randn(60, 15).astype(np.float32)
    y = np.random.choice([0, 1], size=60)

    for name in ["logistic_regression", "random_forest", "gradient_boosting", "svm", "voting_ensemble"]:
        model = ModelFactory.create_model(name)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (60,)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            assert probs.shape == (60, 2)
            assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_deep_learning_tabular_mlp():
    torch.manual_seed(42)
    x = torch.randn(16, 28)

    mlp = ModelFactory.create_model(
        "tabular_mlp",
        config={"hidden_dims": [32, 16], "dropout_rates": [0.1, 0.1], "activation": "relu", "output_dim": 2},
        input_dim=28,
    )
    logits = mlp(x)
    assert logits.shape == (16, 2)

    probs = mlp.predict_proba(x)
    assert probs.shape == (16, 2)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(16), atol=1e-5)
