import torch
import numpy as np
from kidney_ai.models.model_factory import ModelFactory


def test_classical_models():
    rf = ModelFactory.create_model("random_forest")
    assert hasattr(rf, "fit")

    lr = ModelFactory.create_model("logistic_regression")
    assert hasattr(lr, "fit")


def test_neural_models():
    mlp = ModelFactory.create_model("tabular_mlp", {"input_dim": 10, "output_dim": 2})
    x = torch.randn(4, 10)
    out = mlp(x)
    assert out.shape == (4, 2)

    resnet = ModelFactory.create_model("tabular_resnet", {"input_dim": 10, "output_dim": 2})
    out_res = resnet(x)
    assert out_res.shape == (4, 2)
