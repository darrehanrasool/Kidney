"""Model factory instantiating classical, neural, and ensemble model architectures."""

from typing import Any, Dict
from kidney_ai.core.exceptions import ModelNotFoundError
from kidney_ai.models.classical import (
    build_logistic_regression,
    build_random_forest,
    build_gradient_boosting,
    build_svm,
)
from kidney_ai.models.deep_learning import TabularMLP, TabularResNet


class ModelFactory:
    """Instantiates models from string identifiers."""

    @staticmethod
    def create_model(model_name: str, config: Dict[str, Any] = None) -> Any:
        cfg = config or {}
        name = model_name.lower().replace("-", "_")

        if name in ("logistic_regression", "lr"):
            return build_logistic_regression(cfg)
        elif name in ("random_forest", "rf"):
            return build_random_forest(cfg)
        elif name in ("gradient_boosting", "gbdt", "xgboost"):
            return build_gradient_boosting(cfg)
        elif name in ("svm", "svc"):
            return build_svm(cfg)
        elif name in ("tabular_mlp", "mlp"):
            input_dim = cfg.get("input_dim", 28)
            hidden_dims = cfg.get("hidden_dims", [128, 64, 32])
            output_dim = cfg.get("output_dim", 2)
            dropout = cfg.get("dropout", 0.25)
            return TabularMLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, dropout=dropout)
        elif name in ("tabular_resnet", "resnet"):
            input_dim = cfg.get("input_dim", 28)
            hidden_dim = cfg.get("hidden_dim", 128)
            num_blocks = cfg.get("num_blocks", 3)
            output_dim = cfg.get("output_dim", 2)
            dropout = cfg.get("dropout", 0.2)
            return TabularResNet(input_dim=input_dim, hidden_dim=hidden_dim, num_blocks=num_blocks, output_dim=output_dim, dropout=dropout)
        else:
            raise ModelNotFoundError(f"Model architecture '{model_name}' not registered in ModelFactory.")
