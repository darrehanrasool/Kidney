"""Model Factory for instantiating classical, deep learning, and ensemble architectures."""

from typing import Dict, Any, Optional
import torch.nn as nn

from kidneycare.models.classical.logistic_regression import get_logistic_regression
from kidneycare.models.classical.random_forest import get_random_forest
from kidneycare.models.classical.svm import get_svm
from kidneycare.models.classical.gradient_boosting import get_gradient_boosting
from kidneycare.models.deep_learning.tabular_mlp import TabularMLP
from kidneycare.models.ensemble.voting_ensemble import get_voting_ensemble


class ModelFactory:
    """Central factory for building and configuring KidneyCare-AI models."""

    @staticmethod
    def create_model(model_name: str, config: Optional[Dict[str, Any]] = None, input_dim: int = 24) -> Any:
        """Instantiate requested model by identifier.

        Args:
            model_name: Identifier ('logistic_regression', 'random_forest', 'svm',
                        'gradient_boosting', 'tabular_mlp', 'voting_ensemble').
            config: Optional dictionary containing model hyperparameters.
            input_dim: Number of input features (required for deep learning architectures).

        Returns:
            Configured estimator or nn.Module.
        """
        name = model_name.lower().replace("-", "_")
        cfg = config or {}

        if name == "logistic_regression":
            return get_logistic_regression(cfg)
        elif name == "random_forest":
            return get_random_forest(cfg)
        elif name == "svm":
            return get_svm(cfg)
        elif name in ["gradient_boosting", "gbm"]:
            return get_gradient_boosting(cfg)
        elif name in ["tabular_mlp", "mlp", "deep_mlp"]:
            hidden_dims = cfg.get("hidden_dims", [64, 32])
            dropouts = cfg.get("dropout_rates", [0.2, 0.1])
            activation = cfg.get("activation", "relu")
            use_batch_norm = cfg.get("use_batch_norm", True)
            use_residual = cfg.get("use_residual", False)
            output_dim = cfg.get("output_dim", 2)
            return TabularMLP(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                dropout_rates=dropouts,
                activation=activation,
                use_batch_norm=use_batch_norm,
                use_residual=use_residual,
            )
        elif name in ["voting_ensemble", "voting"]:
            voting = cfg.get("voting", "soft")
            weights = cfg.get("weights")
            return get_voting_ensemble(voting=voting, weights=weights)
        else:
            raise ValueError(
                f"Unknown model architecture '{model_name}'. Supported options: "
                f"['logistic_regression', 'random_forest', 'svm', 'gradient_boosting', 'tabular_mlp', 'voting_ensemble']"
            )

    @staticmethod
    def is_deep_learning(model_name: str) -> bool:
        """Check if model is a PyTorch deep neural network."""
        name = model_name.lower().replace("-", "_")
        return name in ["tabular_mlp", "mlp", "deep_mlp"]
