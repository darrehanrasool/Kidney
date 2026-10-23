"""Multi-task homoscedastic uncertainty loss weighting (Kendall & Gal, 2018).

Optimizes loss weights dynamically across multiple clinical task heads:
    L_total = sum_i [ 0.5 * exp(-s_i) * L_i + 0.5 * s_i ]
where s_i = log(sigma_i^2) is a learned task variance parameter.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class MultiTaskUncertaintyLoss(nn.Module):
    """Dynamic multi-task loss weighting with homoscedastic task uncertainties."""

    def __init__(self, task_names: List[str]):
        super().__init__()
        self.task_names = task_names
        # Log variance parameter per task
        self.log_vars = nn.ParameterDict({
            task: nn.Parameter(torch.zeros(1)) for task in task_names
        })

    def forward(self, task_losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Combine individual task losses using learned uncertainty weights.

        Args:
            task_losses: Dictionary of {task_name: scalar_loss_tensor}.

        Returns:
            Tuple of (total_weighted_loss, dictionary_of_weights).
        """
        total_loss = 0.0
        weights_info = {}

        for task, loss in task_losses.items():
            if task in self.log_vars:
                s = self.log_vars[task]
                # Precision weighting: 0.5 * exp(-s) * loss + 0.5 * s
                precision = torch.exp(-s)
                weighted_loss = 0.5 * precision * loss + 0.5 * s
                total_loss = total_loss + weighted_loss
                weights_info[task] = {
                    "raw_loss": float(loss.item()),
                    "sigma": float(torch.exp(0.5 * s).item()),
                    "precision_weight": float(precision.item()),
                }
            else:
                total_loss = total_loss + loss

        return total_loss, weights_info
