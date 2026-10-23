"""Gradient-weighted Class Activation Mapping (Grad-CAM) for renal imaging.

Computes class-specific spatial saliency heatmaps:
    alpha_k^c = (1 / Z) * sum_i sum_j (d y^c / d A_{i,j}^k)
    L_{Grad-CAM}^c = ReLU(sum_k alpha_k^c * A^k)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class GradCAM:
    """Computes Grad-CAM attribution heatmaps for renal vision backbones."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Generate normalized 2D Grad-CAM heatmap [0, 1].

        Args:
            input_tensor: 4D tensor (1, C, H, W).
            target_class: Integer class index. If None, uses argmax predicted class.

        Returns:
            2D numpy array heatmap matching input (H, W).
        """
        self.model.zero_grad()
        input_tensor = input_tensor.clone().detach().requires_grad_(True)

        logits = self.model(input_tensor)
        if target_class is None:
            target_class = int(logits.argmax(dim=-1).item())

        score = logits[0, target_class]
        score.backward()

        gradients = self.model.gradients
        activations = self.model.activations

        if gradients is None or activations is None:
            # Fallback uniform heatmap if hooks failed
            return np.ones((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32) * 0.5

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        # Weighted combination of forward activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        # Apply ReLU to retain only positive influence
        cam = F.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam
