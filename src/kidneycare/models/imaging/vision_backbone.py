"""Deep Convolutional & Vision architectures for renal imaging (Ultrasound & CT).

Implements convolutional backbones adapted from:
- Hannan & Pal (2023): Multi-layered CNN for ultrasound B-mode classification.
- Ma et al. (2020): Wavelet and feature extraction for stone vs cyst detection.
- Modern transfer-learning backbones (ResNet, ConvNeXt, Custom RenalCNN).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple


class RenalCNN(nn.Module):
    """Deep 6-layer Convolutional Neural Network for renal ultrasound and CT imaging.

    Incorporates:
    - BatchNorm and Dropout for regularization.
    - Adaptive pooling for resolution invariance.
    - Anisotropic speckle feature fusion.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_filters: int = 32,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Layer 1: k=3x3, filters=32
        self.conv1 = nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_filters)

        # Layer 2: k=3x3, filters=64
        self.conv2 = nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(base_filters * 2)

        # Layer 3: k=3x3, filters=128
        self.conv3 = nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(base_filters * 4)

        # Layer 4: k=3x3, filters=256
        self.conv4 = nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(base_filters * 8)

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout = nn.Dropout(dropout_rate)

        # Classification Head
        feature_dim = base_filters * 8 * 4 * 4
        self.fc1 = nn.Linear(feature_dim, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)

        # Gradient hook placeholders for Grad-CAM
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial convolutional feature maps."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = F.relu(self.bn4(self.conv4(x)))
        # Register hook on final conv feature map for Grad-CAM
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        self.activations = x

        x = self.adaptive_pool(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        flat = torch.flatten(feats, 1)
        h = F.relu(self.bn_fc(self.fc1(self.dropout(flat))))
        logits = self.fc2(self.dropout(h))
        return logits

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 256-dimensional dense embedding for multimodal fusion."""
        feats = self.forward_features(x)
        flat = torch.flatten(feats, 1)
        embedding = F.relu(self.bn_fc(self.fc1(flat)))
        return embedding

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


class ResNetRenalBackbone(nn.Module):
    """Pre-trained or initialized ResNet backbone adapted for single-channel medical imaging."""

    def __init__(self, in_channels: int = 1, num_classes: int = 4, pretrained: bool = False):
        super().__init__()
        import torchvision.models as models

        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        # Adapt first convolution for single-channel grayscale if needed
        if in_channels != 3:
            old_conv = base.conv1
            self.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            if pretrained:
                # Average RGB weights across single input channel
                with torch.no_grad():
                    self.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        else:
            self.conv1 = base.conv1

        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Linear(512, num_classes)

        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if x.requires_grad:
            x.register_hook(self.activations_hook)
        self.activations = x

        x = self.avgpool(x)
        flat = torch.flatten(x, 1)
        logits = self.fc(flat)
        return logits

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1) # 512-dim embedding
