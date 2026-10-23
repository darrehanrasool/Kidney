"""Unit tests for RenalCNN, NephroFusionNet, GradCAM, and Conformal Prediction."""

import pytest
import torch
import numpy as np

from kidneycare.models.imaging.vision_backbone import RenalCNN
from kidneycare.models.imaging.gradcam import GradCAM
from kidneycare.models.multimodal.ft_transformer import FTTransformer
from kidneycare.models.multimodal.coral_ordinal import CoralOrdinalHead, CoralLoss
from kidneycare.models.multimodal.perceiver_fusion import PerceiverFusionEngine
from kidneycare.models.multimodal.conformal_prediction import ClinicalConformalPredictor
from kidneycare.models.multimodal.nephro_fusion_net import NephroFusionNet


def test_renal_cnn_forward():
    model = RenalCNN(in_channels=1, num_classes=4, base_filters=16)
    x = torch.randn(2, 1, 224, 224)
    logits = model(x)
    embedding = model.extract_embedding(x)
    assert logits.shape == (2, 4)
    assert embedding.shape == (2, 256)


def test_gradcam_computation():
    model = RenalCNN(in_channels=1, num_classes=4, base_filters=16)
    model.eval()
    gradcam = GradCAM(model)
    x = torch.randn(1, 1, 224, 224)
    cam = gradcam.generate_heatmap(x, target_class=1)
    assert cam.shape == (224, 224)
    assert 0.0 <= cam.min() <= cam.max() <= 1.0


def test_ft_transformer_tabular_embedding():
    transformer = FTTransformer(num_features=10, d_token=32, n_layers=2, n_heads=4)
    x = torch.randn(4, 10)
    feat = transformer(x)
    assert feat.shape == (4, 32)


def test_coral_ordinal_head_and_loss():
    head = CoralOrdinalHead(in_features=32, num_classes=5)
    loss_fn = CoralLoss()
    feat = torch.randn(4, 32)
    logits = head(feat)
    assert logits.shape == (4, 4)
    probs = head.predict_probabilities(logits)
    assert probs.shape == (4, 5)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)

    targets = torch.tensor([0, 2, 4, 1], dtype=torch.long)
    loss = loss_fn(logits, targets)
    assert loss.item() >= 0.0


def test_perceiver_fusion_engine():
    fusion = PerceiverFusionEngine(d_latent=64, num_latents=4, n_blocks=1)
    tab_tokens = torch.randn(2, 1, 64)
    img_tokens = torch.randn(2, 1, 64)
    latents = fusion(tabular_tokens=tab_tokens, image_tokens=img_tokens)
    assert latents.shape == (2, 64)


def test_nephro_fusion_net_forward():
    model = NephroFusionNet(
        num_tabular_features=12,
        d_latent=64,
        image_channels=1,
        num_lesion_classes=4,
    )
    x_tab = torch.randn(2, 12)
    x_img = torch.randn(2, 1, 224, 224)

    outputs = model(tabular_x=x_tab, image_x=x_img)
    assert "binary_ckd_logits" in outputs
    assert "kdigo_stage_logits" in outputs
    assert "albuminuria_logits" in outputs
    assert "lesion_logits" in outputs
    assert "fused_latent" in outputs
    assert outputs["binary_ckd_logits"].shape == (2, 2)
    assert outputs["kdigo_stage_logits"].shape == (2, 5)
    assert outputs["lesion_logits"].shape == (2, 4)
    assert outputs["fused_latent"].shape == (2, 64)


def test_clinical_conformal_predictor():
    predictor = ClinicalConformalPredictor(alpha=0.10)
    # Calibrate on dummy probabilities
    np.random.seed(42)
    y_true = np.random.choice([0, 1], size=100)
    # Mostly accurate probs
    y_prob = np.zeros((100, 2))
    for i, y in enumerate(y_true):
        y_prob[i, y] = np.random.uniform(0.6, 0.99)
        y_prob[i, 1 - y] = 1.0 - y_prob[i, y]

    predictor.calibrate(y_prob, y_true)
    assert predictor.is_calibrated
    assert predictor.q_hat > 0.0

    test_prob = np.array([0.95, 0.05])
    result = predictor.predict_set(test_prob)
    assert "prediction_set" in result
    assert 0 in result["prediction_indices"]
