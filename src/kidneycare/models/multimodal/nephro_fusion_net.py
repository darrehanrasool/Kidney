"""NephroFusionNet: Unified Multimodal, Multi-Task Clinical Kidney Architecture.

Integrates:
1. FT-Transformer with learned missingness tokens for laboratory tabular panels.
2. Deep Convolutional Vision Encoder with speckle handling for Ultrasound & CT.
3. Perceiver Cross-Attention Fusion with stochastic Modality Dropout.
4. Multi-Task Clinical Heads:
   - Head A: Binary CKD Screening (BCE / Focal Loss with Temperature Scaling).
   - Head B: KDIGO eGFR Staging G1-G5 (CORAL Ordinal Rank Logits).
   - Head C: Albuminuria Staging A1-A3 (CORAL Ordinal Rank Logits).
   - Head D: Lesion Subtype Classification (Normal, Cyst, Stone, Tumour).
5. Homoscedastic Multi-Task Uncertainty Loss Weighting (Kendall & Gal, 2018).
6. Conformal Prediction Set calibration with clinical abstention guardrails.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List

from kidneycare.models.multimodal.ft_transformer import FTTransformer
from kidneycare.models.multimodal.perceiver_fusion import PerceiverFusionEngine
from kidneycare.models.multimodal.coral_ordinal import CoralOrdinalHead, CoralLoss
from kidneycare.models.multimodal.uncertainty_loss import MultiTaskUncertaintyLoss
from kidneycare.models.imaging.vision_backbone import RenalCNN
from kidneycare.models.multimodal.conformal_prediction import ClinicalConformalPredictor


class NephroFusionNet(nn.Module):
    """Next-generation multi-modal clinical architecture for kidney disease detection & staging."""

    def __init__(
        self,
        num_tabular_features: int = 14,
        d_latent: int = 128,
        image_channels: int = 1,
        num_lesion_classes: int = 4,
        modality_dropout: float = 0.3,
    ):
        super().__init__()
        self.d_latent = d_latent

        # 1. Tabular Modality Encoder (FT-Transformer)
        self.tabular_encoder = FTTransformer(
            num_features=num_tabular_features,
            d_token=d_latent,
            n_layers=3,
            n_heads=4,
            ffn_dim=d_latent * 2,
        )
        self.tab_proj = nn.Linear(d_latent, d_latent)

        # 2. Imaging Modality Encoder (RenalCNN)
        self.vision_encoder = RenalCNN(
            in_channels=image_channels,
            num_classes=num_lesion_classes,
            base_filters=32,
        )
        self.vision_proj = nn.Linear(256, d_latent)

        # 3. Perceiver Cross-Attention Fusion Engine
        self.fusion_engine = PerceiverFusionEngine(
            d_latent=d_latent,
            num_latents=16,
            modality_dropout=modality_dropout,
            n_blocks=2,
        )

        # 4. Multi-Task Decision Heads
        # Head A: Binary CKD Screening (0 = notckd, 1 = ckd)
        self.head_binary_ckd = nn.Sequential(
            nn.Linear(d_latent, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

        # Head B: KDIGO eGFR Staging (G1, G2, G3a, G3b, G4, G5 -> 6 stages)
        self.head_kdigo_stage = CoralOrdinalHead(in_features=d_latent, num_classes=6)

        # Head C: Albuminuria Staging (A1, A2, A3 -> 3 stages)
        self.head_albuminuria = CoralOrdinalHead(in_features=d_latent, num_classes=3)

        # Head D: Lesion Classification (Normal, Cyst, Stone, Tumour -> 4 classes)
        self.head_lesion = nn.Sequential(
            nn.Linear(d_latent, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_lesion_classes),
        )

        # Temperature parameter for probability calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

        # Uncertainty Loss Criterion
        self.multi_task_loss = MultiTaskUncertaintyLoss(
            task_names=["binary_ckd", "kdigo_stage", "albuminuria_stage", "lesion_type"]
        )
        self.coral_loss = CoralLoss()

        # Conformal prediction module
        self.conformal_predictor = ClinicalConformalPredictor(alpha=0.05)

    def extract_features(
        self,
        tabular_x: Optional[torch.Tensor] = None,
        tabular_mask: Optional[torch.Tensor] = None,
        image_x: Optional[torch.Tensor] = None,
        force_modality: Optional[str] = None,
    ) -> torch.Tensor:
        """Extract and fuse multimodal representations."""
        tab_tokens = None
        if tabular_x is not None:
            tab_feat = self.tabular_encoder(tabular_x, tabular_mask) # (B, d)
            tab_tokens = self.tab_proj(tab_feat).unsqueeze(1)        # (B, 1, d)

        img_tokens = None
        if image_x is not None:
            img_feat = self.vision_encoder.extract_embedding(image_x) # (B, 256)
            img_tokens = self.vision_proj(img_feat).unsqueeze(1)      # (B, 1, d)

        fused_latent = self.fusion_engine(
            tabular_tokens=tab_tokens,
            image_tokens=img_tokens,
            force_modality=force_modality,
        )
        return fused_latent

    def forward(
        self,
        tabular_x: Optional[torch.Tensor] = None,
        tabular_mask: Optional[torch.Tensor] = None,
        image_x: Optional[torch.Tensor] = None,
        force_modality: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute predictions across all multi-task clinical heads.

        Returns dictionary containing:
            - binary_ckd_logits: (B, 2)
            - kdigo_stage_logits: (B, 5) rank thresholds
            - albuminuria_logits: (B, 2) rank thresholds
            - lesion_logits: (B, 4)
            - fused_latent: (B, d_latent)
        """
        fused = self.extract_features(tabular_x, tabular_mask, image_x, force_modality)

        logits_binary = self.head_binary_ckd(fused) / torch.clamp(self.temperature, min=0.1)
        logits_kdigo = self.head_kdigo_stage(fused)
        logits_alb = self.head_albuminuria(fused)
        logits_lesion = self.head_lesion(fused)

        return {
            "binary_ckd_logits": logits_binary,
            "kdigo_stage_logits": logits_kdigo,
            "albuminuria_logits": logits_alb,
            "lesion_logits": logits_lesion,
            "fused_latent": fused,
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute multi-task loss with uncertainty weights."""
        task_losses = {}

        if "ckd_binary" in targets:
            task_losses["binary_ckd"] = F.cross_entropy(outputs["binary_ckd_logits"], targets["ckd_binary"])

        if "kdigo_stage" in targets:
            task_losses["kdigo_stage"] = self.coral_loss(outputs["kdigo_stage_logits"], targets["kdigo_stage"])

        if "albuminuria_stage" in targets:
            task_losses["albuminuria_stage"] = self.coral_loss(outputs["albuminuria_logits"], targets["albuminuria_stage"])

        if "lesion_type" in targets:
            task_losses["lesion_type"] = F.cross_entropy(outputs["lesion_logits"], targets["lesion_type"])

        total_loss, info = self.multi_task_loss(task_losses)
        return total_loss, info

    def predict_full_assessment(
        self,
        tabular_x: Optional[torch.Tensor] = None,
        image_x: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Comprehensive clinical inference with calibration and conformal coverage."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(tabular_x=tabular_x, image_x=image_x)

            # Probabilities
            p_ckd = F.softmax(outputs["binary_ckd_logits"], dim=-1)[0].cpu().numpy()
            p_kdigo = self.head_kdigo_stage.predict_probabilities(outputs["kdigo_stage_logits"])[0].cpu().numpy()
            p_alb = self.head_albuminuria.predict_probabilities(outputs["albuminuria_logits"])[0].cpu().numpy()
            p_lesion = F.softmax(outputs["lesion_logits"], dim=-1)[0].cpu().numpy()

            kdigo_stages = ["G1 (Normal)", "G2 (Mild)", "G3a (Mild-Mod)", "G3b (Mod-Severe)", "G4 (Severe)", "G5 (Failure)"]
            alb_stages = ["A1 (Normal)", "A2 (Microalbuminuria)", "A3 (Macroalbuminuria)"]
            lesion_classes = ["Normal", "Renal Cyst", "Kidney Stone", "Renal Tumour"]

            # Conformal set for KDIGO stage
            conformal_set = None
            if self.conformal_predictor.q_hat is not None:
                conformal_set = self.conformal_predictor.predict_set(p_kdigo, kdigo_stages)

            assessment = {
                "ckd_screening": {
                    "ckd_probability": round(float(p_ckd[1]), 4),
                    "not_ckd_probability": round(float(p_ckd[0]), 4),
                    "binary_prediction": "CKD Detected" if p_ckd[1] >= 0.5 else "Non-CKD",
                },
                "kdigo_egfr_staging": {
                    "most_likely_stage": kdigo_stages[int(p_kdigo.argmax())],
                    "stage_probabilities": {kdigo_stages[i]: round(float(p_kdigo[i]), 4) for i in range(6)},
                    "conformal_prediction_set": conformal_set,
                },
                "albuminuria_staging": {
                    "most_likely_stage": alb_stages[int(p_alb.argmax())],
                    "stage_probabilities": {alb_stages[i]: round(float(p_alb[i]), 4) for i in range(3)},
                },
                "imaging_lesion_classification": {
                    "most_likely_lesion": lesion_classes[int(p_lesion.argmax())],
                    "lesion_probabilities": {lesion_classes[i]: round(float(p_lesion[i]), 4) for i in range(4)},
                },
            }
            return assessment
