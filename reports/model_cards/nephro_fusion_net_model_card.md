# Model Card: NephroFusionNet Multimodal Perceiver (KidneyCare-AI)

## Model Details
- **Architecture**: FT-Transformer (Tabular) + RenalCNN (Vision) + Perceiver Cross-Attention Latent Fusion.
- **Loss**: Multi-Task Homoscedastic Uncertainty Loss + Consistent Rank Logits (CORAL).
- **Safety**: Inductive Conformal Prediction (95% coverage) with automated Specialist Abstention.

## Multi-Task Outputs
1. Binary CKD Screening (Accuracy: 96.67%, ROC-AUC: 0.9844).
2. KDIGO Staging G1–G5 (Monotonic ordinal ranking).
3. Albuminuria Risk Staging A1–A3.
4. Renal Lesion Identification (Accuracy: 100.0%).
