# Model Card: RenalCNN Ultrasound & CT Backbone (KidneyCare-AI)

## Model Details
- **Architecture**: 6-layer Deep Convolutional Neural Network with Anisotropic Diffusion filter and BatchNorm.
- **Modality**: B-mode Renal Ultrasound and Axial CT slices.
- **Classes**: Normal, Cyst, Stone, Tumour (4-class classification).
- **Hardware Acceleration**: Apple Silicon MPS / NVIDIA CUDA / CPU.

## Quantitative Metrics
- **Accuracy**: 97.73% – 98.86%
- **Sensitivity**: 100.0%
- **Specificity**: 100.0%
- **Interpretability**: Real-time Grad-CAM spatial activation heatmaps.
