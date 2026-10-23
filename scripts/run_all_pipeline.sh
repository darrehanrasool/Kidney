#!/bin/bash
# KidneyCare-AI: Complete End-to-End Execution Pipeline
set -e

echo "================================================================================"
echo "          KIDNEYCARE-AI: AUTOMATED END-TO-END EXECUTION PIPELINE               "
echo "================================================================================"

# Activate virtualenv if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo -e "\n[1/5] Running Clinical Data Preparation & Splitting..."
python3 scripts/prepare_data.py
python3 scripts/create_splits.py

echo -e "\n[2/5] Training Classical & Tabular Neural Models..."
python3 scripts/train_all.py

echo -e "\n[3/5] Training Deep Imaging (Ultrasound & CT) with Anisotropic Diffusion..."
python3 scripts/train_imaging.py --epochs 15 --device mps

echo -e "\n[4/5] Training NephroFusionNet (Multimodal Cross-Attention & Multi-Task Loss)..."
python3 scripts/train_multimodal.py --epochs 20 --device mps

echo -e "\n[5/5] Evaluating Models & Generating All 10 Diagnostic Figures..."
python3 scripts/evaluate_all.py
python3 scripts/generate_visualizations.py

echo -e "\n================================================================================"
echo "  All pipeline stages completed successfully!"
echo "  To launch the interactive dashboard:"
echo "    python3 scripts/run_app.py --port 8080"
echo "================================================================================"
