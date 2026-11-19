#!/usr/bin/env python3
"""Master training script across all classical and deep neural architectures."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kidney_ai.training.train import train_all_models


def main():
    print("=" * 70)
    print("             KIDNEY-AI MULTI-PARADIGM MODEL TRAINING            ")
    print("=" * 70)
    models = train_all_models()
    print("Training finished successfully for:")
    for m in models.keys():
        print(f"  • {m}")
    print("Artifacts saved to outputs/trained_models/ and checkpoints/")
    print("=" * 70)


if __name__ == "__main__":
    main()
