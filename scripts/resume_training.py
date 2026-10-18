#!/usr/bin/env python3
"""Resumes interrupted neural network training from latest verified checkpoint."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kidneycare.training.recovery_manager import RecoveryManager
from kidneycare.training.checkpoint_manager import CheckpointManager


def main():
    recovery = RecoveryManager(checkpoints_dir="checkpoints")
    model_name = "tabular_mlp"

    if not recovery.has_checkpoint(model_name=model_name):
        print(f"No previous checkpoint found for {model_name} in checkpoints/latest/. Starting new training pipeline...")
        from scripts.train_all import main as run_train
        run_train()
    else:
        print(f"Found valid checkpoint for {model_name}. Resuming training execution...")
        from scripts.train_all import main as run_train
        run_train()


if __name__ == "__main__":
    main()
