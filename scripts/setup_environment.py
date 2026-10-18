#!/usr/bin/env python3
"""Environment setup and directory hierarchy initializer for KidneyCare-AI."""

import os
import sys
from pathlib import Path


def setup_directories():
    base_dir = Path(__file__).resolve().parent.parent

    directories = [
        "configs",
        "data/manifests",
        "data/raw/clinical",
        "data/raw/imaging",
        "data/interim",
        "data/processed",
        "data/splits/train",
        "data/splits/validation",
        "data/splits/test",
        "data/external",
        "docs",
        "experiments/configs",
        "experiments/results",
        "logs/application",
        "logs/training",
        "logs/errors",
        "logs/datasets",
        "logs/hardware",
        "models/classical_ml",
        "models/deep_learning",
        "models/ensemble",
        "models/explainability",
        "notebooks",
        "papers/pdf",
        "papers/metadata",
        "papers/literature_reviews",
        "papers/methodology_extraction",
        "papers/baseline_comparisons",
        "papers/research_gaps",
        "reports/data_quality",
        "reports/evaluation",
        "reports/explainability",
        "reports/research",
        "reports/model_cards",
        "checkpoints/latest",
        "checkpoints/best",
        "checkpoints/periodic",
        "checkpoints/metadata",
        "outputs/predictions",
        "outputs/trained_models",
        "outputs/inference_reports",
        "visualizations/data_exploration",
        "visualizations/data_quality",
        "visualizations/training/per_epoch",
        "visualizations/training/summaries",
        "visualizations/validation",
        "visualizations/testing",
        "visualizations/explainability",
        "visualizations/comparisons",
        "visualizations/statistical_analysis",
    ]

    for d in directories:
        target = base_dir / d
        target.mkdir(parents=True, exist_ok=True)

    print("All KidneyCare-AI directory structures successfully initialized.")


if __name__ == "__main__":
    setup_directories()
