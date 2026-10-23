#!/usr/bin/env python3
"""Training script for Renal Imaging models (Ultrasound and CT).

Benchmarks against:
- Hannan & Pal (2023): Ultrasound CNN (Speckle filtering, 3-4 layer CNN).
- Ma et al. (2020): Wavelet and feature extraction for stone vs cyst detection.
"""

import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kidneycare.core.reproducibility import set_seed
from kidneycare.hardware.device_manager import DeviceManager
from kidneycare.models.imaging.vision_backbone import RenalCNN, ResNetRenalBackbone
from kidneycare.data.imaging_dataset import create_imaging_dataloaders
from kidneycare.evaluation.metrics import ClinicalMetricsCalculator
from kidneycare.visualization.training_curves import plot_training_history
from kidneycare.monitoring.logging_config import setup_logging, get_logger
from kidneycare.utils.file_utils import ensure_dir

logger = get_logger("kidneycare.train_imaging")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Renal Imaging Neural Networks (Ultrasound & CT)")
    parser.add_argument("--dataset", type=str, default="ultrasound", choices=["ultrasound", "ct"], help="Imaging modality")
    parser.add_argument("--model", type=str, default="renal_cnn", choices=["renal_cnn", "resnet"], help="Architecture")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_imaging_pipeline():
    args = parse_args()
    set_seed(args.seed)
    setup_logging(level="INFO", log_file="logs/training/renal_imaging_training.log")
    logger.info(f"Starting Renal Imaging Pipeline for {args.dataset.upper()} using {args.model.upper()}...")

    # Select execution hardware backend
    dev_mgr = DeviceManager()
    device = dev_mgr.get_device(preferred_device=args.device)
    print("=" * 70)
    print(f"  Training Renal Imaging Architecture: {args.model.upper()} ({args.dataset.upper()})")
    print(f"  Execution Backend : {device}")
    print(f"  Epochs            : {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print("=" * 70)

    data_dir = f"data/raw/imaging/{args.dataset}"
    loaders = create_imaging_dataloaders(data_dir, batch_size=args.batch_size)

    if len(loaders["train"]) == 0:
        print(f"Dataset directory '{data_dir}' is empty. Run 'python3 scripts/download_all_datasets.py' first.")
        sys.exit(1)

    # Initialize model
    if args.model == "renal_cnn":
        model = RenalCNN(in_channels=1, num_classes=4, base_filters=32)
    else:
        model = ResNetRenalBackbone(in_channels=1, num_classes=4, pretrained=False)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    calc = ClinicalMetricsCalculator()
    best_val_acc = 0.0
    epoch_records = []
    checkpoint_dir = Path("checkpoints/imaging")
    ensure_dir(checkpoint_dir)

    print(f"\n{'Epoch':<6} | {'Train Loss':<10} | {'Val Loss':<10} | {'Val Acc':<8} | {'Val Sens':<8} | {'Val Spec':<8} | {'Val F1':<8} | {'Status'}")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        # 1. Training Phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        for images, labels in loaders["train"]:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        scheduler.step()
        train_loss_avg = train_loss / max(1, train_batches)

        # 2. Validation Phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in loaders["val"]:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                val_loss += loss.item()
                val_batches += 1

                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())

        val_loss_avg = val_loss / max(1, val_batches)
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        # Binarize for screening metric comparison (0=normal vs 1,2,3=pathology)
        y_true_bin = (y_true > 0).astype(int)
        y_pred_bin = (y_pred > 0).astype(int)
        y_prob_bin = 1.0 - y_prob[:, 0]

        metrics = calc.compute_binary_metrics(y_true_bin, y_pred_bin, y_prob_bin)
        val_acc = float((y_pred == y_true).mean())

        status = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            status = "★ Best Checkpoint"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_acc,
                "metrics": metrics,
            }, checkpoint_dir / f"best_{args.model}_{args.dataset}.pt")

        epoch_records.append({
            "epoch": epoch,
            "train_loss": train_loss_avg,
            "val_loss": val_loss_avg,
            "train_metrics": {"f1": float(val_acc)},
            "val_metrics": {"f1": float(metrics.f1_macro)},
        })

        msg = (
            f"{epoch:<6} | {train_loss_avg:<10.4f} | {val_loss_avg:<10.4f} | {val_acc:<8.4f} | "
            f"{metrics.recall:<8.4f} | {metrics.specificity:<8.4f} | {metrics.f1_macro:<8.4f} | {status}"
        )
        print(msg)
        logger.info(msg)

    # Save training curves to visualizations/training/
    if epoch_records:
        vt_dir = Path("visualizations/training")
        ensure_dir(vt_dir)
        curve_path = vt_dir / f"{args.model}_{args.dataset}_curves.png"
        plot_training_history({"history": epoch_records}, output_path=curve_path, model_name=f"{args.model.upper()} ({args.dataset.title()})")
        logger.info(f"Saved training history curves to {curve_path}")

    # 3. Test Evaluation on Held-Out Split
    print("\n" + "=" * 70)
    print(f"  Evaluating Best Checkpoint on Held-out {args.dataset.upper()} Test Partition")
    print("=" * 70)
    checkpoint = torch.load(checkpoint_dir / f"best_{args.model}_{args.dataset}.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_preds, test_labels, test_probs = [], [], []
    with torch.no_grad():
        for images, labels in loaders["test"]:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=-1)
            test_preds.extend(torch.argmax(probs, dim=-1).cpu().numpy().tolist())
            test_labels.extend(labels.numpy().tolist())
            test_probs.extend(probs.cpu().numpy().tolist())

    y_t = np.array(test_labels)
    y_p = np.array(test_preds)
    y_pr = np.array(test_probs)

    test_acc = float((y_p == y_t).mean())
    test_metrics = calc.compute_binary_metrics((y_t > 0).astype(int), (y_p > 0).astype(int), 1.0 - y_pr[:, 0])

    print(f"  Test Overall Accuracy  : {test_acc * 100:.2f}%")
    print(f"  Test Sensitivity/Recall: {test_metrics.recall * 100:.2f}%")
    print(f"  Test Specificity       : {test_metrics.specificity * 100:.2f}%")
    print(f"  Test Precision         : {test_metrics.precision * 100:.2f}%")
    print(f"  Test F1-Score (Macro)  : {test_metrics.f1_macro:.4f}")
    print(f"  Test ROC-AUC           : {test_metrics.roc_auc:.4f}")
    print(f"  Test Brier Score       : {test_metrics.brier_score:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    train_imaging_pipeline()
