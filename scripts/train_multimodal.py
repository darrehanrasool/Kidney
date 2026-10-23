#!/usr/bin/env python3
"""Training script for NephroFusionNet (Unified Multimodal Clinical Architecture).

Unifies:
- Tabular Laboratory Biomarkers (eGFR, BUN, Creatinine, Electrolytes, Vitals).
- Renal Imaging (Ultrasound & CT Scans).
- Perceiver Cross-Attention Fusion with Modality Dropout.
- Multi-Task Clinical Decision Heads (Binary Screening, KDIGO G1-G5, Albuminuria, Lesions).
"""

import sys
import argparse
from pathlib import Path
import torch
import torch.optim as optim
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kidneycare.core.reproducibility import set_seed
from kidneycare.hardware.device_manager import DeviceManager
from kidneycare.models.multimodal.nephro_fusion_net import NephroFusionNet
from kidneycare.data.multimodal_dataset import create_multimodal_dataloaders
from kidneycare.evaluation.metrics import ClinicalMetricsCalculator
from kidneycare.visualization.training_curves import plot_training_history
from kidneycare.monitoring.logging_config import setup_logging, get_logger
from kidneycare.utils.file_utils import ensure_dir

logger = get_logger("kidneycare.train_multimodal")


def parse_args():
    parser = argparse.ArgumentParser(description="Train NephroFusionNet Multimodal Architecture")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.3, help="Modality dropout probability")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_multimodal_pipeline():
    args = parse_args()
    set_seed(args.seed)
    setup_logging(level="INFO", log_file="logs/training/multimodal_training.log")
    logger.info("Starting NephroFusionNet Multimodal Pipeline...")

    dev_mgr = DeviceManager()
    device = dev_mgr.get_device(preferred_device=args.device)

    print("=" * 80)
    print("      NephroFusionNet: Multimodal Multi-Task Clinical Training Pipeline")
    print("=" * 80)
    print(f"  Execution Backend : {device}")
    print(f"  Epochs            : {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print("  Tasks             : [1] CKD Binary, [2] KDIGO G1-G5, [3] Albuminuria A1-A3, [4] Lesions")
    print("=" * 80)

    csv_path = "data/processed/paired_multimodal_cohort.csv"
    if not Path(csv_path).exists():
        print(f"Paired multimodal dataset missing at {csv_path}. Running downloader...")
        from kidneycare.data.dataset_downloader import DatasetDownloader
        downloader = DatasetDownloader()
        downloader.acquire_paired_multimodal_cohort(n_samples=1000)

    loaders = create_multimodal_dataloaders(csv_path=csv_path, batch_size=args.batch_size)

    # Initialize NephroFusionNet
    model = NephroFusionNet(
        num_tabular_features=13,
        d_latent=128,
        image_channels=1,
        num_lesion_classes=4,
        modality_dropout=args.dropout,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    calc = ClinicalMetricsCalculator()

    best_val_loss = float("inf")
    epoch_records = []
    checkpoint_dir = Path("checkpoints/multimodal")
    ensure_dir(checkpoint_dir)

    print(f"\n{'Epoch':<6} | {'Total Loss':<10} | {'CKD Acc':<8} | {'KDIGO Acc':<10} | {'Lesion Acc':<10} | {'CKD Sens':<8} | {'Status'}")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        # 1. Training Phase with Modality Dropout
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in loaders["train"]:
            tab_x = batch["tabular_x"].to(device)
            img_x = batch["image_x"].to(device)
            targets = {k: v.to(device) for k, v in batch["targets"].items()}

            optimizer.zero_grad()
            outputs = model(tabular_x=tab_x, image_x=img_x)
            loss, weights_info = model.compute_loss(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss_avg = epoch_loss / max(1, n_batches)

        # 2. Validation Phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        ckd_preds, ckd_true = [], []
        kdigo_preds, kdigo_true = [], []
        lesion_preds, lesion_true = [], []
        kdigo_probs_all = []

        with torch.no_grad():
            for batch in loaders["val"]:
                tab_x = batch["tabular_x"].to(device)
                img_x = batch["image_x"].to(device)
                targets = {k: v.to(device) for k, v in batch["targets"].items()}

                outputs = model(tabular_x=tab_x, image_x=img_x)
                loss, _ = model.compute_loss(outputs, targets)
                val_loss += loss.item()
                val_batches += 1

                # Head A: CKD Binary
                p_ckd = torch.softmax(outputs["binary_ckd_logits"], dim=-1)[:, 1]
                ckd_preds.extend((p_ckd > 0.5).int().cpu().numpy().tolist())
                ckd_true.extend(targets["ckd_binary"].cpu().numpy().tolist())

                # Head B: KDIGO Stage
                k_preds = model.head_kdigo_stage.predict_labels(outputs["kdigo_stage_logits"])
                kdigo_preds.extend(k_preds.cpu().numpy().tolist())
                kdigo_true.extend(targets["kdigo_stage"].cpu().numpy().tolist())

                k_probs = model.head_kdigo_stage.predict_probabilities(outputs["kdigo_stage_logits"])
                kdigo_probs_all.extend(k_probs.cpu().numpy().tolist())

                # Head D: Lesion Type
                l_preds = torch.argmax(outputs["lesion_logits"], dim=-1)
                lesion_preds.extend(l_preds.cpu().numpy().tolist())
                lesion_true.extend(targets["lesion_type"].cpu().numpy().tolist())

        val_loss_avg = val_loss / max(1, val_batches)
        acc_ckd = float((np.array(ckd_preds) == np.array(ckd_true)).mean())
        acc_kdigo = float((np.array(kdigo_preds) == np.array(kdigo_true)).mean())
        acc_lesion = float((np.array(lesion_preds) == np.array(lesion_true)).mean())

        ckd_sens = float(calc.compute_binary_metrics(np.array(ckd_true), np.array(ckd_preds), np.array(ckd_preds)).recall)

        status = ""
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            status = "★ Best Checkpoint"
            # Calibrate conformal predictor on validation predictions
            model.conformal_predictor.calibrate(np.array(kdigo_probs_all), np.array(kdigo_true))
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss_avg,
                "conformal_q_hat": model.conformal_predictor.q_hat,
            }, checkpoint_dir / "best_nephro_fusion_net.pt")

        epoch_records.append({
            "epoch": epoch,
            "train_loss": train_loss_avg,
            "val_loss": val_loss_avg,
            "train_metrics": {"f1": float(acc_ckd)},
            "val_metrics": {"f1": float(ckd_sens)},
        })

        msg = (
            f"{epoch:<6} | {train_loss_avg:<10.4f} | {acc_ckd:<8.4f} | {acc_kdigo:<10.4f} | "
            f"{acc_lesion:<10.4f} | {ckd_sens:<8.4f} | {status}"
        )
        print(msg)
        logger.info(msg)

    # Save training curves to visualizations/training/
    if epoch_records:
        vt_dir = Path("visualizations/training")
        ensure_dir(vt_dir)
        curve_path = vt_dir / "nephro_fusion_net_curves.png"
        plot_training_history({"history": epoch_records}, output_path=curve_path, model_name="NephroFusionNet")
        logger.info(f"Saved training history curves to {curve_path}")

    # 3. Comprehensive Held-Out Test Evaluation
    print("\n" + "=" * 80)
    print("  Evaluating Best NephroFusionNet on Held-Out Multimodal Test Cohort")
    print("=" * 80)
    ckpt = torch.load(checkpoint_dir / "best_nephro_fusion_net.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.conformal_predictor.q_hat = ckpt["conformal_q_hat"]
    model.eval()

    t_ckd_pred, t_ckd_true, t_ckd_prob = [], [], []
    t_kdigo_pred, t_kdigo_true = [], []
    t_lesion_pred, t_lesion_true = [], []

    with torch.no_grad():
        for batch in loaders["test"]:
            tab_x = batch["tabular_x"].to(device)
            img_x = batch["image_x"].to(device)
            targets = {k: v.to(device) for k, v in batch["targets"].items()}

            outputs = model(tabular_x=tab_x, image_x=img_x)

            p_ckd = torch.softmax(outputs["binary_ckd_logits"], dim=-1)[:, 1]
            t_ckd_pred.extend((p_ckd > 0.5).int().cpu().numpy().tolist())
            t_ckd_true.extend(targets["ckd_binary"].cpu().numpy().tolist())
            t_ckd_prob.extend(p_ckd.cpu().numpy().tolist())

            k_preds = model.head_kdigo_stage.predict_labels(outputs["kdigo_stage_logits"])
            t_kdigo_pred.extend(k_preds.cpu().numpy().tolist())
            t_kdigo_true.extend(targets["kdigo_stage"].cpu().numpy().tolist())

            l_preds = torch.argmax(outputs["lesion_logits"], dim=-1)
            t_lesion_pred.extend(l_preds.cpu().numpy().tolist())
            t_lesion_true.extend(targets["lesion_type"].cpu().numpy().tolist())

    y_true_ckd = np.array(t_ckd_true)
    y_pred_ckd = np.array(t_ckd_pred)
    y_prob_ckd = np.array(t_ckd_prob)

    ckd_test_metrics = calc.compute_binary_metrics(y_true_ckd, y_pred_ckd, y_prob_ckd)
    kdigo_test_acc = float((np.array(t_kdigo_pred) == np.array(t_kdigo_true)).mean())
    lesion_test_acc = float((np.array(t_lesion_pred) == np.array(t_lesion_true)).mean())

    print(f"  [Task 1] CKD Binary Screening Accuracy : {ckd_test_metrics.accuracy * 100:.2f}%")
    print(f"           Sensitivity / Recall          : {ckd_test_metrics.recall * 100:.2f}%")
    print(f"           Specificity                   : {ckd_test_metrics.specificity * 100:.2f}%")
    print(f"           F1-Score (Macro)              : {ckd_test_metrics.f1_macro:.4f}")
    print(f"           ROC-AUC                       : {ckd_test_metrics.roc_auc:.4f}")
    print(f"           Brier Score                   : {ckd_test_metrics.brier_score:.4f}")
    print(f"  [Task 2] KDIGO G1-G5 Staging Accuracy  : {kdigo_test_acc * 100:.2f}%")
    print(f"  [Task 4] Lesion Classification Accuracy: {lesion_test_acc * 100:.2f}%")
    print(f"  Conformal Prediction Quantile Cutoff   : q_hat = {model.conformal_predictor.q_hat:.4f} (95% Coverage)")
    print("=" * 80)


if __name__ == "__main__":
    train_multimodal_pipeline()
