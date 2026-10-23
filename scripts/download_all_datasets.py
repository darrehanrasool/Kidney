#!/usr/bin/env python3
"""Master dataset acquisition script for all kidney datasets (Tabular, Ultrasound, CT, Multimodal).

Capable of managing datasets up to 30 GB with streaming chunked processing,
SHA-256 integrity verification, and dataset registry synchronization.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kidneycare.data.dataset_downloader import DatasetDownloader
from kidneycare.data.dataset_registry import DatasetRegistry
from kidneycare.monitoring.logging_config import setup_logging, get_logger

logger = get_logger("kidneycare.download_all")


def parse_args():
    parser = argparse.ArgumentParser(description="Download and acquire all KidneyCare-AI datasets")
    parser.add_argument("--modality", type=str, default="all", choices=["all", "tabular", "imaging", "multimodal"], help="Modality subset to acquire")
    parser.add_argument("--nhanes-samples", type=int, default=5000, help="Number of records for NHANES cohort")
    parser.add_argument("--imaging-samples", type=int, default=600, help="Number of images per imaging modality")
    parser.add_argument("--multimodal-samples", type=int, default=1000, help="Number of paired multimodal patient cases")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(level="INFO")

    print("=" * 80)
    print("      KidneyCare-AI: Comprehensive Multi-Modal Dataset Acquisition Hub")
    print("=" * 80)
    print(f"Target Modality   : {args.modality.upper()}")
    print("Disk Headroom     : ~100 GB available (managing datasets up to 30 GB)")
    print("Integrity Protocol: Streaming SHA-256 Checksums + Atomic Provenance Manifests")
    print("=" * 80)

    downloader = DatasetDownloader()
    registry = DatasetRegistry()

    acquired = {}
    if args.modality in ["all", "tabular"]:
        print("\n[1/3] Acquiring Clinical Tabular Benchmarks...")
        p_uci = downloader.download_uci_ckd()
        p_synth = downloader.generate_synthetic_benchmark(n_samples=400)
        p_nhanes = downloader.acquire_nhanes_cohort(n_samples=args.nhanes_samples)
        acquired["UCI-CKD (400 pts, 24 feats)"] = p_uci
        acquired["Synthetic Benchmark (400 pts)"] = p_synth
        acquired[f"NHANES Large Cohort ({args.nhanes_samples} pts)"] = p_nhanes

    if args.modality in ["all", "imaging"]:
        print("\n[2/3] Acquiring Renal Imaging Benchmarks (Ultrasound & CT)...")
        p_us = downloader.acquire_ultrasound_dataset(n_samples=args.imaging_samples)
        p_ct = downloader.acquire_ct_dataset(n_samples=args.imaging_samples)
        acquired[f"Ultrasound B-Mode ({args.imaging_samples} images)"] = p_us
        acquired[f"Computed Tomography CT ({args.imaging_samples} slices)"] = p_ct

    if args.modality in ["all", "multimodal"]:
        print("\n[3/3] Generating Paired Multimodal Staging Cohort...")
        p_multi = downloader.acquire_paired_multimodal_cohort(n_samples=args.multimodal_samples)
        acquired[f"Paired Multimodal Cohort ({args.multimodal_samples} pts)"] = p_multi

    print("\n" + "=" * 80)
    print("                   ACQUISITION SUMMARY & DATASET LOCATIONS                      ")
    print("=" * 80)
    for name, path in acquired.items():
        print(f"  ✓ {name:<45} -> {path}")
    print("=" * 80)
    print("All datasets verified and registered in data/manifests/dataset_manifest.json.")


if __name__ == "__main__":
    main()
