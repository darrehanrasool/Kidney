#!/usr/bin/env python3
"""Automated Downloader and Benchmark Script for Kidney Disease Datasets.

Maintains direct academic links, checksums, and synthetic generation for:
- UCI Chronic Kidney Disease Dataset (Apollo Hospitals, India)
- Synthetic Clinical CKD Benchmark Cohort (Zero-Network Mode)
- Optional Kidney Imaging & Kaggle CKD Datasets
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kidneycare.data.dataset_registry import DatasetRegistry
from kidneycare.data.dataset_downloader import DatasetDownloader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and verify canonical Kidney Disease research datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="UCI-CKD",
        choices=["UCI-CKD", "Synthetic-CKD-Benchmark", "all"],
        help="Target benchmark dataset to download or generate",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="download",
        choices=["download", "synthetic", "list"],
        help="'download': Ingest or download UCI CKD dataset; 'synthetic': Generate zero-network synthetic benchmark; 'list': List registry.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets in registry",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=400,
        help="Number of patient records for synthetic benchmark mode",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list:
        args.mode = "list"
    registry = DatasetRegistry()
    downloader = DatasetDownloader()

    if args.mode == "list":
        print("=" * 80)
        print("          CANONICAL CHRONIC KIDNEY DISEASE (CKD) DATASETS CATALOG          ")
        print("=" * 80)
        for ds in registry.list_datasets():
            print(f"Dataset ID   : {ds['id']}")
            print(f"Name         : {ds['name']}")
            print(f"Modality     : {ds['modality']}")
            print(f"Source       : {ds['source']}")
            if "records" in ds:
                print(f"Records      : {ds['records']} patients ({ds.get('features', 24)} attributes)")
            if "source_url" in ds:
                print(f"Source URL   : {ds['source_url']}")
            print(f"License      : {ds.get('license', 'Open')}")
            print("-" * 80)
        return

    if args.mode == "synthetic":
        path = downloader.generate_synthetic_benchmark(n_samples=args.n_samples)
        print(f"Successfully generated synthetic clinical CKD benchmark at: {path}")

    elif args.mode == "download":
        if args.dataset in ["UCI-CKD", "all"]:
            path = downloader.download_uci_ckd()
            print(f"Successfully acquired primary UCI CKD dataset at: {path}")


if __name__ == "__main__":
    main()
