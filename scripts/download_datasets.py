#!/usr/bin/env python3
"""Dataset download & synthetic generator script."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kidney_ai.data.dataset_downloader import DatasetDownloader


def main():
    parser = argparse.ArgumentParser(description="Download or generate kidney datasets")
    parser.add_argument("--mode", type=str, choices=["sample", "full"], default="sample")
    args = parser.parse_args()

    downloader = DatasetDownloader()
    downloader.download_all(mode=args.mode)
    print("Dataset generation complete.")


if __name__ == "__main__":
    main()
