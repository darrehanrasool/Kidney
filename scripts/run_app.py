#!/usr/bin/env python3
"""Run KidneyCare-AI Clinical Interactive Dashboard."""

import argparse
import sys
from pathlib import Path

# Add src to pythonpath
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kidneycare.ui.dashboard_app import run_dashboard_server


def main():
    parser = argparse.ArgumentParser(description="KidneyCare-AI Interactive Dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind dashboard server (default: 8080)")
    args = parser.parse_args()

    run_dashboard_server(port=args.port)


if __name__ == "__main__":
    main()
