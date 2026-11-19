#!/usr/bin/env python3
"""Hardware detection script."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kidney_ai.hardware.hardware_detector import detect_hardware


def main():
    profile = detect_hardware()
    print("=" * 70)
    print("             KIDNEY-AI ACCELERATOR & COMPUTE AUDIT               ")
    print("=" * 70)
    print(f"OS: {profile.os_name} {profile.os_version} ({profile.cpu_architecture})")
    print(f"CPUs: {profile.cpu_count_physical} Cores ({profile.cpu_count_logical} Logical)")
    print(f"RAM: {profile.total_ram_gb} GB Total, {profile.available_ram_gb} GB Available")
    print(f"CUDA: {profile.has_cuda} ({profile.cuda_device_name})")
    print(f"Apple Silicon MPS: {profile.has_mps}")
    print(f"Optimal Compute Device: '{profile.recommended_device}'")
    print("=" * 70)


if __name__ == "__main__":
    main()
