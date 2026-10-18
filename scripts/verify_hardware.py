#!/usr/bin/env python3
"""Hardware detection and acceleration verification script for KidneyCare-AI."""

import sys
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kidneycare.hardware.hardware_detector import HardwareDetector
from kidneycare.hardware.memory_monitor import MemoryMonitor


def main():
    detector = HardwareDetector()
    report = detector.generate_startup_report()
    print(report)

    mem = MemoryMonitor()
    summary = mem.get_summary()
    print(f"Memory Status: {summary['used_ram_percent']}% used, {summary['available_ram_gb']} GB available. Safe headroom: {summary['is_safe']}")


if __name__ == "__main__":
    main()
