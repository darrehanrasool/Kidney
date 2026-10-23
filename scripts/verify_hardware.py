#!/usr/bin/env python3
"""Hardware detection and acceleration verification script for KidneyCare-AI."""

import sys
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kidneycare.hardware.hardware_detector import HardwareDetector
from kidneycare.hardware.memory_monitor import MemoryMonitor
from kidneycare.monitoring.logging_config import setup_logging, get_logger

logger = get_logger("kidneycare.hardware")


def main():
    setup_logging(log_file="logs/hardware/hardware.log")
    detector = HardwareDetector()
    report = detector.generate_startup_report()
    print(report)
    logger.info(report)

    mem = MemoryMonitor()
    summary = mem.get_summary()
    mem_msg = f"Memory Status: {summary['used_ram_percent']}% used, {summary['available_ram_gb']} GB available. Safe headroom: {summary['is_safe']}"
    print(mem_msg)
    logger.info(mem_msg)


if __name__ == "__main__":
    main()
