"""Unit tests for hardware detection, device selection, and memory monitoring."""

import pytest
from unittest.mock import patch

from kidneycare.hardware.hardware_detector import HardwareDetector
from kidneycare.hardware.device_manager import DeviceManager
from kidneycare.hardware.memory_monitor import MemoryMonitor


def test_hardware_detection():
    detector = HardwareDetector()
    profile = detector.detect()
    assert profile["os"] != ""
    assert profile["system_ram_gb"] > 0
    assert profile["selected_backend"] in ("cuda", "mps", "cpu")
    assert "KidneyCare-AI Hardware" in detector.generate_startup_report()


def test_device_selection_cpu_fallback():
    dm = DeviceManager(preferred="cpu")
    dev = dm.get_device()
    assert dev.type == "cpu"


def test_memory_monitor():
    mm = MemoryMonitor()
    summary = mm.get_summary()
    assert "used_ram_percent" in summary
    assert "available_ram_gb" in summary
    assert isinstance(summary["is_safe"], bool)
