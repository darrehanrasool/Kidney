"""Unit tests for checkpoint saving, atomic writes, and safe recovery."""

import tempfile
import torch
import torch.nn as nn
import pytest

from kidneycare.training.checkpoint_manager import CheckpointManager
from kidneycare.training.recovery_manager import RecoveryManager


def test_checkpoint_save_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(base_dir=tmpdir)
        recovery = RecoveryManager(checkpoints_dir=tmpdir)

        model = nn.Linear(8, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        saved_path = mgr.save_neural_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=5,
            metrics={"f1": 0.985, "accuracy": 0.980},
            is_best=True,
            model_name="test_mlp",
        )

        assert saved_path.exists()
        assert recovery.has_checkpoint("test_mlp", use_best=True)

        # Restore into new model
        model2 = nn.Linear(8, 2)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)

        loaded = recovery.load_neural_checkpoint(
            model=model2,
            optimizer=optimizer2,
            model_name="test_mlp",
            use_best=True,
        )

        assert loaded["epoch"] == 5
        assert loaded["metrics"]["f1"] == 0.985
