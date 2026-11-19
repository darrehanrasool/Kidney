from pathlib import Path
import torch
import torch.nn as nn
from kidney_ai.training.checkpoint_manager import CheckpointManager


def test_checkpoint_manager(tmp_path):
    mgr = CheckpointManager(checkpoints_dir=tmp_path)
    model = nn.Linear(5, 2)
    saved = mgr.save_checkpoint(model, 1, {"val_f1": 0.95}, "dummy_model", is_best=True)
    assert saved.exists()

    new_model = nn.Linear(5, 2)
    ckpt = mgr.load_checkpoint(saved, new_model)
    assert ckpt["epoch"] == 1
