"""Model checkpoint serialization and metadata tracking."""

from pathlib import Path
from typing import Any, Dict, Optional
import torch


class CheckpointManager:
    """Manages saving and loading of model weights and experiment state."""

    def __init__(self, checkpoints_dir: Optional[Path] = None):
        if checkpoints_dir is None:
            checkpoints_dir = Path(__file__).resolve().parent.parent.parent.parent / "checkpoints"
        self.checkpoints_dir = Path(checkpoints_dir)
        (self.checkpoints_dir / "best").mkdir(parents=True, exist_ok=True)
        (self.checkpoints_dir / "latest").mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, model: torch.nn.Module, epoch: int, metrics: Dict[str, float], model_name: str, is_best: bool = False) -> Path:
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "model_name": model_name
        }
        latest_path = self.checkpoints_dir / "latest" / f"{model_name}_latest.pt"
        torch.save(state, latest_path)

        if is_best:
            best_path = self.checkpoints_dir / "best" / f"{model_name}_best.pt"
            torch.save(state, best_path)
            return best_path

        return latest_path

    def load_checkpoint(self, path: Path, model: torch.nn.Module) -> Dict[str, Any]:
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint
