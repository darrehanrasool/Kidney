"""PyTorch Dataset and DataLoader loaders for renal ultrasound and CT imaging."""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

from kidneycare.core.constants import IMAGING_CLASS_MAPPING


class RenalImageDataset(Dataset):
    """Loads renal ultrasound or CT images organized by split and class directories."""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (224, 224),
    ):
        self.root_dir = Path(root_dir) / split
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.samples: List[Tuple[Path, int]] = []

        if self.root_dir.exists():
            for cls_name, cls_idx in IMAGING_CLASS_MAPPING.items():
                cls_dir = self.root_dir / cls_name
                if cls_dir.exists():
                    for p in cls_dir.glob("*.png"):
                        self.samples.append((p, cls_idx))
                    for p in cls_dir.glob("*.jpg"):
                        self.samples.append((p, cls_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        with Image.open(img_path) as img:
            img_gray = img.convert("L").resize(self.target_size)
            arr = np.array(img_gray, dtype=np.float32) / 255.0

        # Normalization (ImageNet style single channel)
        arr = (arr - 0.456) / 0.224
        tensor = torch.from_numpy(arr).unsqueeze(0) # (1, H, W)

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, label


def create_imaging_dataloaders(
    dataset_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    target_size: Tuple[int, int] = (224, 224),
) -> Dict[str, DataLoader]:
    """Create train, validation, and test DataLoader dictionary."""
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = RenalImageDataset(root_dir=dataset_dir, split=split, target_size=target_size)
        shuffle = (split == "train")
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
        )
    return loaders
