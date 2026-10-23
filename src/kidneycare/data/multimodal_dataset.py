"""PyTorch Dataset loader for paired Multimodal Clinical Kidney Cohort."""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image

from kidneycare.core.constants import IMAGING_CLASS_MAPPING


class MultimodalKidneyDataset(Dataset):
    """Loads paired tabular laboratory markers and renal ultrasound/CT scans."""

    def __init__(
        self,
        csv_path: str = "data/processed/paired_multimodal_cohort.csv",
        split: str = "train",
        target_size: Tuple[int, int] = (224, 224),
    ):
        self.split = split
        self.target_size = target_size
        df = pd.read_csv(csv_path)

        if "split" in df.columns:
            self.df = df[df["split"] == split].reset_index(drop=True)
        else:
            self.df = df

        self.tab_cols = [
            "age", "female", "sc", "bu", "bp", "bgr", "al", "htn", "dm",
            "sod", "pot", "hemo", "egfr_ckd_epi_2021"
        ]
        # Compute mean & std for tabular normalization
        self.tab_mean = self.df[self.tab_cols].mean().values
        self.tab_std = (self.df[self.tab_cols].std() + 1e-5).values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        # 1. Tabular features normalized
        tab_raw = row[self.tab_cols].values.astype(float)
        tab_norm = (tab_raw - self.tab_mean) / self.tab_std
        tab_tensor = torch.from_numpy(tab_norm).float()

        # 2. Image loading (Ultrasound or CT)
        img_path_str = row["ultrasound_path"] if "ultrasound_path" in row and pd.notna(row["ultrasound_path"]) else ""
        if not img_path_str or not Path(img_path_str).exists():
            img_path_str = row["ct_path"] if "ct_path" in row and pd.notna(row["ct_path"]) else ""

        if img_path_str and Path(img_path_str).exists():
            with Image.open(img_path_str) as img:
                img_gray = img.convert("L").resize(self.target_size)
                arr = (np.array(img_gray, dtype=np.float32) / 255.0 - 0.456) / 0.224
                img_tensor = torch.from_numpy(arr).unsqueeze(0) # (1, H, W)
        else:
            # Zero tensor placeholder if image is missing
            img_tensor = torch.zeros((1, self.target_size[0], self.target_size[1]), dtype=torch.float32)

        # 3. Targets
        ckd_bin = int(row["ckd_binary"])
        kdigo_g = int(row["kdigo_g_stage"])
        alb_stage = int(row["albuminuria_stage"])
        lesion_name = str(row["lesion_type"]).lower()
        lesion_idx = IMAGING_CLASS_MAPPING.get(lesion_name, 0)

        return {
            "tabular_x": tab_tensor,
            "image_x": img_tensor,
            "targets": {
                "ckd_binary": torch.tensor(ckd_bin, dtype=torch.long),
                "kdigo_stage": torch.tensor(kdigo_g, dtype=torch.long),
                "albuminuria_stage": torch.tensor(alb_stage, dtype=torch.long),
                "lesion_type": torch.tensor(lesion_idx, dtype=torch.long),
            },
            "patient_id": row["patient_id"],
        }


def create_multimodal_dataloaders(
    csv_path: str = "data/processed/paired_multimodal_cohort.csv",
    batch_size: int = 32,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Create train, validation, and test loaders for paired multimodal cohort."""
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = MultimodalKidneyDataset(csv_path=csv_path, split=split)
        shuffle = (split == "train")
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
        )
    return loaders
