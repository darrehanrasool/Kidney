"""Confusion matrix plot generator."""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path, title: str = "Confusion Matrix") -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5), dpi=300)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-CKD", "CKD"], yticklabels=["Non-CKD", "CKD"])
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Diagnosis")
    plt.ylabel("True Diagnosis")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
