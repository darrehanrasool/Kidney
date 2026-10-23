"""Renal image preprocessing: Anisotropic diffusion speckle filtering and GLCM texture extraction.

Directly implements mathematical formulations from:
- Hannan & Pal (2023): Anisotropic diffusion speckle reduction (Eq. 1).
- Ma et al. (2020): Level-set, Gabor, and GLCM Haralick texture features (Eqs. 4-19).
"""

import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, Optional


def anisotropic_diffusion(
    img: np.ndarray,
    n_iter: int = 10,
    kappa: float = 30.0,
    gamma: float = 0.15,
    option: int = 1,
) -> np.ndarray:
    """Perona-Malik Anisotropic Diffusion Filter for ultrasound speckle reduction.

    Solves:
        dI/dt = div(c(x,y,t) * grad(I)) = c * Laplacian(I) + grad(c) . grad(I)

    Conductance coefficients:
        Option 1: c(||grad(I)||) = exp(-(||grad(I)|| / kappa)^2)  (favors high contrast edges)
        Option 2: c(||grad(I)||) = 1 / (1 + (||grad(I)|| / kappa)^2) (favors wider regions)

    Args:
        img: 2D numpy array with grayscale pixel intensities [0, 255] or [0, 1].
        n_iter: Number of diffusion iterations (default 10).
        kappa: Gradient threshold / conduction coefficient.
        gamma: Integration constant (must be <= 0.25 for 4-neighbor stability).
        option: 1 for exponential conductance, 2 for reciprocal conductance.

    Returns:
        Filtered 2D image preserving structural renal borders while smoothing speckle.
    """
    im = img.astype(np.float32)
    diff = im.copy()

    for _ in range(n_iter):
        # 4-directional spatial finite differences
        delta_north = np.pad(diff[:-1, :], ((1, 0), (0, 0)), mode="edge") - diff
        delta_south = np.pad(diff[1:, :], ((0, 1), (0, 0)), mode="edge") - diff
        delta_east = np.pad(diff[:, 1:], ((0, 0), (0, 1)), mode="edge") - diff
        delta_west = np.pad(diff[:, :-1], ((0, 0), (1, 0)), mode="edge") - diff

        # Conduction coefficients
        if option == 1:
            c_north = np.exp(-((delta_north / kappa) ** 2))
            c_south = np.exp(-((delta_south / kappa) ** 2))
            c_east = np.exp(-((delta_east / kappa) ** 2))
            c_west = np.exp(-((delta_west / kappa) ** 2))
        else:
            c_north = 1.0 / (1.0 + (delta_north / kappa) ** 2)
            c_south = 1.0 / (1.0 + (delta_south / kappa) ** 2)
            c_east = 1.0 / (1.0 + (delta_east / kappa) ** 2)
            c_west = 1.0 / (1.0 + (delta_west / kappa) ** 2)

        # Discrete flux update
        diff += gamma * (
            c_north * delta_north +
            c_south * delta_south +
            c_east * delta_east +
            c_west * delta_west
        )

    return diff


def compute_glcm(image: np.ndarray, levels: int = 16, offset: Tuple[int, int] = (1, 0)) -> np.ndarray:
    """Compute normalized Gray-Level Co-occurrence Matrix (GLCM).

    Args:
        image: 2D grayscale image array.
        levels: Number of quantized intensity bins (default 16).
        offset: (dy, dx) pixel displacement offset.

    Returns:
        Normalized GLCM of shape (levels, levels).
    """
    img = np.asarray(image, dtype=np.float32)
    # Normalize to [0, levels - 1]
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        quantized = np.clip(((img - img_min) / (img_max - img_min) * (levels - 1)).astype(int), 0, levels - 1)
    else:
        quantized = np.zeros_like(img, dtype=int)

    glcm = np.zeros((levels, levels), dtype=np.float32)
    dy, dx = offset
    h, w = quantized.shape

    # Vectorized / bounded co-occurrence accumulation
    y_start, y_end = max(0, -dy), min(h, h - dy)
    x_start, x_end = max(0, -dx), min(w, w - dx)

    src = quantized[y_start:y_end, x_start:x_end]
    tgt = quantized[y_start + dy : y_end + dy, x_start + dx : x_end + dx]

    for i, j in zip(src.flat, tgt.flat):
        glcm[i, j] += 1.0

    # Symmetric co-occurrence
    glcm_sym = glcm + glcm.T
    total = glcm_sym.sum()
    if total > 0:
        glcm_sym /= total
    return glcm_sym


def extract_haralick_glcm_features(glcm: np.ndarray, raw_image: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute Haralick texture features and first-order statistics following Ma et al. (2020).

    Features computed (Eqs. 4-19 in Ma et al.):
    - Contrast: sum_ij |i - j|^2 * P(i,j)
    - Dissimilarity: sum_ij |i - j| * P(i,j)
    - Homogeneity: sum_ij P(i,j) / (1 + |i - j|)
    - Energy (ASM): sum_ij P(i,j)^2
    - Correlation: sum_ij (i - mu_i)(j - mu_j) * P(i,j) / (sigma_i * sigma_j)
    - Cluster Shade: sum_ij (i + j - mu_i - mu_j)^3 * P(i,j)
    - First-order: Mean, Variance, Skewness, Kurtosis.
    """
    levels = glcm.shape[0]
    i_indices, j_indices = np.indices((levels, levels))

    # Marginal moments
    mu_i = np.sum(i_indices * glcm)
    mu_j = np.sum(j_indices * glcm)
    sigma_i = np.sqrt(np.sum(((i_indices - mu_i) ** 2) * glcm) + 1e-8)
    sigma_j = np.sqrt(np.sum(((j_indices - mu_j) ** 2) * glcm) + 1e-8)

    diff = np.abs(i_indices - j_indices)

    contrast = float(np.sum((diff ** 2) * glcm))
    dissimilarity = float(np.sum(diff * glcm))
    homogeneity = float(np.sum(glcm / (1.0 + diff)))
    energy = float(np.sum(glcm ** 2))
    correlation = float(np.sum((i_indices - mu_i) * (j_indices - mu_j) * glcm) / (sigma_i * sigma_j))
    cluster_shade = float(np.sum(((i_indices + j_indices - mu_i - mu_j) ** 3) * glcm))

    feats = {
        "glcm_contrast": round(contrast, 4),
        "glcm_dissimilarity": round(dissimilarity, 4),
        "glcm_homogeneity": round(homogeneity, 4),
        "glcm_energy": round(energy, 4),
        "glcm_correlation": round(correlation, 4),
        "glcm_cluster_shade": round(cluster_shade, 4),
    }

    if raw_image is not None:
        flat = raw_image.ravel().astype(float)
        mean_val = float(np.mean(flat))
        std_val = float(np.std(flat) + 1e-8)
        skew = float(np.mean(((flat - mean_val) / std_val) ** 3))
        kurt = float(np.mean(((flat - mean_val) / std_val) ** 4))
        feats["intensity_mean"] = round(mean_val, 3)
        feats["intensity_std"] = round(std_val, 3)
        feats["intensity_skewness"] = round(skew, 3)
        feats["intensity_kurtosis"] = round(kurt, 3)

    return feats
