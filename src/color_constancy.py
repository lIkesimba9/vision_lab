"""Color constancy preprocessing for cross-device robustness (binary screening).

Dermatoscopes differ in white balance / illuminant; normalizing the illuminant
before the model removes that device-specific colour cast, which is a documented
lever for cross-device melanoma classification (Barata et al.; most ISIC winners).
This is the deterministic, principled version of the colour-invariance the hugeron
recipe got by accident (ToGray p_eff 0.0001->0.10).

    from src.color_constancy import shades_of_gray
    img_cc = shades_of_gray(img, power=6)   # img: HxWx3 float in [0,1]

Apply consistently at train AND inference — a test-only application creates a
train/test mismatch and typically hurts a model that never saw corrected images.
"""
from __future__ import annotations

import numpy as np


def shades_of_gray(img: np.ndarray, power: int = 6, eps: float = 1e-8) -> np.ndarray:
    """Shades-of-Gray colour constancy (Finlayson & Trezzi, 2004).

    Estimates the illuminant as the Minkowski-p mean of each channel, normalizes
    it to unit L2 norm, and divides it out so a gray-world scene maps to neutral.

    power: Minkowski norm. 1 = Gray-World, ->inf = max-RGB. 6 is the common
           dermoscopy default (robust to a few bright specular pixels).
    img:   HxWx3 float in [0, 1]. Returns the corrected image, clipped to [0, 1].
    """
    x = np.clip(img, 0.0, 1.0).astype(np.float32)
    illum = np.power(np.mean(np.power(x, power), axis=(0, 1)), 1.0 / power)  # [3]
    illum = illum / (np.sqrt(np.sum(illum ** 2)) + eps)                      # unit L2 norm
    corrected = x / (illum[None, None, :] * np.sqrt(3.0) + eps)
    # float32: np.mean/np.power above promote to float64, which albumentations'
    # cvtColor-based augs (CLAHE, colour ops) reject (they accept uint8 / float32 only).
    return np.clip(corrected, 0.0, 1.0).astype(np.float32)
