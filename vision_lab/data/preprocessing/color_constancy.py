"""Color constancy — детерминированная инвариантность к освещению/устройству (ТЗ §7.4).

Применяется ОДИНАКОВО на train и inference: применение только на тесте создаёт
train/test-рассинхрон и обычно вредит модели, не видевшей скорректированных
изображений.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def shades_of_gray(img: np.ndarray, power: int = 6, eps: float = 1e-8) -> np.ndarray:
    """Shades-of-Gray (Finlayson & Trezzi, 2004).

    Оценивает освещение как Минковски-p среднее каналов, нормирует к единичной
    L2-норме и делит — сцена «серого мира» становится нейтральной.

    power: 1 = Gray-World, ->inf = max-RGB; 6 — дерматоскопический дефолт
    (устойчив к одиночным бликам). img: HWC float в [0, 1].
    """
    x = np.clip(img, 0.0, 1.0).astype(np.float32)
    illum = np.power(np.mean(np.power(x, power), axis=(0, 1)), 1.0 / power)  # (3,)
    illum = illum / (np.sqrt(np.sum(illum**2)) + eps)
    corrected = x / (illum[None, None, :] * np.sqrt(3.0) + eps)
    # float32 обязателен: np.mean/np.power промоутят в float64, а cvtColor-обвязка
    # albumentations (CLAHE, цветовые операции) принимает только uint8/float32.
    return np.clip(corrected, 0.0, 1.0).astype(np.float32)


class ColorConstancy:
    """Preprocessing-шаг для :class:`~vision_lab.data.manifest.ManifestDataset`."""

    def __init__(self, power: int = 6):
        self.power = power

    def __call__(self, image: np.ndarray, sample: Mapping) -> np.ndarray:
        return shades_of_gray(image, power=self.power)
